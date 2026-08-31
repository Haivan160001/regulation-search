"""
benchmark_reranker.py - Welcher Reranker passt zu meiner Sprache?

Die Empfehlungen im README wurden an UN R85 auf Deutsch und Englisch gemessen.
Fuer andere Sprachen (und andere Regelungen) gilt: selbst nachmessen. Dieses
Skript macht das reproduzierbar.

Ablauf
------
1. Regelungen wie gewohnt in Reg-Search indizieren.
2. Fragebogen anlegen - JSON-Liste mit erwarteter Fundstelle::

     [
       {"question": "Welche Bezugskraftstoffe sind vorgeschrieben?", "annex": "8"},
       {"question": "Wie ist die Nutzleistung definiert?",           "section": "2"},
       {"question": "Which reference fuels are specified?",          "annex": "8"}
     ]

   ``annex`` = erwarteter Anhang, ``section`` = erwarteter Hauptabschnitt im
   Hauptteil. Genau eines von beiden angeben. 20+ Fragen empfohlen - bei
   weniger sind die Unterschiede nicht von Rauschen zu trennen.
3. Laufen lassen::

     python tools/benchmark_reranker.py fragen.json
     python tools/benchmark_reranker.py fragen.json --models BAAI/bge-reranker-v2-m3 \\
         mixedbread-ai/mxbai-rerank-base-v2 --trust-remote-code

   Lizenzen vorher pruefen: die Vorgabemodelle und die oben genannten sind
   MIT bzw. Apache-2.0. jinaai/jina-reranker-v2-base-multilingual steht
   dagegen unter CC-BY-NC-4.0 und ist damit auf Forschung und Evaluation
   beschraenkt - im kommerziellen Betrieb scheidet es aus.

Kennzahlen
----------
MRR     Mean Reciprocal Rank der ersten korrekten Passage (hoeher = besser).
Top-N   Anteil Fragen, bei denen die korrekte Passage unter den ersten N liegt.
        **Die entscheidende Groesse** - nur diese Passagen sieht das LLM.
Spread  Mittlere Score-Spreizung. Nahe 0 heisst: der Cross-Encoder hat nichts
        unterschieden, seine Sortierung ist Zufall.

Die Kandidaten stammen fuer alle Modelle aus derselben Vektorsuche, der
Vergleich ist also fair. Die Zeile "ohne Reranker" ist die Kontrollgruppe:
Bleibt ein Modell darunter, verschlechtert es die Suche.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import rag_engine  # noqa: E402

DEFAULT_MODELS = [
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
]


def is_correct(metadata: dict, expected: dict) -> bool:
    """Passt die Fundstelle der Passage zur Erwartung im Fragebogen?"""
    if expected.get("annex"):
        return str(metadata.get("annex", "")) == str(expected["annex"])
    if expected.get("section"):
        top = str(metadata.get("paragraph", "")).split(".")[0]
        return not metadata.get("annex") and top == str(expected["section"])
    if expected.get("paragraph"):
        return str(metadata.get("paragraph", "")) == str(expected["paragraph"])
    return False


def load_reranker(model_name: str, trust_remote_code: bool):
    import torch
    from sentence_transformers import CrossEncoder

    kwargs: dict = {
        "max_length": config.RERANKER_MAX_LENGTH,
        "device": config.resolve_device(),
    }
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if config.USE_FP16 and torch.cuda.is_available():
        kwargs["model_kwargs"] = {"dtype": torch.float16}
    return CrossEncoder(model_name, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("questions", type=Path, help="JSON-Fragebogen")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--top-k", type=int, default=config.RETRIEVAL_TOP_K)
    parser.add_argument("--top-n", type=int, default=config.RERANK_TOP_N)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Modellcode von HuggingFace ausfuehren (fuer mxbai/jina/gte noetig)",
    )
    args = parser.parse_args()

    try:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Fragebogen nicht lesbar: {exc}")
        return 1

    engine = rag_engine.RegSearchEngine()
    total = engine.count_chunks()
    if total == 0:
        print("Der Index ist leer - bitte zuerst Dokumente indizieren.")
        return 1
    print(f"Index: {total} Chunks, {len(questions)} Fragen, Top-K {args.top_k}\n")

    # Kandidaten einmal bestimmen - identische Basis fuer jedes Modell
    pools: list[tuple[str, list, dict]] = []
    for entry in questions:
        question = entry.get("question", "").strip()
        if not question:
            continue
        docs = [d for d, _ in engine.vectorstore.similarity_search_with_score(
            question, k=args.top_k)]
        if not any(is_correct(d.metadata, entry) for d in docs):
            print(f"  uebersprungen (Soll nicht im Kandidatenpool): {question[:60]}")
            continue
        pools.append((question, docs, entry))

    if not pools:
        print("\nKeine auswertbare Frage. Stimmen die erwarteten Fundstellen?")
        return 1
    print(f"\nauswertbar: {len(pools)}/{len(questions)} Fragen\n")

    rows: list[tuple[str, float, float, float, float]] = []

    # Kontrollgruppe: reine Vektorsuche
    ranks = [
        next((i for i, d in enumerate(docs, 1) if is_correct(d.metadata, entry)), None)
        for _, docs, entry in pools
    ]
    rows.append((
        f"ohne Reranker ({config.EMBEDDING_MODEL})",
        sum(1 / r for r in ranks if r) / len(ranks),
        sum(1 for r in ranks if r and r <= args.top_n) / len(ranks),
        float("nan"),
        0.0,
    ))

    for model_name in args.models:
        print(f"lade {model_name} ...")
        try:
            model = load_reranker(model_name, args.trust_remote_code)
        except Exception as exc:
            hint = ""
            if "trust_remote_code" in str(exc).lower():
                hint = "  -> mit --trust-remote-code erneut starten"
            elif "einops" in str(exc).lower():
                hint = "  -> pip install einops"
            print(f"  FEHLER: {exc.__class__.__name__}: {str(exc)[:140]}{hint}\n")
            continue

        ranks, spreads = [], []
        started = time.perf_counter()
        for question, docs, entry in pools:
            scores = [
                float(s)
                for s in model.predict(
                    [(question, d.page_content) for d in docs],
                    batch_size=config.RERANKER_BATCH_SIZE,
                    show_progress_bar=False,
                )
            ]
            order = sorted(range(len(docs)), key=lambda i: -scores[i])
            ranks.append(
                next(
                    (p for p, i in enumerate(order, 1) if is_correct(docs[i].metadata, entry)),
                    None,
                )
            )
            spreads.append(max(scores) - min(scores))
        rows.append((
            model_name,
            sum(1 / r for r in ranks if r) / len(ranks),
            sum(1 for r in ranks if r and r <= args.top_n) / len(ranks),
            sum(spreads) / len(spreads),
            time.perf_counter() - started,
        ))
        del model
        rag_engine.free_vram()
        print("  fertig\n")

    width = max(len(name) for name, *_ in rows) + 2
    print("=" * (width + 34))
    print(f"{'Modell':<{width}}{'MRR':>7}{f'Top-{args.top_n}':>8}{'Spread':>9}{'Zeit':>10}")
    print("=" * (width + 34))
    for name, mrr, topn, spread, seconds in rows:
        spread_text = "-" if spread != spread else f"{spread:.3f}"  # NaN-Check
        print(f"{name:<{width}}{mrr:>7.3f}{topn:>8.0%}{spread_text:>9}{seconds:>9.1f}s")
    print("=" * (width + 34))
    print(
        f"\nTop-{args.top_n} ist die entscheidende Groesse: nur diese Passagen "
        "erreichen das LLM.\nSpread nahe 0 = das Modell trennt die Kandidaten "
        "nicht, seine Sortierung ist Zufall."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

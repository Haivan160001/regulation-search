"""
config.py - Zentrale Konfiguration fuer Reg-Search.

Alle Parameter koennen ueber Umgebungsvariablen (Praefix ``REG_SEARCH_``) oder
eine ``.env``-Datei im Projektverzeichnis ueberschrieben werden.
Die Defaults sind auf eine **NVIDIA-GPU mit 16 GB VRAM** abgestimmt:

    ~9.0 GB  LLM        (qwen2.5:14b, Q4_K_M, num_ctx 8192, via Ollama)
    ~1.2 GB  Embedding  (BAAI/bge-m3, fp16, max_seq_len 1024)
    ~0.8 GB  Reranker   (BAAI/bge-reranker-large, fp16, max_len 512)
    ------------------------------------------------------------------
    ~11 GB gesamt -> ausreichend Puffer fuer Aktivierungen & Desktop.

Bei weniger VRAM: ``REG_SEARCH_LLM_MODEL=llama3.1:8b`` setzen oder
``REG_SEARCH_DEVICE=cpu`` fuer Embedding/Reranker (langsamer, spart ~2 GB).
"""

from __future__ import annotations

import os
from pathlib import Path

# ChromaDB-Telemetrie abschalten, *bevor* chromadb irgendwo importiert wird.
# Das Settings-Flag allein genuegt nicht: chromadb 0.6.x protokolliert sonst bei
# jeder Collection-Operation "Failed to send telemetry event ... capture() takes
# 1 positional argument but 3 were given" (Signaturkonflikt mit neueren
# posthog-Versionen). Reg-Search sendet ohnehin nichts nach aussen.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

# --------------------------------------------------------------------------- #
# .env laden (ohne Zusatzabhaengigkeit - minimaler Parser)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Sehr einfacher .env-Loader: KEY=VALUE je Zeile, Rauten leiten Kommentare ein."""
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Bereits gesetzte Systemvariablen haben Vorrang vor der .env
            os.environ.setdefault(key, value)
    except OSError:
        pass


_load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------------------------------- #
# Typsichere Env-Helper
# --------------------------------------------------------------------------- #
def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# --------------------------------------------------------------------------- #
# Pfade
# --------------------------------------------------------------------------- #
DATA_DIR: Path = Path(env_str("REG_SEARCH_DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR: Path = DATA_DIR / "uploads"

#: ChromaDB laeuft vollstaendig *embedded* (wie eine lokale SQLite-Datei) -
#: kein Server, kein Docker. Der gesamte Index liegt in diesem Ordner.
CHROMA_DIR: Path = Path(env_str("REG_SEARCH_CHROMA_DIR", str(BASE_DIR / "chroma_db")))
CHROMA_COLLECTION: str = env_str("REG_SEARCH_COLLECTION", "unece_regulations")

#: Distanzmetrik der HNSW-Indizes (bge-m3-Vektoren sind L2-normalisiert).
CHROMA_DISTANCE: str = env_str("REG_SEARCH_CHROMA_DISTANCE", "cosine")


def ensure_directories() -> None:
    """Legt alle benoetigten Verzeichnisse an (idempotent)."""
    for directory in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Ollama / LLM
# --------------------------------------------------------------------------- #
OLLAMA_BASE_URL: str = env_str("REG_SEARCH_OLLAMA_URL", "http://localhost:11434")
LLM_MODEL: str = env_str("REG_SEARCH_LLM_MODEL", "qwen2.5:14b")

#: Vorschlaege fuer die Modellauswahl in der UI (Fallback, falls Ollama
#: gerade nicht erreichbar ist und die installierten Modelle unbekannt sind).
LLM_MODEL_SUGGESTIONS: tuple[str, ...] = (
    "qwen2.5:14b",  # Default - beste Qualitaet bei 16 GB VRAM
    "qwen2.5:14b-instruct-q4_K_M",
    "qwen3:14b",  # Reasoning-Modell; Gedankenkette wird herausgefiltert
    "llama3.1:8b",  # schlanke Alternative (~5 GB VRAM)
    "qwen2.5:7b",
    "mistral-nemo:12b",
    "gemma2:9b",
)

LLM_TEMPERATURE: float = env_float("REG_SEARCH_LLM_TEMPERATURE", 0.1)
LLM_TOP_P: float = env_float("REG_SEARCH_LLM_TOP_P", 0.9)

#: Kontextfenster. 8192 Tokens reichen fuer 5 Reranker-Treffer + Frage +
#: Antwort und halten den KV-Cache klein (VRAM!). 16384 erst ab ~24 GB.
LLM_NUM_CTX: int = env_int("REG_SEARCH_NUM_CTX", 8192)
LLM_NUM_PREDICT: int = env_int("REG_SEARCH_NUM_PREDICT", 1024)

#: Wie lange das Modell nach der letzten Anfrage im VRAM bleibt.
LLM_KEEP_ALIVE: str = env_str("REG_SEARCH_KEEP_ALIVE", "30m")

#: Timeout fuer Ollama-Requests in Sekunden (erstes Laden dauert laenger).
OLLAMA_TIMEOUT: int = env_int("REG_SEARCH_OLLAMA_TIMEOUT", 300)

# --------------------------------------------------------------------------- #
# Embedding & Reranking
# --------------------------------------------------------------------------- #
#: Mehrsprachiges Embedding-Modell - findet mit einer deutschen Frage auch
#: englische Absaetze. Fuer alle Sprachen dieselbe Wahl, kein Tuning noetig.
EMBEDDING_MODEL: str = env_str("REG_SEARCH_EMBEDDING_MODEL", "BAAI/bge-m3")

#: Cross-Encoder fuer Stufe 2. Die Sprachwahl ist hier relevant - siehe
#: Abschnitt "Reranker nach Sprache waehlen" im README und die Messwerte in
#: tools/benchmark_reranker.py. Gemessen an UN R85 (Deutsch/Englisch) ist
#: bge-reranker-large die beste Wahl fuer Deutsch (Top-5 100 %).
RERANKER_MODEL: str = env_str("REG_SEARCH_RERANKER_MODEL", "BAAI/bge-reranker-large")

#: Manche Reranker (mxbai-rerank-v2, jina-reranker-v2, gte-multilingual) liefern
#: eigenen Modellcode ueber den HuggingFace-Hub mit, der beim Laden *ausgefuehrt*
#: wird. Das ist bewusst nicht der Default - wer ein solches Modell einsetzt,
#: schaltet es hier explizit frei.
RERANKER_TRUST_REMOTE_CODE: bool = env_bool("REG_SEARCH_TRUST_REMOTE_CODE", False)

#: "auto" -> CUDA wenn verfuegbar, sonst CPU. Alternativ "cuda" / "cpu".
DEVICE: str = env_str("REG_SEARCH_DEVICE", "auto")

#: fp16 halbiert den VRAM-Bedarf der Encoder (nur auf CUDA aktiv).
USE_FP16: bool = env_bool("REG_SEARCH_FP16", True)

#: bge-m3 beherrscht 8192 Token - das kostet quadratisch Speicher. 1024 Token
#: entsprechen ~4000 Zeichen und decken unsere Chunk-Groesse komfortabel ab.
EMBEDDING_MAX_LENGTH: int = env_int("REG_SEARCH_EMBEDDING_MAX_LENGTH", 1024)
EMBEDDING_BATCH_SIZE: int = env_int("REG_SEARCH_EMBEDDING_BATCH_SIZE", 8)
EMBEDDING_NORMALIZE: bool = env_bool("REG_SEARCH_EMBEDDING_NORMALIZE", True)

RERANKER_MAX_LENGTH: int = env_int("REG_SEARCH_RERANKER_MAX_LENGTH", 512)
RERANKER_BATCH_SIZE: int = env_int("REG_SEARCH_RERANKER_BATCH_SIZE", 8)

# --------------------------------------------------------------------------- #
# Retrieval-Pipeline (2-stufig)
# --------------------------------------------------------------------------- #
#: Stufe 1 - Vektorsuche (optimiert auf Recall)
RETRIEVAL_TOP_K: int = env_int("REG_SEARCH_TOP_K", 20)

#: Stufe 2 - Cross-Encoder-Reranking (optimiert auf Precision)
RERANK_TOP_N: int = env_int("REG_SEARCH_TOP_N", 5)

#: Treffer mit Reranker-Score darunter werden verworfen.
#: sentence-transformers wendet bei Cross-Encodern mit einem Label per Default
#: eine Sigmoid-Funktion an - die Scores liegen also in [0, 1], nicht als rohe
#: Logits vor. Beobachtet: klar relevante Passagen > 0.1, irrelevante < 0.005.
#: 0.0 deaktiviert den Filter. Es bleibt immer mindestens der beste Treffer
#: erhalten, auch wenn alle unter dem Schwellwert liegen.
RERANK_MIN_SCORE: float = env_float("REG_SEARCH_RERANK_MIN_SCORE", 0.01)

#: Mindest-Spreizung der Reranker-Scores, damit deren Reihenfolge ueberhaupt
#: verwertbar ist. Liegen alle Kandidaten praktisch gleichauf, hat der
#: Cross-Encoder nichts unterschieden und seine Sortierung ist Rauschen -
#: gemessen an UN R85 passiert das bei deutschsprachigen Fragen auf englischen
#: Text regelmaessig (Spreizung 0.0002). Dann zaehlt die Reihenfolge der
#: mehrsprachigen Vektorsuche, die in diesen Faellen korrekt liegt.
RERANK_MIN_SPREAD: float = env_float("REG_SEARCH_RERANK_MIN_SPREAD", 0.01)

#: Zu einem Treffer wie "5.3.2" den uebergeordneten Absatz "5.3" mitgeben.
#: In UNECE-Texten stehen die Rahmenbedingungen im Elternabschnitt: die
#: 30-Minuten-Leistung wird in 5.3.2 gemessen, aber *womit* geprueft wird
#: (Aufbau nach Annex 6, DC-Quelle mit max. 5 % Spannungsabfall) steht in 5.3.
#: Der Cross-Encoder findet solche Rahmenabschnitte nicht - gemessen an
#: UN R85 landete 5.3 bei genau dieser Frage auf Rang 13 von 20, hinter einer
#: leeren Formularzeile. Die Ableitung ueber die Nummer ist dagegen exakt.
RETRIEVAL_PARENT_CONTEXT: bool = env_bool("REG_SEARCH_PARENT_CONTEXT", True)

#: Hoechstzahl zusaetzlicher Elternabschnitte pro Anfrage.
RETRIEVAL_MAX_PARENTS: int = env_int("REG_SEARCH_MAX_PARENTS", 2)

#: Anteil an Punkten, ab dem ein Chunk als leere Formularzeile gilt
#: ("Maximum 30 minutes power: ......... kW" aus den Pruefbericht-Mustern in
#: Annex 3a/3b). Solche Zeilen belegen nichts, konkurrieren aber im Reranking
#: mit echten Fundstellen. 0.0 deaktiviert die Aussortierung.
FORM_FIELD_DOT_RATIO: float = env_float("REG_SEARCH_FORM_FIELD_DOT_RATIO", 0.25)

#: Beim Indizieren aus dem Langtitel einen englischen Kurztitel erzeugen
#: ("UN Regulation No. 85 - Measurement of engine and electric drive net power").
#: UNECE-Titel sind bis zu 300 Zeichen lang und in einer Liste unbrauchbar.
#: Schlaegt die Erzeugung fehl, wird weiter der bisherige Titel angezeigt.
SHORT_TITLE_ENABLED: bool = env_bool("REG_SEARCH_SHORT_TITLE", True)

#: Laengenbegrenzung des Kurztitels in Zeichen.
SHORT_TITLE_MAX_CHARS: int = env_int("REG_SEARCH_SHORT_TITLE_MAX", 110)

#: Anzahl vorheriger Chat-Turns, die als Gespraechskontext mitgegeben werden.
CHAT_HISTORY_TURNS: int = env_int("REG_SEARCH_HISTORY_TURNS", 3)

# --------------------------------------------------------------------------- #
# Dokumentenverarbeitung / Structural Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE: int = env_int("REG_SEARCH_CHUNK_SIZE", 1200)  # Zeichen
CHUNK_OVERLAP: int = env_int("REG_SEARCH_CHUNK_OVERLAP", 150)  # Zeichen

#: Kleinere Struktureinheiten werden mit der naechsten zusammengefasst, damit
#: einzelne Nummern-Zeilen keine nutzlosen Mini-Chunks erzeugen.
MIN_CHUNK_CHARS: int = env_int("REG_SEARCH_MIN_CHUNK_CHARS", 280)

#: Kopf-/Fusszeilen, die auf mind. diesem Anteil der Seiten auftauchen,
#: werden als Boilerplate entfernt.
HEADER_FOOTER_RATIO: float = env_float("REG_SEARCH_HEADER_FOOTER_RATIO", 0.6)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx")
MAX_UPLOAD_MB: int = env_int("REG_SEARCH_MAX_UPLOAD_MB", 200)

#: Batchgroesse beim Schreiben in ChromaDB (begrenzt Peak-VRAM beim Embedden).
INGEST_BATCH_SIZE: int = env_int("REG_SEARCH_INGEST_BATCH_SIZE", 32)

# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT: str = (
    "You are Reg-Search, a meticulous regulatory analyst for UNECE/UN vehicle "
    "regulations (UN Regulations, GTRs, ECE/TRANS/WP.29 documents).\n"
    "Rules you MUST follow:\n"
    "1. Answer ONLY from the numbered context passages provided. Never invent "
    "requirements, numbers, limits, dates or paragraph references.\n"
    "2. Cite every factual statement with the passage number in square "
    "brackets, e.g. [1] or [2][3]. Name the exact paragraph/annex reference "
    "(e.g. 'Annex 3, para. 6.1.2') whenever the context provides it.\n"
    "3. If the context does not contain the answer, say so explicitly and name "
    "what would be needed - do not speculate.\n"
    "4. Preserve legal precision: distinguish 'shall' (mandatory), 'should' "
    "(recommended) and 'may' (optional). Keep technical values and units "
    "verbatim.\n"
    "5. Answer in the SAME language as the user's question (German question -> "
    "German answer). When you translate a provision, keep the key technical "
    "term from the source in parentheses, e.g. 'Nutzleistung (net power)' - "
    "a mistranslated term makes the answer unusable for homologation work.\n"
    "6. Structure longer answers with short paragraphs or bullet points."
)

#: Platzhalter: {context}, {question}
ANSWER_PROMPT: str = (
    "Context passages retrieved from the indexed regulations:\n"
    "--------------------------------------------------------\n"
    "{context}\n"
    "--------------------------------------------------------\n\n"
    "Question: {question}\n\n"
    "Answer using only the passages above and cite them as [n]."
)

#: Wird dem Prompt vorangestellt, wenn keine Passage den Relevanzschwellwert
#: erreicht hat. Ohne diesen Hinweis beantwortet das Modell die Frage aus
#: Allgemeinwissen - bei einer Regelung ist das die gefaehrlichste Fehlerart.
LOW_CONFIDENCE_NOTICE: str = (
    "WARNING - LOW RETRIEVAL CONFIDENCE: none of the passages below scored as "
    "clearly relevant to this question. They may be off-topic. Read them "
    "carefully. If they do not actually answer the question, say so plainly, "
    "name the annex/paragraph the reader should consult instead, and do NOT "
    "fill the gap with general knowledge or plausible-sounding lists."
)

#: Fallback-Antwort, wenn das Retrieval nichts Relevantes findet.
NO_CONTEXT_MESSAGE: str = (
    "In den indizierten Dokumenten wurde keine passende Stelle gefunden. "
    "Bitte die Frage praezisieren oder weitere Regelungen hochladen.\n\n"
    "*No matching passage found in the indexed documents.*"
)

# --------------------------------------------------------------------------- #
# App-Metadaten
# --------------------------------------------------------------------------- #
APP_NAME: str = "Reg-Search"
APP_ICON: str = "📘"
APP_VERSION: str = "1.0.0"

#: Startsprache der Oberflaeche ("de" oder "en"). Umschaltbar unter
#: Einstellungen; die Wahl landet als ?lang= in der URL und ueberlebt damit
#: einen Reload. Diese Variable legt nur fest, womit die App startet.
#:
#: Die *Antwort*-Sprache ist davon unabhaengig: SYSTEM_PROMPT weist das Modell
#: an, in der Sprache der Frage zu antworten.
APP_LANGUAGE: str = env_str("REG_SEARCH_LANG", "de")


def resolve_device(preferred: str | None = None) -> str:
    """
    Ermittelt das Torch-Device fuer die Encoder-Modelle.

    ``auto`` waehlt CUDA, sofern verfuegbar. Torch wird bewusst lazy importiert,
    damit ``config`` auch ohne installiertes Torch importierbar bleibt.
    """
    choice = (preferred or DEVICE or "auto").lower()
    if choice != "auto":
        return choice
    try:
        import torch  # lokal importiert: haelt config leichtgewichtig

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def summary() -> dict[str, object]:
    """Kompakte Konfigurationsuebersicht (fuer UI-Statusanzeige/Debugging)."""
    return {
        "app_version": APP_VERSION,
        "ollama_url": OLLAMA_BASE_URL,
        "llm_model": LLM_MODEL,
        "num_ctx": LLM_NUM_CTX,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "device": resolve_device(),
        "fp16": USE_FP16,
        "top_k": RETRIEVAL_TOP_K,
        "top_n": RERANK_TOP_N,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chroma_dir": str(CHROMA_DIR),
        "collection": CHROMA_COLLECTION,
    }

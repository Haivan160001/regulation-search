"""
Tests fuer die Trefferauswahl der Suche.

Laeuft ohne pytest::

    python tests/test_rag_engine.py

Getestet wird die Direktabfrage ausdruecklich genannter Fundstellen. Dichte
Embeddings kodieren Bedeutung, keine Bezeichner - "5.2.3.3.3" und "5.2.3.3.6"
liegen im Vektorraum praktisch aufeinander. Ohne diese Stufe verfehlt die Suche
genau die Klausel, nach der gefragt wurde.

Die Tests kommen ohne Modelle und ohne GPU aus: geprueft werden die
Referenzerkennung und der Metadatenfilter, nicht das Embedding.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Niemals den Produktivindex anfassen - siehe README, Abschnitt Konfiguration.
os.environ.setdefault("REG_SEARCH_CHROMA_DIR", str(Path(tempfile.gettempdir()) / "reg_search_test"))
os.environ.setdefault("REG_SEARCH_COLLECTION", "test")

import rag_engine as rag  # noqa: E402


def test_parse_citation_reference() -> None:
    """Nur ausdrueckliche Fundstellen - Messwerte duerfen nichts ausloesen."""
    cases = {
        "Ist R85 \u00a75.2.3.3.3. relevant fuer BEVs?": (["5.2.3.3.3"], ""),
        "Was steht in Annex 5, Para. 5.4.2.2?": (["5.4.2.2"], "5"),
        "Absatz 5.2.3 und Ziffer 6.1.2 vergleichen": (["5.2.3", "6.1.2"], ""),
        "Anhang 3a Paragraph 1.1": (["1.1"], "3a"),
        # Keine Referenz: Messwerte, Grenzwerte, einstufige Nummern
        "Wie ist der Engine factor definiert?": ([], ""),
        "Gilt ein Wobbe-Index von 52.6 MJm-3?": ([], ""),
        "Was steht in \u00a75?": ([], ""),
        "fm betraegt 0.3 bei qc unter 40": ([], ""),
    }
    for query, expected in cases.items():
        assert rag.parse_citation_reference(query) == expected, (
            f"{query!r} -> {rag.parse_citation_reference(query)}, erwartet {expected}"
        )


class _FakeCollection:
    """Minimaler Ersatz fuer die Chroma-Collection: filtert nur nach paragraph."""

    def __init__(self, rows: list[tuple[str, dict]]) -> None:
        self.rows = rows
        self.last_where: dict | None = None

    def get(self, where=None, limit=None, include=None):  # noqa: ANN001
        self.last_where = where
        criteria = dict(where or {})
        for part in criteria.pop("$and", []):
            criteria.update(part)
        wanted = criteria.get("paragraph")
        if isinstance(wanted, dict):
            wanted = set(wanted.get("$in", []))
        elif wanted is not None:
            wanted = {wanted}
        rows = [r for r in self.rows if wanted is None or r[1].get("paragraph") in wanted]
        rows = rows[: limit or len(rows)]
        return {"documents": [r[0] for r in rows], "metadatas": [r[1] for r in rows]}


class _FakeClient:
    """``collection`` ist eine Property und holt sich die Collection vom Client."""

    def __init__(self, collection: _FakeCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, name: str):  # noqa: ANN201
        return self.collection


def _engine_with(rows: list[tuple[str, dict]]) -> rag.RegSearchEngine:
    """Engine ohne Modelle - gebraucht wird nur der Zugriff auf die Collection."""
    engine = rag.RegSearchEngine.__new__(rag.RegSearchEngine)
    engine.chroma_client = _FakeClient(_FakeCollection(rows))
    engine.collection_name = "test"
    return engine


def test_lookup_reference_finds_exact_clause() -> None:
    """Die genannte Klausel wird direkt geholt und als Direkttreffer markiert."""
    rows = [
        ("Text 5.2.3.3.3", {"paragraph": "5.2.3.3.3", "chunk_index": 7, "chunk_id": "a:7"}),
        ("Text 5.2.3.3.6", {"paragraph": "5.2.3.3.6", "chunk_index": 9, "chunk_id": "a:9"}),
    ]
    engine = _engine_with(rows)

    hits = engine._lookup_reference("Ist R85 \u00a75.2.3.3.3. relevant fuer BEVs?", None, 3)
    assert len(hits) == 1, f"Erwartet 1 Direkttreffer, erhalten {len(hits)}"
    assert hits[0].metadata["paragraph"] == "5.2.3.3.3"
    assert hits[0].exact_match is True, "Direkttreffer nicht gekennzeichnet"

    # Ohne Referenz in der Frage darf gar nicht erst abgefragt werden.
    assert engine._lookup_reference("Wie wird die Nutzleistung gemessen?", None, 3) == []


def test_lookup_reference_combines_filters() -> None:
    """Annex und Quellenfilter werden mit der Nummer zu einem $and verknuepft."""
    engine = _engine_with([("T", {"paragraph": "1.1", "chunk_index": 0, "chunk_id": "a:0"})])
    engine._lookup_reference("Anhang 3a Paragraph 1.1", {"source": "R085r1e.pdf"}, 3)

    where = engine.chroma_client.collection.last_where
    assert "$and" in where, f"Erwartet $and-Verknuepfung, erhalten: {where}"
    conditions = where["$and"]
    assert {"paragraph": "1.1"} in conditions
    assert {"annex": "3a"} in conditions
    assert {"source": "R085r1e.pdf"} in conditions


def test_is_form_field() -> None:
    """Leere Formularzeilen aus den Pruefbericht-Mustern erkennen."""
    form = "Maximum 30 minutes power: ......................................... kW"
    assert rag.is_form_field(form), "Formularzeile nicht erkannt"
    assert rag.is_form_field("   ") is True

    real = (
        "The electric drive train shall be supplied from a DC voltage source "
        "with a maximum voltage drop of 5 per cent depending on time and current."
    )
    assert not rag.is_form_field(real), "Echter Vorschriftentext faelschlich verworfen"
    assert not rag.is_form_field("fm = 0.036 qc - 1.14"), "Formel faelschlich verworfen"


def test_parent_sections_adds_framework_clause() -> None:
    """Zu einem Treffer 5.3.2 kommt der uebergeordnete Abschnitt 5.3 dazu."""
    rows = [
        ("Aufbau nach Annex 6, DC-Quelle", {"paragraph": "5.3", "chunk_id": "a:1"}),
        ("Annex-7-Text", {"paragraph": "5.3", "annex": "7", "chunk_id": "a:2"}),
    ]
    engine = _engine_with(rows)
    hit = rag.RetrievedChunk(text="Messung", metadata={"paragraph": "5.3.2", "chunk_id": "a:9"})

    parents = engine._parent_sections([hit], None)
    assert len(parents) == 1, f"Erwartet genau einen Elternabschnitt, erhalten {len(parents)}"
    assert parents[0].metadata["chunk_id"] == "a:1", (
        "Der Abschnitt 5.3 aus Annex 7 darf nicht als Elternteil des Hauptteils gelten"
    )
    assert parents[0].parent_context is True, "Elternabschnitt nicht gekennzeichnet"


def test_parent_sections_skips_duplicates_and_top_level() -> None:
    """Schon vorhandene Eltern und einstufige Nummern loesen nichts aus."""
    engine = _engine_with([("5.3-Text", {"paragraph": "5.3", "chunk_id": "a:1"})])
    hits = [
        rag.RetrievedChunk(text="Messung", metadata={"paragraph": "5.3.2", "chunk_id": "a:9"}),
        rag.RetrievedChunk(text="Rahmen", metadata={"paragraph": "5.3", "chunk_id": "a:1"}),
    ]
    assert engine._parent_sections(hits, None) == [], "Vorhandener Elternabschnitt doppelt geladen"

    top = [rag.RetrievedChunk(text="T", metadata={"paragraph": "5", "chunk_id": "a:5"})]
    assert engine._parent_sections(top, None) == [], "Einstufige Nummer hat kein Elternteil"


def test_clean_short_title() -> None:
    """Beiwerk der Modellantwort entfernen - Erklaerungen ganz verwerfen."""
    assert rag.clean_short_title("Subject: Braking", 90) == "Braking"
    assert rag.clean_short_title('"Cyber security"', 90) == "Cyber security"
    thinking = "<think>ueberlege</think>" + chr(10) + "Braking"
    assert rag.clean_short_title(thinking, 90) == "Braking"
    # Eine Erlaeuterung statt eines Titels ist unbrauchbar
    assert rag.clean_short_title("Here is the short title you asked for " * 6, 90) == ""
    assert rag.clean_short_title("", 90) == ""
    # Gekuerzt wird an der Wortgrenze, nicht mitten im Wort
    assert rag.clean_short_title("Measurement of net power and maximum power", 20) == "Measurement of net"


class _FakeOllama:
    """Ollama-Ersatz: liefert eine feste Antwort oder wirft."""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error

    def chat(self, model, messages, options=None, think=None):  # noqa: ANN001
        if self.error:
            raise self.error
        return {"message": {"content": self.content}}


def _engine_with_llm(client: _FakeOllama) -> rag.RegSearchEngine:
    engine = rag.RegSearchEngine.__new__(rag.RegSearchEngine)
    engine.ollama = client
    engine.llm_model = "testmodell"
    return engine


def test_short_title_prefixes_identifier() -> None:
    """Die Kennung stammt aus den Metadaten, nicht aus dem Modell."""
    engine = _engine_with_llm(_FakeOllama("Net power measurement"))
    title = engine.short_title("Regulation No 85 ... net power", regulation="UN Regulation No. 85")
    assert title == "UN Regulation No. 85 — Net power measurement", repr(title)

    # Ohne bekannte Kennung bleibt das Sachthema allein stehen - kein "No. -"
    assert engine.short_title("Global technical regulation on EVS") == "Net power measurement"


def test_short_title_survives_ollama_failure() -> None:
    """Ein Upload darf nicht scheitern, weil Ollama nicht laeuft."""
    engine = _engine_with_llm(_FakeOllama(error=ConnectionError("Ollama aus")))
    assert engine.short_title("Irgendein Titel", regulation="UN R85") == ""

    engine = _engine_with_llm(_FakeOllama(""))
    assert engine.short_title("Irgendein Titel", regulation="UN R85") == ""
    assert engine.short_title("") == ""


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {name}: {exc.__class__.__name__}: {exc}")
    print("\n" + ("Alle Tests bestanden." if not failures else f"{failures} Test(s) fehlgeschlagen."))
    sys.exit(1 if failures else 0)

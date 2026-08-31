"""
Tests fuer das Structural Chunking.

Laeuft ohne pytest::

    python tests/test_document_processor.py

Getestet wird der kritische Teil der Pipeline: Erkennt der Parser die
UNECE-Hierarchie - und faellt er *nicht* auf Querverweise im Fliesstext
herein ("... specified in Annex 1 to this Regulation;")? Eine falsche
Fundstelle waere schlimmer als gar keine.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import document_processor as dp  # noqa: E402

# Zwei Seiten im typischen UNECE-Layout - inklusive der beiden Fallen:
# ein Annex-Querverweis am Zeilenanfang und Messwerte in einer Umbruchzeile.
SAMPLE_PAGES = [
    """E/ECE/324/Rev.2/Add.154
Regulation No. 155
Uniform provisions concerning cyber security and cyber security management
Incorporating the 01 series of amendments

3. Application for approval
3.1. The application for approval shall be submitted by the vehicle
manufacturer or by their duly accredited representative.

3.2. It shall be accompanied by the documents mentioned below in triplicate:
3.2.1. a description of the vehicle type with regard to the items specified in
Annex 1 to this Regulation;

5.1.2. The vehicle manufacturer shall demonstrate that the processes used
within their Cyber Security Management System ensure security is adequately
considered, including the risks and mitigations listed in Annex 5.
""",
    """Annex 3
Model of the communication concerning the approval of a vehicle type

1. General
The communication shall be issued using the format set out in this annex.

Appendix 1
Test procedure for the verification of the CSMS

1. The test shall be carried out at an ambient temperature of
23 +- 5 degrees C. The measured value shall not exceed
5.0 mg/km according to the procedure described in Annex 4 to this Regulation.
""",
]


def _build_sample_pdf(target: Path, pages: list[str] | None = None) -> Path:
    import fitz  # PyMuPDF

    document = fitz.open()
    for text in pages if pages is not None else SAMPLE_PAGES:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(56, 56, 540, 780), text, fontsize=10, fontname="helv")
    document.save(str(target))
    document.close()
    return target


def test_structural_chunking() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        pdf = _build_sample_pdf(Path(workdir) / "R155-sample.pdf")
        result = dp.process_document(pdf)

    chunks = result.chunks
    assert chunks, "Es wurden keine Chunks erzeugt"

    # --- Dokument-Metadaten -------------------------------------------------
    assert result.metadata["regulation"] == "UN Regulation No. 155"
    assert result.metadata["un_symbol"].startswith("E/ECE/324")
    assert result.metadata["amendment_series"] == "01 series of amendments"

    def by_paragraph(number: str) -> list[dp.Chunk]:
        return [c for c in chunks if c.metadata.get("paragraph") == number]

    # --- Querverweise duerfen den Annex-Zustand nicht setzen ----------------
    for number in ("3.2.1", "5.1.2"):
        for chunk in by_paragraph(number):
            assert not chunk.metadata.get("annex"), (
                f"Para. {number} wurde faelschlich Annex "
                f"{chunk.metadata['annex']} zugeordnet (Querverweis missdeutet)"
            )

    # --- Echte Ueberschriften muessen erkannt werden -------------------------
    annexes = {c.metadata.get("annex") for c in chunks if c.metadata.get("annex")}
    assert annexes == {"3"}, f"Erwartet Annex 3, erkannt: {annexes}"
    appendices = {c.metadata.get("appendix") for c in chunks if c.metadata.get("appendix")}
    assert appendices == {"1"}, f"Erwartet Appendix 1, erkannt: {appendices}"

    # --- Messwerte sind keine Paragraphen -----------------------------------
    numbers = {c.metadata.get("paragraph") for c in chunks}
    assert "23" not in numbers and "5.0" not in numbers, (
        f"Messwert als Paragraph interpretiert: {sorted(n for n in numbers if n)}"
    )

    # --- Zitierfaehigkeit ---------------------------------------------------
    annex_chunks = [c for c in chunks if c.metadata.get("annex") == "3"]
    assert annex_chunks
    assert "Annex 3" in annex_chunks[0].metadata["citation"]
    assert "UN R155" in annex_chunks[0].metadata["citation"]

    # --- Chroma-Kompatibilitaet: nur skalare Metadaten ----------------------
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            assert isinstance(value, (str, int, float, bool)), (
                f"Metadatum '{key}' ist {type(value).__name__} - ChromaDB "
                "akzeptiert nur str/int/float/bool"
            )
        assert chunk.metadata["chunk_id"], "chunk_id fehlt"


def test_split_text_respects_boundaries() -> None:
    text = "Erster Satz. " * 200
    parts = dp._split_text(text, size=400, overlap=50)
    assert len(parts) > 1
    assert all(len(part) <= 400 + 50 for part in parts), "Chunk deutlich zu gross"


def test_strip_thinking() -> None:
    """Gedankenketten von Reasoning-Modellen duerfen nicht in der Antwort landen.

    Die Tags kommen tokenweise an und werden dabei zerschnitten - der Filter
    muss bei jeder Token-Groesse dasselbe Ergebnis liefern.
    """
    from rag_engine import strip_thinking  # importiert nur config, kein torch

    cases = [
        ("<think>Ich pruefe Annex 3</think>Die Antwort lautet [1].", "Die Antwort lautet [1]."),
        ("Kein Thinking hier.", "Kein Thinking hier."),
        ("Vor<think>mitten</think>nach", "Vornach"),
        ("<think>ohne Abschluss", ""),
        ("<think>x</think><think>y</think>Z", "Z"),
        ("5 < 6 und 7 > 3", "5 < 6 und 7 > 3"),
        ("a<thi", "a<thi"),  # Fragment am Stream-Ende bleibt erhalten
    ]
    for raw, expected in cases:
        for size in (1, 2, 3, 7, 999):
            tokens = (raw[i : i + size] for i in range(0, len(raw), size))
            got = "".join(strip_thinking(tokens))
            assert got == expected, f"Tokengroesse {size}: {raw!r} -> {got!r} statt {expected!r}"


def test_sanitize_metadata() -> None:
    cleaned = dp.sanitize_metadata(
        {"a": None, "b": "", "c": ["x", "y"], "d": 3, "e": True, "f": "text"}
    )
    assert cleaned == {"c": "x, y", "d": 3, "e": True, "f": "text"}


# Nachbau des Layouts aus UN R85, Annex 5, Para. 5.4.2.2: Nummern auf eigener
# Zeile, ein Formel-Exponent als Zahlenzeile und zwei Saetze, deren Grenzwert
# erst in der Folgezeile steht.
FORMULA_PAGE = """Annex 5
Method for measuring internal combustion engine net power

5.4.2.1.1.
Naturally aspirated and mechanically supercharged engines

7.0

5.4.2.2.
Engine factor fm

fm is a function of qc (fuel flow corrected) as follows:

fm = 0.036 qc - 1.14

This formula is valid for a value interval of qc included between
40 mg/(l.cycle) and 65 mg/(l.cycle.)

For qc values lower than 40 mg/(l.cycle), a constant value of fm equal to
0.3 (fm =  0.3) will be taken.

For qc values higher than 65 mg/(l.cycle), a constant value of fm equal to
1.2 (fm =  1.2) will be taken.
"""


def test_values_do_not_open_paragraphs() -> None:
    """
    Grenzwerte und Formelfragmente duerfen keinen Abschnitt eroeffnen.

    Regressionstest zu einem beobachteten Fehler: "0.3" und "7.0" galten als
    Paragraphennummern. Der Satz "... a constant value of fm equal to" brach
    dadurch vor seinem Grenzwert ab, das Sprachmodell ergaenzte eine falsche
    Zahl - und die Fundstelle dazu sah weiterhin korrekt aus.
    """
    with tempfile.TemporaryDirectory() as workdir:
        pdf = _build_sample_pdf(Path(workdir) / "R085-formula.pdf", [FORMULA_PAGE])
        chunks = dp.process_document(pdf).chunks

    numbers = {c.metadata.get("paragraph") for c in chunks if c.metadata.get("paragraph")}
    for phantom in ("7.0", "0.3", "1.2"):
        assert phantom not in numbers, (
            f"'{phantom}' wurde als Paragraph gewertet - erkannt: {sorted(numbers)}"
        )

    # Der Absatz muss vollstaendig in *einem* Chunk stehen, sonst fehlen die
    # Grenzwerte im Kontext des Sprachmodells.
    formula = [c for c in chunks if "0.036" in c.text]
    assert len(formula) == 1, f"Formel in {len(formula)} Chunks statt in einem"
    text = formula[0].text
    for value in ("0.3", "1.2", "40 mg", "65 mg"):
        assert value in text, f"'{value}' fehlt im Chunk zur fm-Formel"

    # ... und unter der Fundstelle, unter der er auch wirklich steht.
    citation = formula[0].metadata["citation"]
    assert "Para. 5.4.2.2" in citation, f"Falsche Fundstelle: {citation}"


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
        except Exception as exc:  # z. B. fehlendes PyMuPDF
            failures += 1
            print(f"ERROR {name}: {exc.__class__.__name__}: {exc}")
    print("\n" + ("Alle Tests bestanden." if not failures else f"{failures} Test(s) fehlgeschlagen."))
    sys.exit(1 if failures else 0)

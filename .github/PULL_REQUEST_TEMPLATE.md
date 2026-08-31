## Was ändert dieser PR?

<!-- Kurze Beschreibung + verlinktes Issue, z. B. "Closes #12" -->

## Art der Änderung

- [ ] Bugfix
- [ ] Neues Feature
- [ ] Parser-Erweiterung (neue Struktur-Muster)
- [ ] Dokumentation
- [ ] Refactoring / Performance

## Checkliste

- [ ] `black --line-length 100 .` und `ruff check .` laufen sauber
- [ ] `python tests/test_document_processor.py` läuft durch
- [ ] Neue Parser-Logik hat einen eigenen Testfall
- [ ] Neue Parameter liegen in `config.py` und sind per Env überschreibbar
- [ ] Keine PDFs, Modellgewichte oder `chroma_db/`-Inhalte im Commit
- [ ] Modulgrenzen eingehalten (`document_processor.py` ohne Streamlit/LangChain,
      `rag_engine.py` ohne Streamlit)

## Getestet mit

- Regelung(en): <!-- z. B. UN R155, UN R79 -->
- Modell: <!-- z. B. qwen2.5:14b -->
- GPU / VRAM:

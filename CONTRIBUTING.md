# Beitragen zu Reg-Search / Contributing

Danke für dein Interesse! 🎉 Reg-Search lebt von Beiträgen aus der
Fahrzeug-Homologation, der Normung und der NLP-Community.
*Contributions are welcome — German or English, both are fine.*

## Wie du beitragen kannst

| Art                    | Vorgehen                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| 🐛 **Bug melden**       | Issue mit Vorlage *Bug report* — bitte Regelung, Seite/Absatz und Traceback angeben |
| 💡 **Feature vorschlagen** | Issue mit Vorlage *Feature request* — gern mit konkretem Anwendungsfall |
| 📄 **Parser verbessern** | Neue Struktur-Muster (EU-Verordnungen, FMVSS, ISO) in `document_processor.py` |
| 📝 **Doku**             | Auch Tippfehler-PRs sind willkommen                                       |

Für Fragen zur Installation reicht ein Issue — kein Code nötig.

## Entwicklungsumgebung

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate                          # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install black ruff pytest        # Dev-Werkzeuge (nicht in requirements.txt)
```

**Voraussetzung: Python 3.10+** (die Codebasis nutzt `X | None`-Typannotationen).

## Vor dem Pull Request

```bash
black --line-length 100 .
ruff check .
python tests/test_document_processor.py     # oder: pytest tests/
```

Checkliste:

- [ ] Code ist formatiert (`black`, Zeilenlänge 100) und `ruff` ist sauber
- [ ] Tests laufen durch; neue Parser-Logik hat einen **eigenen Testfall**
- [ ] Neue Konfiguration liegt in `config.py` (env-überschreibbar) — keine Magic Numbers im Code
- [ ] Kommentare erklären das *Warum*, nicht das *Was*; Docstrings auf Deutsch oder Englisch, aber konsistent zur Datei
- [ ] Keine Modellgewichte, PDFs oder `chroma_db/`-Inhalte im Commit

## Architekturregeln

Damit die Module austauschbar bleiben:

- `document_processor.py` kennt **kein** Streamlit und **kein** LangChain — reines
  Parsing, damit es eigenständig testbar bleibt.
- `rag_engine.py` kennt **kein** Streamlit — das Caching passiert in `app.py`
  über `@st.cache_resource`.
- `app.py` enthält keine Retrieval-Logik, nur Darstellung und Zustand.
- Teure Objekte (Embedding-Modell, Reranker, ChromaDB-Client) werden **einmal**
  erzeugt und gecacht — ein Regressionsfehler hier kostet pro Chat-Nachricht
  Gigabyte an Ladezeit.

## Beiträge zum Struktur-Parser

Der heikelste Teil ist die Erkennung der Fundstellen. Bitte beachte:

> Eine **falsche** Fundstelle ist schlimmer als eine **grobe**.

Im Zweifel bleibt Text lieber beim vorherigen Paragraphen, statt eine neue,
womöglich erfundene Nummer zu erzeugen. Typische Fallen (alle im Test abgedeckt):

- Querverweise am Zeilenanfang: `… specified in Annex 1 to this Regulation;`
- Messwerte in Umbruchzeilen: `23 ± 5 degrees C.` ist kein Paragraph „23"
- Inhaltsverzeichnisse: `5.1 Anforderungen ........... 12`
- Kopf-/Fußzeilen, die auf jeder Seite wiederkehren

Neue Muster bitte mit einem Beispiel aus einer **öffentlichen** UNECE-Regelung
belegen (Regelungsnummer + Absatz genügt, keine Dateien beilegen).

## Commits & Branches

- Branch: `feature/<kurz>`, `fix/<kurz>` oder `docs/<kurz>`
- Commit-Nachrichten im Imperativ: `fix: Annex-Querverweis nicht als Überschrift werten`
- Ein Thema pro PR — das erleichtert das Review erheblich

## Verhaltenskodex

Sei freundlich und sachlich. Wir diskutieren Code und Regelungen, keine Personen.

## Lizenz

Mit deinem Beitrag stimmst du zu, dass er unter der [MIT-Lizenz](LICENSE)
veröffentlicht wird.

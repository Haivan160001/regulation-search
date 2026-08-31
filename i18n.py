"""
i18n.py - Sprachen der Oberflaeche (Deutsch / Englisch).

Verwendung
----------
    import i18n
    from i18n import t

    i18n.init()                      # einmal pro Skriptlauf, vor der ersten Ausgabe
    st.write(t("nav.chat"))
    st.write(t("chat.sources_expander", count=5))

Ablage der Sprachwahl
---------------------
Die gewaehlte Sprache steht in ``st.session_state["lang"]`` und zusaetzlich als
``?lang=`` in der URL. Der Umweg ueber die URL ist Absicht: ohne ihn faellt die
App bei jedem Reload auf die Startsprache zurueck, weil Streamlit den
Session-State pro Browser-Sitzung haelt. Die Startsprache selbst kommt aus
``config.APP_LANGUAGE`` (env: ``REG_SEARCH_LANG``).

Nicht betroffen: die Sprache der *Antworten*. ``config.SYSTEM_PROMPT`` weist
das Modell an, in der Sprache der Frage zu antworten - eine englische Frage
wird also auch bei deutscher Oberflaeche englisch beantwortet.

Neue Texte
----------
Jeder Eintrag in ``STRINGS`` traegt beide Sprachen. Fehlt ein Schluessel,
liefert ``t()`` den Schluessel selbst zurueck - sichtbar, aber nicht kaputt.
``tools/check_i18n.py`` prueft Vollstaendigkeit und findet ungenutzte Eintraege.
"""

from __future__ import annotations

import streamlit as st

import config

#: Auswahl in den Einstellungen: Code -> Eigenbezeichnung.
LANGUAGES: dict[str, str] = {"de": "Deutsch", "en": "English"}

FALLBACK = "de"

STRINGS: dict[str, dict[str, str]] = {
    # -- Navigation ---------------------------------------------------------
    "app.tagline": {
        "de": "Lokale RAG-Suche fuer UNECE-Regelungen",
        "en": "Local RAG search for UNECE regulations",
    },
    "nav.chat": {"de": "Chat", "en": "Chat"},
    "nav.documents": {"de": "Dokumente", "en": "Documents"},
    "nav.system": {"de": "System", "en": "System"},
    "nav.settings": {"de": "Einstellungen", "en": "Settings"},
    "nav.footer": {
        "de": "Version {version} &middot; laeuft lokal",
        "en": "Version {version} &middot; runs locally",
    },
    # -- Chat ---------------------------------------------------------------
    "chat.title": {"de": "Chat", "en": "Chat"},
    "chat.model": {"de": "Modell", "en": "Model"},
    "chat.model_help": {
        "de": "Ueber Ollama bereitgestellte Modelle - zur Auswahl stehen nur die "
        "bereits installierten. Nicht installierte Modelle muessen zuerst mit "
        "'ollama pull' geladen werden. Neue Modelle koennen in den "
        "System-Einstellungen (System > Status) geladen werden.",
        "en": "Models served by Ollama - only the ones already installed are "
        "offered here. Models that are not installed must be pulled first with "
        "'ollama pull'. New models can be added in the system settings "
        "(System > Status).",
    },
    "chat.sources": {"de": "Quellen", "en": "Sources"},
    "chat.sources_all": {"de": "alle Dokumente", "en": "all documents"},
    "chat.sources_help": {
        "de": "Leer = alle Dokumente durchsuchen.",
        "en": "Empty = search all documents.",
    },
    "chat.clear": {"de": "Verlauf leeren", "en": "Clear history"},
    "chat.model_missing": {
        "de": "Modell nicht installiert - 'ollama pull {model}'",
        "en": "Model not installed - 'ollama pull {model}'",
    },
    "chat.ollama_down": {
        "de": "Ollama nicht erreichbar - siehe System > Status",
        "en": "Ollama unreachable - see System > Status",
    },
    "chat.empty_index": {
        "de": "Die Wissensbasis ist leer. Unter *Dokumente > Bibliothek* eine "
        "UNECE-Regelung (PDF/DOCX) hochladen und indizieren.",
        "en": "The knowledge base is empty. Upload and index a UNECE regulation "
        "(PDF/DOCX) under *Documents > Library*.",
    },
    "chat.examples": {"de": "Beispielfragen", "en": "Example questions"},
    "chat.example_1": {
        "de": "Welche Anforderungen gelten fuer das Cyber Security Management System?",
        "en": "What requirements apply to the Cyber Security Management System?",
    },
    "chat.example_2": {
        "de": "Was muss der Antrag auf Genehmigung enthalten?",
        "en": "What must the application for approval contain?",
    },
    "chat.example_3": {
        "de": "Welche Pruefbedingungen sind in Annex 3 festgelegt?",
        "en": "Which test conditions are specified in Annex 3?",
    },
    "chat.input_placeholder": {
        "de": "Frage zu den indizierten Regelungen stellen ...",
        "en": "Ask a question about the indexed regulations ...",
    },
    "chat.retrieving": {
        "de": "Suche relevante Passagen ...",
        "en": "Searching for relevant passages ...",
    },
    "chat.vector_search": {
        "de": "Vektorsuche (Top-{top_k}) in ChromaDB ...",
        "en": "Vector search (top {top_k}) in ChromaDB ...",
    },
    "chat.reranking": {
        "de": "Reranking mit Cross-Encoder -> {count} Passagen ausgewaehlt.",
        "en": "Reranking with cross-encoder -> {count} passages selected.",
    },
    "chat.below_threshold": {
        "de": "Weitere Kandidaten lagen unter dem Relevanz-Schwellwert "
        "({threshold}) und wurden verworfen.",
        "en": "Further candidates fell below the relevance threshold "
        "({threshold}) and were discarded.",
    },
    "chat.hits": {"de": "{count} Fundstellen", "en": "{count} passages"},
    "chat.low_confidence": {
        "de": "**Geringe Trefferkonfidenz** - keine Passage wurde als klar "
        "relevant bewertet. Die Antwort ist mit besonderer Vorsicht zu "
        "pruefen; ggf. die Frage mit Begriffen aus der Regelung formulieren "
        "(z. B. englische Fachbegriffe).",
        "en": "**Low retrieval confidence** - no passage was rated clearly "
        "relevant. Treat the answer with particular caution; consider "
        "rephrasing the question using terms from the regulation itself.",
    },
    "chat.footer": {
        "de": "Modell: `{model}` · Top-K {top_k} → Top-N {top_n} · Reranker: `{reranker}`",
        "en": "Model: `{model}` · top-K {top_k} → top-N {top_n} · reranker: `{reranker}`",
    },
    "chat.error": {"de": "Fehler: {message}", "en": "Error: {message}"},
    "sources.expander": {
        "de": "Quellen ({count}) - exakte Fundstellen anzeigen",
        "en": "Sources ({count}) - show exact passages",
    },
    "sources.page": {"de": "S. {page}", "en": "p. {page}"},
    "sources.pages": {"de": "S. {first}-{last}", "en": "pp. {first}-{last}"},
    "sources.exact": {"de": "Direkttreffer", "en": "Direct match"},
    "sources.parent": {"de": "Elternabschnitt", "en": "Parent section"},
    "sources.rerank": {"de": "Rerank {score}", "en": "Rerank {score}"},
    "sources.vector": {"de": "Vektor {score}", "en": "Vector {score}"},
    # -- Dokumente ----------------------------------------------------------
    "documents.title": {"de": "Dokumente", "en": "Documents"},
    "documents.tab_library": {"de": "Bibliothek", "en": "Library"},
    "documents.tab_preview": {"de": "Vorschau", "en": "Preview"},
    "documents.add": {"de": "Regelungen hinzufuegen", "en": "Add regulations"},
    "documents.add_hint": {
        "de": "PDF oder DOCX, maximal {max_mb} MB pro Datei. Gescannte PDFs "
        "bitte vorher mit OCR versehen.",
        "en": "PDF or DOCX, at most {max_mb} MB per file. Please run OCR on "
        "scanned PDFs beforehand.",
    },
    "documents.files": {"de": "Dateien", "en": "Files"},
    "documents.force": {
        "de": "Bereits indizierte Dateien neu einlesen",
        "en": "Re-read files that are already indexed",
    },
    "documents.force_help": {
        "de": "Ohne Haken werden inhaltsgleiche Dateien uebersprungen.",
        "en": "Unchecked, files with identical content are skipped.",
    },
    "documents.index": {"de": "Indizieren", "en": "Index"},
    "documents.indexed": {"de": "Indizierte Dokumente", "en": "Indexed documents"},
    "documents.none_indexed": {"de": "Noch nichts indiziert.", "en": "Nothing indexed yet."},
    "documents.remove": {"de": "Entfernen", "en": "Remove"},
    "documents.removed_toast": {
        "de": "{count} Chunks entfernt.",
        "en": "{count} chunks removed.",
    },
    "documents.chunks_suffix": {"de": "{count} Chunks", "en": "{count} chunks"},
    "documents.pages_suffix": {"de": "{count} S.", "en": "{count} pp."},
    "documents.annexes": {"de": "Annexe: {list}", "en": "Annexes: {list}"},
    "documents.no_files": {
        "de": "Noch keine Dateien im Ordner `data/uploads`.",
        "en": "No files in the `data/uploads` folder yet.",
    },
    "documents.file": {"de": "Datei", "en": "File"},
    "documents.metadata": {"de": "Metadaten", "en": "Metadata"},
    "documents.meta_file": {"de": "Datei", "en": "File"},
    "documents.meta_type": {"de": "Typ", "en": "Type"},
    "documents.meta_size": {"de": "Groesse (MB)", "en": "Size (MB)"},
    "documents.meta_pages": {"de": "Seiten", "en": "Pages"},
    "documents.meta_title": {"de": "Titel", "en": "Title"},
    "documents.meta_author": {"de": "Autor", "en": "Author"},
    "documents.meta_regulation": {"de": "Erkannte Regelung", "en": "Detected regulation"},
    "documents.is_indexed": {
        "de": "Indiziert - {count} Chunks",
        "en": "Indexed - {count} chunks",
    },
    "documents.not_indexed": {"de": "Noch nicht indiziert", "en": "Not indexed yet"},
    "documents.index_now": {"de": "Jetzt indizieren", "en": "Index now"},
    "documents.preview": {"de": "Vorschau", "en": "Preview"},
    "documents.page": {"de": "Seite", "en": "Page"},
    "documents.render_failed": {
        "de": "Seite konnte nicht gerendert werden.",
        "en": "Page could not be rendered.",
    },
    "documents.text_excerpt": {
        "de": "Textauszug (DOCX/ohne Seitenrendering):",
        "en": "Text excerpt (DOCX / no page rendering):",
    },
    "documents.excerpt": {"de": "Auszug", "en": "Excerpt"},
    # -- Indizierung --------------------------------------------------------
    "ingest.start": {"de": "Starte Indizierung ...", "en": "Starting indexing ..."},
    "ingest.too_large": {
        "de": "{name}: {size} MB ueberschreiten das Limit.",
        "en": "{name}: {size} MB exceeds the limit.",
    },
    "ingest.save_failed": {
        "de": "{name}: konnte nicht gespeichert werden ({error}).",
        "en": "{name}: could not be saved ({error}).",
    },
    "ingest.embedding": {
        "de": "{name}: {done}/{total} Chunks eingebettet ...",
        "en": "{name}: {done}/{total} chunks embedded ...",
    },
    "ingest.ok": {
        "de": "{name}: {chunks} Chunks in {seconds}s",
        "en": "{name}: {chunks} chunks in {seconds}s",
    },
    "ingest.progress": {
        "de": "{done}/{total} verarbeitet",
        "en": "{done}/{total} processed",
    },
    # -- System -------------------------------------------------------------
    "system.title": {"de": "System", "en": "System"},
    "system.tab_status": {"de": "Status", "en": "Status"},
    "system.tab_retrieval": {"de": "Retrieval", "en": "Retrieval"},
    "system.tab_database": {"de": "Datenbank", "en": "Database"},
    "system.ollama": {"de": "Ollama", "en": "Ollama"},
    "system.ollama_ok": {
        "de": "Erreichbar - installierte Modelle:",
        "en": "Reachable - installed models:",
    },
    "system.ollama_empty": {
        "de": "Erreichbar - noch kein Modell installiert",
        "en": "Reachable - no model installed yet",
    },
    "system.model": {"de": "LLM-Modell", "en": "LLM model"},
    "system.model_help": {
        "de": "Blass dargestellte Eintraege sind bereits installiert. Alle "
        "uebrigen lassen sich mit 'Installieren' per 'ollama pull' nachladen.",
        "en": "Dimmed entries are already installed. Any other entry can be "
        "fetched with 'Install' - the equivalent of 'ollama pull'.",
    },
    "system.model_missing": {"de": "Modell nicht installiert", "en": "Model not installed"},
    "system.install": {"de": "Installieren", "en": "Install"},
    "system.installing": {"de": "Installiert ...", "en": "Installing ..."},
    "system.install_help": {
        "de": "Laedt das gewaehlte Modell nach. Nur aktiv, solange das Modell "
        "noch nicht installiert ist.",
        "en": "Downloads the selected model. Only active while the model is "
        "not installed yet.",
    },
    "system.install_start": {
        "de": "Lade {model} ...",
        "en": "Downloading {model} ...",
    },
    "system.install_progress": {
        "de": "{model}: {status} - {percent} %",
        "en": "{model}: {status} - {percent}%",
    },
    "system.install_done": {
        "de": "{model} installiert.",
        "en": "{model} installed.",
    },
    "system.install_failed": {
        "de": "Installation fehlgeschlagen: {message}",
        "en": "Installation failed: {message}",
    },
    "system.uninstall_help": {
        "de": "{model} deinstallieren",
        "en": "Uninstall {model}",
    },
    "system.uninstall_title": {
        "de": "Modell deinstallieren",
        "en": "Uninstall model",
    },
    "system.uninstall_question": {
        "de": "**{model}** wirklich aus Ollama entfernen?",
        "en": "Really remove **{model}** from Ollama?",
    },
    "system.uninstall_hint": {
        "de": "Der Speicherplatz wird frei. Die indizierten Dokumente bleiben "
        "erhalten; erneutes Installieren laedt das Modell wieder herunter.",
        "en": "This frees the disk space. Indexed documents are unaffected; "
        "installing it again downloads the model anew.",
    },
    "system.uninstall_cancel": {"de": "Abbrechen", "en": "Cancel"},
    "system.uninstall_confirm": {"de": "Deinstallieren", "en": "Uninstall"},
    "system.uninstall_done": {
        "de": "{model} deinstalliert.",
        "en": "{model} uninstalled.",
    },
    "system.uninstall_failed": {
        "de": "Deinstallation fehlgeschlagen: {message}",
        "en": "Uninstall failed: {message}",
    },
    "system.loaded": {"de": "Geladen: {list}", "en": "Loaded: {list}"},
    "system.compute": {"de": "Rechenwerk", "en": "Compute"},
    "system.vram": {
        "de": "VRAM {used} / {total} GB · {name}",
        "en": "VRAM {used} / {total} GB · {name}",
    },
    "system.gpu_unknown": {
        "de": "GPU erkannt, Auslastung nicht auslesbar",
        "en": "GPU detected, utilisation not readable",
    },
    "system.no_gpu": {
        "de": "Keine CUDA-GPU erkannt - Encoder laufen auf der CPU (langsam)",
        "en": "No CUDA GPU detected - encoders run on the CPU (slow)",
    },
    "system.device": {"de": "Device: `{device}`", "en": "Device: `{device}`"},
    "system.two_stage": {"de": "Zweistufige Suche", "en": "Two-stage search"},
    "system.two_stage_hint": {
        "de": "Stufe 1 holt Kandidaten aus der Vektordatenbank, Stufe 2 sortiert "
        "sie mit einem Cross-Encoder neu. Nur die besten Passagen wandern "
        "ins Prompt.",
        "en": "Stage 1 fetches candidates from the vector database, stage 2 "
        "re-ranks them with a cross-encoder. Only the best passages go into "
        "the prompt.",
    },
    "system.top_k": {
        "de": "Stufe 1 - Vektorsuche (Top-K)",
        "en": "Stage 1 - vector search (top K)",
    },
    "system.top_k_help": {
        "de": "Kandidaten aus ChromaDB. Mehr = besserer Recall, langsamer.",
        "en": "Candidates from ChromaDB. More = better recall, slower.",
    },
    "system.top_n": {
        "de": "Stufe 2 - Reranker (Top-N)",
        "en": "Stage 2 - reranker (top N)",
    },
    "system.top_n_help": {
        "de": "Wie viele Passagen nach dem Cross-Encoder ins Prompt wandern.",
        "en": "How many passages go into the prompt after the cross-encoder.",
    },
    "system.generation": {"de": "Generierung", "en": "Generation"},
    "system.temperature": {"de": "Temperatur", "en": "Temperature"},
    "system.temperature_help": {
        "de": "Niedrig halten - juristische Antworten sollen praezise sein.",
        "en": "Keep it low - legal answers should be precise.",
    },
    "system.models": {"de": "Modelle", "en": "Models"},
    "system.model_info": {
        "de": "Embedding: `{embedding}`  \nReranker: `{reranker}`  \n"
        "Relevanz-Schwellwert: `{threshold}`",
        "en": "Embedding: `{embedding}`  \nReranker: `{reranker}`  \n"
        "Relevance threshold: `{threshold}`",
    },
    "system.scope": {"de": "Umfang", "en": "Scope"},
    "system.documents": {"de": "Dokumente", "en": "Documents"},
    "system.chunks": {"de": "Chunks", "en": "Chunks"},
    "system.location": {"de": "Speicherort", "en": "Location"},
    "system.chroma": {
        "de": "ChromaDB (embedded) · Collection `{collection}`",
        "en": "ChromaDB (embedded) · collection `{collection}`",
    },
    "system.reset": {"de": "Zuruecksetzen", "en": "Reset"},
    "system.reset_hint": {
        "de": "Loescht alle indizierten Chunks. Die Originaldateien in "
        "data/uploads bleiben erhalten und koennen neu indiziert werden.",
        "en": "Deletes all indexed chunks. The original files in data/uploads "
        "are kept and can be re-indexed.",
    },
    "system.clear_db": {"de": "Datenbank leeren", "en": "Clear database"},
    "system.clear_warning": {
        "de": "Alle indizierten Chunks werden unwiderruflich geloescht.",
        "en": "All indexed chunks will be deleted irreversibly.",
    },
    "system.clear_confirm": {"de": "Ja, alles loeschen", "en": "Yes, delete everything"},
    "system.cleared_toast": {"de": "Collection geleert.", "en": "Collection cleared."},
    # -- Einstellungen ------------------------------------------------------
    "settings.title": {"de": "Einstellungen", "en": "Settings"},
    "settings.language": {"de": "Sprache", "en": "Language"},
    "settings.language_hint": {
        "de": "Gilt fuer die Oberflaeche. Antworten kommen weiterhin in der "
        "Sprache der jeweiligen Frage - eine englische Frage wird auch bei "
        "deutscher Oberflaeche englisch beantwortet.",
        "en": "Applies to the interface. Answers still come in the language of "
        "the question - an English question is answered in English even with "
        "a German interface.",
    },
    "settings.language_label": {"de": "Sprache der Oberflaeche", "en": "Interface language"},
    "settings.about": {"de": "Ueber", "en": "About"},
    "settings.about_text": {
        "de": "**{name}** {version} - laeuft vollstaendig lokal: Ollama fuer die "
        "Generierung, ChromaDB als Vektorindex. Es verlaesst kein Dokument "
        "diesen Rechner.",
        "en": "**{name}** {version} - runs entirely locally: Ollama for "
        "generation, ChromaDB as the vector index. No document leaves this "
        "machine.",
    },
    "settings.paths": {"de": "Ablage", "en": "Storage"},
    "settings.uploads_path": {"de": "Hochgeladene Dateien", "en": "Uploaded files"},
    "settings.index_path": {"de": "Vektorindex", "en": "Vector index"},
    # -- Startfehler --------------------------------------------------------
    "error.title": {"de": "Start fehlgeschlagen", "en": "Startup failed"},
    "error.headline": {
        "de": "Reg-Search konnte nicht initialisiert werden",
        "en": "Reg-Search could not be initialised",
    },
    "error.unknown": {"de": "Unbekannter Fehler", "en": "Unknown error"},
    "error.checklist": {"de": "Checkliste", "en": "Checklist"},
    "error.checklist_items": {
        "de": "1. Abhaengigkeiten installiert? `pip install -r requirements.txt`\n"
        "2. Beim ersten Start laedt Reg-Search die Modelle von HuggingFace "
        "(~2,5 GB) - dafuer wird eine Internetverbindung benoetigt.\n"
        "3. Zu wenig VRAM? `REG_SEARCH_DEVICE=cpu` setzen.\n"
        "4. Laeuft eine zweite Instanz auf `./chroma_db`? Dann beenden.",
        "en": "1. Dependencies installed? `pip install -r requirements.txt`\n"
        "2. On first start Reg-Search downloads the models from HuggingFace "
        "(~2.5 GB) - this needs an internet connection.\n"
        "3. Not enough VRAM? Set `REG_SEARCH_DEVICE=cpu`.\n"
        "4. Is a second instance using `./chroma_db`? Then stop it.",
    },
}


# --------------------------------------------------------------------------- #
# Sprachwahl
# --------------------------------------------------------------------------- #
def init() -> None:
    """
    Legt die Sprache fuer diesen Lauf fest.

    Reihenfolge: bereits gewaehlte Sprache -> ``?lang=`` aus der URL ->
    ``config.APP_LANGUAGE``.
    """
    if st.session_state.get("lang") in LANGUAGES:
        return
    from_url = st.query_params.get("lang")
    st.session_state.lang = from_url if from_url in LANGUAGES else _default()


def _default() -> str:
    return config.APP_LANGUAGE if config.APP_LANGUAGE in LANGUAGES else FALLBACK


def current() -> str:
    """Aktueller Sprachcode."""
    lang = st.session_state.get("lang")
    return lang if lang in LANGUAGES else _default()


def set_language(code: str) -> None:
    """Wechselt die Sprache und haelt sie in der URL fest (ueberlebt Reload)."""
    if code not in LANGUAGES or code == current():
        return
    st.session_state.lang = code
    st.query_params["lang"] = code


def t(key: str, **kwargs) -> str:
    """
    Uebersetzt ``key``. Platzhalter werden per ``str.format`` gefuellt.

    Unbekannte Schluessel liefern den Schluessel zurueck, statt eine Ausnahme
    zu werfen - eine fehlende Uebersetzung soll die Oberflaeche nicht zerlegen.
    """
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(current()) or entry.get(FALLBACK, key)
    if not kwargs:
        return text
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError):
        # Platzhalter passt nicht zum Aufruf - lieber Rohtext als Absturz.
        return text

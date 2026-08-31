"""
app.py - Streamlit-Oberflaeche von Reg-Search.

Aufbau
------
Die Oberflaeche folgt dem Muster eines Einstellungs-Dashboards, wie es bei
Entwickler-Werkzeugen im Dark Mode ueblich ist:

Sidebar:   ausschliesslich Navigation - Chat, Dokumente, System, Einstellungen.
Hauptteil: Seitentitel, darunter Reiter, darunter der Inhalt.

  Chat           Modellauswahl und Quellenfilter als schmale Zeile ueber dem
                 Verlauf, danach Fragen und Antworten mit Fundstellen.
  Dokumente      Reiter "Bibliothek" (Upload, Indizierung, indizierte Dateien)
                 und "Vorschau" (Metadaten, Seitenansicht).
  System         Reiter "Status" (Ollama, GPU), "Retrieval" (Top-K, Top-N,
                 Temperatur) und "Datenbank" (Umfang, Pfad, Leeren).
  Einstellungen  Sprache der Oberflaeche, Ablageorte.

Sprachen
--------
Alle sichtbaren Texte laufen ueber ``i18n.t()``. Deshalb traegt jedes Widget
mit Zustand ein festes ``key=``: ohne das leitet Streamlit die Identitaet aus
der Beschriftung ab, und ein Sprachwechsel wuerde Modellauswahl, Filter und
Regler zuruecksetzen.

Caching
-------
ChromaDB-Client, Embedding-Modell, Reranker und die Engine werden mit
``@st.cache_resource`` gehalten. Streamlit fuehrt bei *jeder* Interaktion das
gesamte Skript erneut aus - ohne diesen Cache wuerden pro Chat-Nachricht
mehrere Gigabyte Modellgewichte neu geladen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

import config
import document_processor as dp
import i18n
import rag_engine
import ui
from i18n import t
from rag_engine import RagEngineError, RegSearchEngine, RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# --------------------------------------------------------------------------- #
# Seiten-Setup (muss der erste Streamlit-Aufruf sein)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

i18n.init()

# Dark Mode im Stil eines Einstellungs-Dashboards, mit Lime als einziger
# Akzentfarbe. Farben/Radien/Schriften stehen in .streamlit/config.toml,
# Layout und eigene Bauteile in ui.py.
ui.inject()

config.ensure_directories()

#: Reihenfolge der Navigation. Die Beschriftungen kommen uebersetzt dazu.
NAV_KEYS = ["chat", "documents", "system", "settings"]


# --------------------------------------------------------------------------- #
# Gecachte Ressourcen
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Oeffne ChromaDB (embedded, ./chroma_db) ...")
def get_chroma_client():
    """Persistenter ChromaDB-Client - genau einmal pro Prozess."""
    return rag_engine.create_chroma_client()


@st.cache_resource(show_spinner="Lade Embedding-Modell (BAAI/bge-m3) ...")
def get_embeddings():
    return rag_engine.create_embeddings()


@st.cache_resource(show_spinner="Initialisiere RAG-Engine ...")
def get_engine() -> RegSearchEngine:
    """
    Die komplette Pipeline. Der Reranker wird innerhalb der Engine lazy geladen
    (beim ersten Suchlauf) und bleibt dann Teil dieses gecachten Objekts.
    """
    return RegSearchEngine(chroma_client=get_chroma_client(), embeddings=get_embeddings())


@st.cache_data(ttl=15, show_spinner=False)
def get_ollama_status(url: str) -> dict:
    """Erreichbarkeit + installierte Modelle (kurz gecacht, damit die UI fluessig bleibt)."""
    return rag_engine.check_ollama(url)


def init_state() -> None:
    defaults = {
        "messages": [],
        "llm_model": config.LLM_MODEL,
        "top_k": config.RETRIEVAL_TOP_K,
        "top_n": config.RERANK_TOP_N,
        "temperature": config.LLM_TEMPERATURE,
        "source_filter": [],
        "pending_question": None,
        "model_notice": None,
        "installing": None,
        "uninstall_target": None,
        "nav": "chat",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# --------------------------------------------------------------------------- #
# Hilfsfunktionen fuer die Darstellung
# --------------------------------------------------------------------------- #
def pick_default_model(installed: list[str]) -> str:
    """
    Waehlt ein sinnvolles Startmodell.

    Ist das konfigurierte Standardmodell nicht installiert, waere die App sonst
    beim ersten Start unbenutzbar. Dann greift die Vorschlagsliste, danach das
    erste ohnehin vorhandene Modell.
    """
    if not installed or config.LLM_MODEL in installed:
        return config.LLM_MODEL
    for suggestion in config.LLM_MODEL_SUGGESTIONS:
        family = suggestion.split(":")[0]
        for name in installed:
            if name.startswith(family):
                return name
    return installed[0]


def model_options(status: dict, installed_only: bool = False) -> list[str]:
    """
    Auswahlliste fuer das LLM: installierte Modelle, das aktuell gewaehlte und
    die Vorschlaege aus config.py - ohne Doppelte.

    ``installed_only`` laesst die Vorschlaege weg. Im Chat soll nur zur Wahl
    stehen, was sofort antwortet; das Nachladen per 'ollama pull' gehoert auf
    die Systemseite. Antwortet Ollama nicht, sind die installierten Modelle
    unbekannt - dann bleiben die Vorschlaege auch hier die einzige Auswahl.

    Der aktuell gewaehlte Eintrag muss immer enthalten sein: die Auswahlfelder
    laufen ueber ``key="llm_model"``, und Streamlit wirft einen Fehler, wenn
    der Wert im Zustand nicht in den Optionen steht.
    """
    if status["ok"] and not st.session_state.get("model_chosen"):
        # Einmalig ein installiertes Modell vorauswaehlen, falls der Standard
        # aus config.py nicht heruntergeladen wurde.
        st.session_state.llm_model = pick_default_model(status["models"])
        st.session_state.model_chosen = True

    models = list(status["models"])
    if st.session_state.llm_model not in models:
        models.insert(0, st.session_state.llm_model)
    if installed_only and status["ok"]:
        return models
    for suggestion in config.LLM_MODEL_SUGGESTIONS:
        if suggestion not in models:
            models.append(suggestion)
    return models


def indexed_sources(engine: RegSearchEngine) -> list[str]:
    """Dateinamen der indizierten Dokumente - Grundlage fuer den Quellenfilter."""
    try:
        return [str(doc["source"]) for doc in engine.list_documents()]
    except RagEngineError:
        return []


def render_sources(sources: list[RetrievedChunk]) -> None:
    """Zeigt die Fundstellen als ausklappbare Zitate mit Struktur-Metadaten."""
    if not sources:
        return
    with st.expander(t("sources.expander", count=len(sources)), expanded=False):
        for chunk in sources:
            meta = chunk.metadata
            st.markdown(f"**[{chunk.rank}] {chunk.citation}**")

            badges = [ui.badge(meta.get("source", "?"), grey=True)]
            if meta.get("annex"):
                badges.append(ui.badge(f"Annex {meta['annex']}"))
            if meta.get("appendix"):
                badges.append(ui.badge(f"Appendix {meta['appendix']}"))
            if meta.get("paragraph"):
                badges.append(ui.badge(f"Para. {meta['paragraph']}"))
            if meta.get("page_start"):
                pages = (
                    t("sources.page", page=meta["page_start"])
                    if meta.get("page_start") == meta.get("page_end")
                    else t(
                        "sources.pages",
                        first=meta.get("page_start"),
                        last=meta.get("page_end"),
                    )
                )
                badges.append(ui.badge(pages, grey=True))
            if chunk.parent_context:
                # Kein eigener Treffer, sondern der uebergeordnete Abschnitt
                # eines Treffers - Scores gibt es dafuer nicht.
                badges.append(ui.badge(t("sources.parent")))
                st.markdown(" ".join(badges), unsafe_allow_html=True)
                st.markdown(ui.quote(chunk.body), unsafe_allow_html=True)
                st.divider()
                continue

            badges.append(ui.badge(t("sources.rerank", score=f"{chunk.rerank_score:.2f}"), grey=True))
            if chunk.exact_match:
                # Direkt ueber die Fundstelle geholt - ein Vektorabstand
                # existiert dafuer nicht, "Vektor 0.00" waere irrefuehrend.
                badges.append(ui.badge(t("sources.exact")))
            else:
                badges.append(
                    ui.badge(t("sources.vector", score=f"{chunk.vector_score:.2f}"), grey=True)
                )
            st.markdown(" ".join(badges), unsafe_allow_html=True)

            st.markdown(ui.quote(chunk.body), unsafe_allow_html=True)
            st.divider()


def save_upload(uploaded_file) -> Path:
    """Speichert einen Upload im Ordner ``data/uploads`` und gibt den Pfad zurueck."""
    target = config.UPLOAD_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


# --------------------------------------------------------------------------- #
# Sidebar: nur Navigation
# --------------------------------------------------------------------------- #
def render_nav() -> str:
    """Zeichnet die Navigation und liefert den aktiven Bereich."""
    items = [(key, t(f"nav.{key}")) for key in NAV_KEYS]
    with st.sidebar:
        ui.brand(config.APP_NAME)
        selected = ui.nav(items, st.session_state.nav)
        ui.nav_footer(
            f"{t('app.tagline')}<br>{t('nav.footer', version=config.APP_VERSION)}"
        )

    if selected != st.session_state.nav:
        st.session_state.nav = selected
        st.rerun()
    return st.session_state.nav


# --------------------------------------------------------------------------- #
# Bereich: Chat
# --------------------------------------------------------------------------- #
EXAMPLE_KEYS = ["chat.example_1", "chat.example_2", "chat.example_3"]


def render_chat_controls(engine: RegSearchEngine) -> None:
    """Schmale Zeile ueber dem Verlauf: Modell, Quellenfilter, Verlauf leeren."""
    status = get_ollama_status(config.OLLAMA_BASE_URL)
    models = model_options(status, installed_only=True)

    # Der Filter muss eine Teilmenge der Optionen bleiben, sonst wirft
    # Streamlit beim Rendern mit key= einen Fehler (z. B. nach dem Loeschen
    # eines Dokuments).
    sources = indexed_sources(engine)
    st.session_state.source_filter = [
        src for src in st.session_state.source_filter if src in sources
    ]

    model_col, filter_col, clear_col = st.columns([1.1, 1.5, 0.6], vertical_alignment="bottom")

    model_col.selectbox(
        t("chat.model"),
        options=models,
        key="llm_model",
        help=t("chat.model_help"),
    )
    filter_col.multiselect(
        t("chat.sources"),
        options=sources,
        key="source_filter",
        placeholder=t("chat.sources_all"),
        help=t("chat.sources_help"),
    )
    if clear_col.button(t("chat.clear"), key="rs-clear-chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    if status["ok"] and st.session_state.llm_model not in status["models"]:
        ui.status(t("chat.model_missing", model=st.session_state.llm_model), "warn")
    elif not status["ok"]:
        ui.status(t("chat.ollama_down"), "bad")


def render_chat(engine: RegSearchEngine) -> None:
    ui.page_title(t("chat.title"))
    render_chat_controls(engine)
    st.divider()

    indexed = engine.count_chunks()
    if indexed == 0:
        st.info(t("chat.empty_index"))

    # bisherigen Verlauf ausgeben
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                if message.get("footer"):
                    st.caption(message["footer"])

    if not st.session_state.messages and indexed:
        ui.group(t("chat.examples"))
        columns = st.columns(len(EXAMPLE_KEYS))
        for index, (column, key) in enumerate(zip(columns, EXAMPLE_KEYS)):
            if column.button(t(key), key=f"rs-example-{index}", width="stretch"):
                st.session_state.pending_question = t(key)
                st.rerun()

    typed = st.chat_input(t("chat.input_placeholder"))
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # --- Stufe 1 + 2: Retrieval ----------------------------------------
        try:
            with st.status(t("chat.retrieving"), expanded=False) as status_box:
                status_box.write(t("chat.vector_search", top_k=st.session_state.top_k))
                sources = engine.retrieve(
                    question,
                    top_k=st.session_state.top_k,
                    top_n=st.session_state.top_n,
                    sources=st.session_state.source_filter or None,
                )
                status_box.write(t("chat.reranking", count=len(sources)))
                if len(sources) < st.session_state.top_n:
                    status_box.write(
                        t("chat.below_threshold", threshold=config.RERANK_MIN_SCORE)
                    )
                status_box.update(label=t("chat.hits", count=len(sources)), state="complete")
        except RagEngineError as exc:
            st.error(str(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": t("chat.error", message=exc), "sources": []}
            )
            return

        # Ehrlichkeit vor Bequemlichkeit: schwache Treffer werden markiert,
        # damit niemand eine erfundene Antwort fuer belastbar haelt.
        if any(chunk.low_confidence for chunk in sources):
            st.warning(t("chat.low_confidence"))

        # --- Stufe 3: Generierung (streamend) ------------------------------
        placeholder = st.empty()
        answer = ""
        history = st.session_state.messages[:-1]
        try:
            for token in engine.stream_answer(
                question,
                sources,
                history=history,
                model=st.session_state.llm_model,
                temperature=st.session_state.temperature,
            ):
                answer += token
                placeholder.markdown(answer + " ▌")
            placeholder.markdown(answer)
        except RagEngineError as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.append(
                {"role": "assistant", "content": t("chat.error", message=exc), "sources": []}
            )
            return

        render_sources(sources)
        footer = t(
            "chat.footer",
            model=st.session_state.llm_model,
            top_k=st.session_state.top_k,
            top_n=len(sources),
            reranker=config.RERANKER_MODEL.split("/")[-1],
        )
        st.caption(footer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "footer": footer}
    )


# --------------------------------------------------------------------------- #
# Bereich: Dokumente
# --------------------------------------------------------------------------- #
def run_ingestion(engine: RegSearchEngine, uploads, force: bool) -> None:
    """Speichert Uploads, indiziert sie und meldet das Ergebnis pro Datei."""
    progress = st.progress(0.0, text=t("ingest.start"))
    status_box = st.empty()

    for position, uploaded in enumerate(uploads, start=1):
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > config.MAX_UPLOAD_MB:
            status_box.error(t("ingest.too_large", name=uploaded.name, size=f"{size_mb:.0f}"))
            continue

        try:
            path = save_upload(uploaded)
        except OSError as exc:
            status_box.error(t("ingest.save_failed", name=uploaded.name, error=exc))
            continue

        def on_progress(done: int, total: int, name: str = uploaded.name) -> None:
            progress.progress(
                min(1.0, (position - 1 + done / max(total, 1)) / len(uploads)),
                text=t("ingest.embedding", name=name, done=done, total=total),
            )

        try:
            result = engine.index_file(path, force=force, progress_callback=on_progress)
        except RagEngineError as exc:
            status_box.error(f"{uploaded.name}: {exc}")
            continue

        if result.status == "ok":
            status_box.success(
                t(
                    "ingest.ok",
                    name=result.filename,
                    chunks=result.n_chunks,
                    seconds=result.duration_s,
                )
            )
        elif result.status == "skipped":
            status_box.info(f"{result.filename}: {result.message}")
        else:
            status_box.error(f"{result.filename}: {result.message}")

        for warning in result.warnings:
            st.warning(f"{result.filename}: {warning}")

        progress.progress(
            position / len(uploads),
            text=t("ingest.progress", done=position, total=len(uploads)),
        )

    progress.empty()
    # Kein st.rerun(): die Liste der indizierten Dateien wird weiter unten auf
    # derselben Seite gerendert und ist damit ohnehin aktuell - so bleiben die
    # Ergebnismeldungen pro Datei sichtbar.


def render_library(engine: RegSearchEngine) -> None:
    """Reiter "Bibliothek": hochladen, indizieren, indizierte Dateien verwalten."""
    ui.group(t("documents.add"), t("documents.add_hint", max_mb=config.MAX_UPLOAD_MB))
    uploads = st.file_uploader(
        t("documents.files"),
        type=[ext.lstrip(".") for ext in config.SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        key="rs-uploader",
        label_visibility="collapsed",
    )
    force = st.checkbox(
        t("documents.force"),
        key="rs-force-reindex",
        help=t("documents.force_help"),
    )
    if uploads and st.button(t("documents.index"), key="rs-index", type="primary"):
        run_ingestion(engine, uploads, force)

    ui.group(t("documents.indexed"))
    try:
        documents = engine.list_documents()
    except RagEngineError as exc:
        st.error(str(exc))
        return

    if not documents:
        st.caption(t("documents.none_indexed"))
        return

    for document in documents:
        # key= vergibt die CSS-Klasse st-key-rs-card-<doc_id>; darueber
        # bekommt die Karte in ui.py ihren Rahmen.
        with st.container(border=True, key=f"rs-card-{document['doc_id']}"):
            head, remove = st.columns([6, 1], vertical_alignment="center")
            # Kurztitel zuerst: der Langtitel beginnt bei jeder UNECE-Regelung
            # gleich ("Uniform provisions concerning the approval of ...") und
            # unterscheidet in einer Liste nichts.
            label = (
                document.get("short_title")
                or document["regulation"]
                or document["doc_title"]
                or document["source"]
            )
            head.markdown(f"**{str(label)[:90]}**")
            details = [
                document["source"],
                t("documents.chunks_suffix", count=document["chunks"]),
            ]
            if document.get("n_pages"):
                details.append(t("documents.pages_suffix", count=document["n_pages"]))
            if document["annexes"]:
                details.append(
                    t("documents.annexes", list=", ".join(document["annexes"][:6]))
                )
            head.caption(" · ".join(details))
            if remove.button(t("documents.remove"), key=f"del_{document['doc_id']}"):
                removed = engine.delete_document(document["doc_id"])
                st.toast(t("documents.removed_toast", count=removed))
                st.rerun()


def render_preview(engine: RegSearchEngine) -> None:
    """Reiter "Vorschau": Metadaten und Seitenansicht einer hochgeladenen Datei."""
    files = sorted(
        path
        for path in config.UPLOAD_DIR.glob("*")
        if path.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    if not files:
        st.caption(t("documents.no_files"))
        return

    selection = st.selectbox(
        t("documents.file"),
        options=files,
        key="rs-preview-file",
        format_func=lambda p: p.name,
    )
    info = dp.get_document_info(selection)

    meta_col, preview_col = st.columns([1, 1.3], gap="large")

    with meta_col:
        ui.group(t("documents.metadata"))
        st.write(
            {
                t("documents.meta_file"): info["filename"],
                t("documents.meta_type"): info["file_type"],
                t("documents.meta_size"): info["size_mb"],
                t("documents.meta_pages"): info["pages"] or "-",
                t("documents.meta_title"): info["title"],
                t("documents.meta_author"): info["author"] or "-",
                t("documents.meta_regulation"): info["regulation"] or "-",
            }
        )

        # Struktur der bereits indizierten Chunks
        try:
            indexed = [
                doc for doc in engine.list_documents() if doc["source"] == selection.name
            ]
        except RagEngineError:
            indexed = []
        if indexed:
            document = indexed[0]
            ui.status(t("documents.is_indexed", count=document["chunks"]), "ok")
            if document["annexes"]:
                st.markdown(
                    " ".join(ui.badge(f"Annex {a}") for a in document["annexes"]),
                    unsafe_allow_html=True,
                )
        else:
            ui.status(t("documents.not_indexed"), "warn")
            if st.button(t("documents.index_now"), key="rs-index-single", type="primary"):
                result = engine.index_file(selection)
                st.toast(f"{result.filename}: {result.message}")
                st.rerun()

    with preview_col:
        ui.group(t("documents.preview"))
        if selection.suffix.lower() == ".pdf" and info["pages"]:
            page = st.number_input(
                t("documents.page"),
                min_value=1,
                max_value=int(info["pages"]),
                value=1,
                step=1,
                key="rs-preview-page",
            )
            image = dp.render_pdf_page(selection, int(page))
            if image:
                # bewusst ohne use_container_width/use_column_width:
                # der Parametername wechselte zwischen Streamlit-Versionen
                st.image(image)
            else:
                st.warning(t("documents.render_failed"))
        else:
            st.caption(t("documents.text_excerpt"))
            try:
                processed = dp.process_document(selection)
                st.text_area(
                    t("documents.excerpt"),
                    value="\n\n".join(chunk.text for chunk in processed.chunks[:3]),
                    height=420,
                    key="rs-preview-text",
                    label_visibility="collapsed",
                )
            except dp.DocumentProcessingError as exc:
                st.error(str(exc))


def render_documents(engine: RegSearchEngine) -> None:
    ui.page_title(t("documents.title"))
    library_tab, preview_tab = st.tabs(
        [t("documents.tab_library"), t("documents.tab_preview")]
    )
    with library_tab:
        render_library(engine)
    with preview_tab:
        render_preview(engine)


# --------------------------------------------------------------------------- #
# Bereich: System
# --------------------------------------------------------------------------- #
def install_model(model: str, slot: Any) -> None:
    """
    Laedt ein Modell per 'ollama pull' nach und zeigt den Fortschritt in ``slot``.

    Der Skriptlauf haengt waehrenddessen am Stream - bei einem 9-GB-Modell
    sind das etliche Minuten. Der Fortschrittsbalken ist deshalb Pflicht, nicht
    Zierde. Er steht in einem vorher reservierten ``st.empty()`` direkt unter
    der Auswahl; der Rest der Seite ist da schon gezeichnet und bleibt
    waehrend des Downloads sichtbar.
    """
    bar = slot.progress(0.0, text=t("system.install_start", model=model))
    try:
        for event in rag_engine.pull_model(model, config.OLLAMA_BASE_URL):
            share = event["completed"] / event["total"] if event["total"] else 0.0
            share = min(max(share, 0.0), 1.0)
            bar.progress(
                share,
                text=t(
                    "system.install_progress",
                    model=model,
                    status=event["status"],
                    percent=int(share * 100),
                ),
            )
    except Exception as exc:  # noqa: BLE001 - jede Ursache gehoert in die UI
        finish_model_change(t("system.install_failed", message=str(exc)), "bad")
    finish_model_change(t("system.install_done", model=model), "ok")


def uninstall_model(model: str) -> None:
    """Entfernt ein Modell aus Ollama - aufgerufen aus dem Bestaetigungsdialog."""
    try:
        rag_engine.remove_model(model, config.OLLAMA_BASE_URL)
    except Exception as exc:  # noqa: BLE001 - jede Ursache gehoert in die UI
        finish_model_change(t("system.uninstall_failed", message=str(exc)), "bad")
    if st.session_state.llm_model == model:
        # Das aktive Modell ist weg; model_options() waehlt beim naechsten
        # Lauf ein vorhandenes aus.
        st.session_state.model_chosen = False
    finish_model_change(t("system.uninstall_done", model=model), "ok")


def finish_model_change(notice: str, tone: str) -> None:
    """Raeumt nach Installation/Deinstallation auf und zeichnet die Seite neu."""
    st.session_state.installing = None
    st.session_state.uninstall_target = None
    # Der Status ist 15 s gecacht; ohne Leeren stimmt die Modellliste nicht.
    get_ollama_status.clear()
    st.session_state.model_notice = (notice, tone)
    st.rerun()


def render_installed_models(models: list[str]) -> None:
    """Installierte Modelle untereinander, je Zeile ein Papierkorb."""
    for model in models:
        # Waagerechter Container statt Spalten: die Zeile ist nur so breit wie
        # ihr Inhalt, der Papierkorb steht damit direkt neben dem Namen und
        # nicht am rechten Seitenrand.
        row = st.container(horizontal=True, vertical_alignment="center", width="content")
        # Feste Breite fuer den Namen, damit die Papierkoerbe untereinander
        # stehen und nicht der Laenge des Modellnamens folgen.
        row.container(width=300).markdown(f"`{model}`")
        if row.button(
            "",
            icon=":material/delete:",
            key=f"rs-uninstall-{model}",
            help=t("system.uninstall_help", model=model),
            # Waehrend eines Downloads gesperrt: der Klick loeste einen Rerun
            # aus und der Pull finge von vorne an.
            disabled=bool(st.session_state.installing),
        ):
            st.session_state.uninstall_target = model
            st.rerun()


def confirm_uninstall(model: str) -> None:
    """
    Sicherheitsabfrage vor dem Loeschen eines Modells.

    Der Dialog wird hier drinnen definiert, weil ``@st.dialog`` seinen Titel
    beim Dekorieren festhaelt - auf Modulebene stuende dort die Sprache des
    ersten Skriptlaufs.
    """

    @st.dialog(t("system.uninstall_title"))
    def dialog() -> None:
        st.markdown(t("system.uninstall_question", model=model))
        st.caption(t("system.uninstall_hint"))
        cancel_col, confirm_col = st.columns(2)
        if cancel_col.button(t("system.uninstall_cancel"), key="rs-uninstall-no", width="stretch"):
            st.session_state.uninstall_target = None
            st.rerun()
        if confirm_col.button(
            t("system.uninstall_confirm"),
            key="rs-uninstall-yes",
            type="primary",
            width="stretch",
        ):
            uninstall_model(model)

    dialog()


def render_system_status() -> None:
    """Reiter "Status": Ollama, Modellverwaltung, GPU."""
    status = get_ollama_status(config.OLLAMA_BASE_URL)

    ui.group(t("system.ollama"))
    if not status["ok"]:
        ui.status(status["error"], "bad")
        st.code("ollama serve", language="bash")
    elif status["models"]:
        ui.status(t("system.ollama_ok"), "ok")
        render_installed_models(status["models"])
    else:
        ui.status(t("system.ollama_empty"), "warn")

    if st.session_state.model_notice:
        ui.status(*st.session_state.model_notice)
        st.session_state.model_notice = None

    models = model_options(status)
    label = t("system.model")
    # Was schon da ist, wird blass dargestellt; zu holen ist nur der Rest.
    ui.dim_options(label, [i for i, name in enumerate(models) if name in status["models"]])

    installing = st.session_state.installing
    model_col, install_col = st.columns([3, 1], vertical_alignment="bottom")
    model_col.selectbox(
        label,
        options=models,
        key="llm_model",
        help=t("system.model_help"),
        disabled=bool(installing),
    )
    missing = st.session_state.llm_model not in status["models"]
    if install_col.button(
        t("system.installing") if installing else t("system.install"),
        key="rs-install-model",
        width="stretch",
        type="primary",
        disabled=bool(installing) or not (status["ok"] and missing),
        help=t("system.install_help"),
    ):
        # Erst umschalten, dann laden: so traegt der Knopf waehrend des
        # Downloads "Installiert ..." statt weiter zum Klicken einzuladen.
        st.session_state.installing = st.session_state.llm_model
        st.rerun()
    progress_slot = st.empty()

    if status["ok"] and missing and not installing:
        ui.status(t("system.model_missing"), "warn")

    if status["ok"]:  # nur abfragen, wenn der Dienst antwortet
        loaded = rag_engine.loaded_models(config.OLLAMA_BASE_URL)
        if loaded:
            st.caption(
                t(
                    "system.loaded",
                    list=" · ".join(
                        f"{e['model']} (~{e['vram_gb']} GB)" for e in loaded
                    ),
                )
            )

    ui.group(t("system.compute"))
    gpu = rag_engine.gpu_status()
    if gpu.get("available") and "total_gb" in gpu:
        st.progress(
            min(1.0, gpu["used_gb"] / max(gpu["total_gb"], 0.1)),
            text=t(
                "system.vram",
                used=f"{gpu['used_gb']:.1f}",
                total=f"{gpu['total_gb']:.1f}",
                name=gpu["name"],
            ),
        )
    elif gpu.get("available"):
        ui.status(t("system.gpu_unknown"), "warn")
    else:
        ui.status(t("system.no_gpu"), "warn")
    st.caption(t("system.device", device=config.resolve_device()))

    # Ganz zum Schluss: beides haelt den Skriptlauf an (Download bzw. Dialog),
    # die Seite darunter soll aber schon stehen.
    if installing:
        install_model(installing, progress_slot)
    if st.session_state.uninstall_target:
        confirm_uninstall(st.session_state.uninstall_target)


def render_system_retrieval() -> None:
    """Reiter "Retrieval": die drei Stellschrauben der Suche."""
    ui.group(t("system.two_stage"), t("system.two_stage_hint"))
    st.slider(t("system.top_k"), 5, 50, step=5, key="top_k", help=t("system.top_k_help"))
    st.slider(t("system.top_n"), 1, 10, key="top_n", help=t("system.top_n_help"))

    ui.group(t("system.generation"))
    st.slider(
        t("system.temperature"),
        0.0,
        1.0,
        step=0.05,
        key="temperature",
        help=t("system.temperature_help"),
    )

    ui.group(t("system.models"))
    st.caption(
        t(
            "system.model_info",
            embedding=config.EMBEDDING_MODEL,
            reranker=config.RERANKER_MODEL,
            threshold=config.RERANK_MIN_SCORE,
        )
    )


def render_system_database(engine: RegSearchEngine) -> None:
    """Reiter "Datenbank": Umfang, Speicherort, Leeren."""
    try:
        stats = engine.stats()
    except RagEngineError as exc:
        st.error(str(exc))
        return

    ui.group(t("system.scope"))
    left, right, _ = st.columns([1, 1, 2])
    left.metric(t("system.documents"), stats["documents"])
    right.metric(t("system.chunks"), stats["chunks"])

    ui.group(t("system.location"))
    st.caption(t("system.chroma", collection=stats["collection"]))
    st.code(str(stats["path"]), language="text")

    ui.group(t("system.reset"), t("system.reset_hint"))
    with st.popover(t("system.clear_db")):
        st.write(t("system.clear_warning"))
        if st.button(t("system.clear_confirm"), key="rs-clear-db", type="primary"):
            engine.reset_collection()
            st.session_state.source_filter = []
            st.toast(t("system.cleared_toast"))
            st.rerun()


def render_system(engine: RegSearchEngine) -> None:
    ui.page_title(t("system.title"))
    status_tab, retrieval_tab, database_tab = st.tabs(
        [t("system.tab_status"), t("system.tab_retrieval"), t("system.tab_database")]
    )
    with status_tab:
        render_system_status()
    with retrieval_tab:
        render_system_retrieval()
    with database_tab:
        render_system_database(engine)


# --------------------------------------------------------------------------- #
# Bereich: Einstellungen
# --------------------------------------------------------------------------- #
def render_settings() -> None:
    ui.page_title(t("settings.title"))

    ui.group(t("settings.language"), t("settings.language_hint"))
    codes = list(i18n.LANGUAGES)
    chosen = st.radio(
        t("settings.language_label"),
        options=codes,
        index=codes.index(i18n.current()),
        format_func=lambda code: i18n.LANGUAGES[code],
        key="rs-language",
        horizontal=True,
        label_visibility="collapsed",
    )
    if chosen != i18n.current():
        i18n.set_language(chosen)
        st.rerun()

    ui.group(t("settings.about"))
    st.markdown(
        t("settings.about_text", name=config.APP_NAME, version=config.APP_VERSION)
    )

    ui.group(t("settings.paths"))
    st.caption(t("settings.uploads_path"))
    st.code(str(config.UPLOAD_DIR), language="text")
    st.caption(t("settings.index_path"))
    st.code(str(config.CHROMA_DIR), language="text")


# --------------------------------------------------------------------------- #
# Einstiegspunkt
# --------------------------------------------------------------------------- #
def main() -> None:
    engine: RegSearchEngine | None = None
    init_error = ""
    try:
        engine = get_engine()
    except Exception as exc:  # Modelle/DB nicht ladbar -> UI trotzdem anzeigen
        init_error = str(exc)

    section = render_nav()

    # Die Einstellungen brauchen keine Engine - sie bleiben auch erreichbar,
    # wenn der Start fehlgeschlagen ist.
    if section == "settings":
        render_settings()
        return

    if engine is None:
        ui.page_title(t("error.title"))
        ui.status(t("error.headline"), "bad")
        st.code(init_error or t("error.unknown"), language="text")
        ui.group(t("error.checklist"))
        st.markdown(t("error.checklist_items"))
        return

    if section == "chat":
        render_chat(engine)
    elif section == "documents":
        render_documents(engine)
    else:
        render_system(engine)


if __name__ == "__main__":
    main()

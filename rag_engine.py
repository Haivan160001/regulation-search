"""
rag_engine.py - Retrieval-Augmented-Generation-Kern von Reg-Search.

Pipeline
--------
    Frage
      |
      | 1) Dense Retrieval  - BAAI/bge-m3 -> ChromaDB (Top-K = 20)   [Recall]
      v
    Kandidaten
      |
      | 2) Reranking        - BAAI/bge-reranker-large (Top-N = 5)    [Precision]
      v
    Kontext (+ Fundstellen)
      |
      | 3) Generierung      - Ollama (qwen2.5:14b / llama3.1:8b)
      v
    Antwort mit Quellenangaben

Persistenz
----------
ChromaDB laeuft **embedded** wie eine lokale SQLite-Datei: kein Server, kein
Docker. Der Client wird hier ueber ``chromadb.PersistentClient(path="./chroma_db")``
erzeugt; Vektoren *und* UNECE-Metadaten (Paragraph, Annex, Dateiname, Chunk-ID)
liegen gemeinsam in dieser Collection. In der Streamlit-App werden sowohl der
Client als auch die Engine mit ``@st.cache_resource`` gecacht, damit Modelle und
DB-Verbindung nicht bei jeder Chat-Interaktion neu geladen werden.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import config
from document_processor import (
    DocumentProcessingError,
    ProcessedDocument,
    process_document,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Fehlertypen
# --------------------------------------------------------------------------- #
class RagEngineError(RuntimeError):
    """Basisfehler der Engine."""


class OllamaUnavailableError(RagEngineError):
    """Der Ollama-Dienst ist nicht erreichbar."""


class ModelNotAvailableError(RagEngineError):
    """Das gewuenschte Modell ist in Ollama nicht installiert."""


class VectorStoreError(RagEngineError):
    """Fehler beim Zugriff auf ChromaDB."""


# --------------------------------------------------------------------------- #
# Datenklassen
# --------------------------------------------------------------------------- #
# Fundstellen, die in einer Frage ausdruecklich genannt werden:
# "§5.2.3.3.3", "Para. 5.4.2.2", "Absatz 5.2.3", oder eine freistehende Nummer
# mit mindestens drei Ebenen. Einstufige Nummern ("§5") bleiben aussen vor -
# sie sind zu mehrdeutig, und Messwerte im Fliesstext ("0.3", "52.6") duerfen
# keine Direktabfrage ausloesen.
RE_CITATION_REFERENCE = re.compile(
    r"(?:§+|paragraphs?|para\.?|absatz|abschnitt|ziffer|klausel|clauses?)\s*"
    r"(\d{1,2}(?:\.\d{1,3})+)"
    r"|\b(\d{1,2}(?:\.\d{1,3}){2,})\b",
    re.IGNORECASE,
)
RE_ANNEX_REFERENCE = re.compile(r"\b(?:annex|anhang)\s*(\d{1,2}[a-z]?)\b", re.IGNORECASE)


def parse_citation_reference(query: str) -> tuple[list[str], str]:
    """
    Zieht ausdruecklich genannte Fundstellen aus einer Frage.

    >>> parse_citation_reference("Ist R85 §5.2.3.3.3. relevant fuer BEVs?")
    (['5.2.3.3.3'], '')
    >>> parse_citation_reference("Was steht in Annex 5, Para. 5.4.2.2?")
    (['5.4.2.2'], '5')

    Gibt die Paragraphennummern in der Reihenfolge ihres Auftretens zurueck und
    - falls genannt - die Annex-Nummer.
    """
    numbers: list[str] = []
    for marked, bare in RE_CITATION_REFERENCE.findall(query or ""):
        number = (marked or bare).strip().rstrip(".")
        if number and number not in numbers:
            numbers.append(number)
    annex = RE_ANNEX_REFERENCE.search(query or "")
    return numbers, (annex.group(1) if annex else "")


def is_form_field(text: str) -> bool:
    """
    Ist dieser Chunk eine leere Zeile aus einem Pruefbericht-Muster?

    Annex 3a/3b von UN R85 sind Formulare; ihre Zeilen bestehen im Wesentlichen
    aus Punktfuehrungen::

        15.1.4.
        Maximum 30 minutes power: ....................................... kW

    Als Beleg taugt so eine Zeile nie - sie enthaelt keinen Wert und keine
    Vorschrift. Im Reranking konkurriert sie aber mit echten Fundstellen und
    hat bei der Frage nach der 30-Minuten-Leistung einen Platz in den Top-5
    belegt, waehrend der Abschnitt mit den Pruefbedingungen herausfiel.
    """
    body = text.strip()
    if not body:
        return True
    if config.FORM_FIELD_DOT_RATIO <= 0:
        return False
    return body.count(".") / len(body) >= config.FORM_FIELD_DOT_RATIO


@dataclass
class RetrievedChunk:
    """Ein Treffer der zweistufigen Suche."""

    text: str
    metadata: dict[str, Any]
    vector_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0
    #: True, wenn *kein* Treffer den Relevanzschwellwert erreicht hat. Die
    #: Passage wird dann trotzdem mitgegeben, aber als unsicher gekennzeichnet -
    #: sonst beantwortet das LLM die Frage aus Allgemeinwissen.
    low_confidence: bool = False
    #: True, wenn die Frage diese Fundstelle ausdruecklich genannt hat
    #: ("§5.2.3.3.3") und der Chunk direkt ueber die Metadaten geholt wurde.
    exact_match: bool = False
    #: True, wenn der Chunk als *uebergeordneter* Abschnitt eines Treffers
    #: ergaenzt wurde (5.3 zu 5.3.2) - Rahmenbedingungen, kein eigener Treffer.
    parent_context: bool = False

    @property
    def citation(self) -> str:
        return str(self.metadata.get("citation") or self.metadata.get("source") or "Quelle")

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def body(self) -> str:
        """Text ohne den technischen Struktur-Header (fuer die Anzeige)."""
        if self.text.startswith("[") and "\n" in self.text:
            head, _, rest = self.text.partition("\n")
            if head.endswith("]"):
                return rest.strip()
        return self.text


@dataclass
class IngestResult:
    """Ergebnis einer Indizierung."""

    filename: str
    doc_id: str = ""
    n_chunks: int = 0
    status: str = "ok"  # ok | skipped | error
    message: str = ""
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Torch-/VRAM-Helfer
# --------------------------------------------------------------------------- #
def _torch():
    """Importiert torch lazy (kann fehlen, wenn nur CPU-Betrieb gewuenscht ist)."""
    try:
        import torch

        return torch
    except ImportError:  # pragma: no cover
        return None


def _is_oom(exc: BaseException) -> bool:
    """Erkennt CUDA-Out-of-Memory unabhaengig von der Torch-Version."""
    torch = _torch()
    if torch is not None and isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message


def free_vram() -> None:
    """Gibt den CUDA-Cache frei (nach OOM oder vor grossen Batches)."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover
            pass


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """
    Liefert das passende Schluesselwort fuer die Modellpraezision.

    transformers hat ``torch_dtype`` ab 4.56 zugunsten von ``dtype`` abgeloest;
    die alte Variante warnt und faellt in 5.x weg. Ein falscher Schluessel
    wuerde stillschweigend ignoriert - das Modell laedt dann in fp32 und
    braucht doppelt so viel VRAM.
    """
    key = "torch_dtype"
    try:
        from transformers import __version__ as transformers_version

        major, minor = (int(part) for part in transformers_version.split(".")[:2])
        if (major, minor) >= (4, 56):
            key = "dtype"
    except Exception:  # pragma: no cover - unbekanntes Versionsformat
        pass
    return {key: dtype}


def gpu_status() -> dict[str, Any]:
    """VRAM-Auslastung fuer die Statusanzeige der UI."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return {"available": False}
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(total_bytes / 1024**3, 1),
            "free_gb": round(free_bytes / 1024**3, 1),
            "used_gb": round((total_bytes - free_bytes) / 1024**3, 1),
        }
    except Exception as exc:  # pragma: no cover
        return {"available": True, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Komponenten-Factories (in Streamlit jeweils via @st.cache_resource cachen)
# --------------------------------------------------------------------------- #
def create_chroma_client(path: str | Path | None = None) -> Any:
    """
    Erzeugt den **embedded** ChromaDB-Client.

    ``PersistentClient`` schreibt eine lokale SQLite-Datei plus HNSW-Index nach
    ``./chroma_db`` - es laeuft kein separater Server und kein Docker-Container.
    Der Client ist prozessweit teuer (Dateihandles, Index) und gehoert deshalb
    in Streamlit hinter ``@st.cache_resource``.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:  # pragma: no cover
        raise VectorStoreError(
            "chromadb ist nicht installiert - bitte 'pip install chromadb'."
        ) from exc

    # Zweiter Riegel gegen Telemetrie-Rauschen im Log (siehe config.py)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

    config.ensure_directories()
    target = Path(path) if path else config.CHROMA_DIR
    target.mkdir(parents=True, exist_ok=True)

    try:
        client = chromadb.PersistentClient(
            path=str(target),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    except Exception as exc:
        raise VectorStoreError(
            f"ChromaDB konnte unter '{target}' nicht geoeffnet werden: {exc}\n"
            "Haeufige Ursache: eine zweite laufende Instanz der App greift auf "
            "denselben Ordner zu."
        ) from exc

    logger.info("ChromaDB (embedded) geoeffnet: %s", target)
    return client


def create_embeddings(
    model_name: str | None = None,
    device: str | None = None,
) -> Any:
    """
    Laedt das Embedding-Modell (Default: ``BAAI/bge-m3``).

    VRAM-Optimierung: fp16 auf CUDA und eine begrenzte Sequenzlaenge
    (``EMBEDDING_MAX_LENGTH``) - bge-m3 koennte 8192 Token, was quadratisch
    Speicher kostet und fuer unsere Chunk-Groesse unnoetig ist.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise RagEngineError(
            "langchain-huggingface fehlt - bitte 'pip install langchain-huggingface'."
        ) from exc

    model_name = model_name or config.EMBEDDING_MODEL
    device = config.resolve_device(device)
    torch = _torch()

    model_kwargs: dict[str, Any] = {"device": device}
    if config.USE_FP16 and device.startswith("cuda") and torch is not None:
        # wird an SentenceTransformer(..., model_kwargs=...) durchgereicht
        model_kwargs["model_kwargs"] = _dtype_kwargs(torch.float16)

    encode_kwargs = {
        "batch_size": config.EMBEDDING_BATCH_SIZE,
        "normalize_embeddings": config.EMBEDDING_NORMALIZE,
    }

    logger.info("Lade Embedding-Modell %s auf %s", model_name, device)
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    except Exception as exc:
        if _is_oom(exc):
            free_vram()
            raise RagEngineError(
                "Zu wenig VRAM fuer das Embedding-Modell. Bitte andere GPU-Prozesse "
                "beenden (z. B. 'ollama stop <modell>') oder "
                "REG_SEARCH_DEVICE=cpu setzen."
            ) from exc
        raise RagEngineError(
            f"Embedding-Modell '{model_name}' konnte nicht geladen werden: {exc}\n"
            "Beim ersten Start wird es von HuggingFace geladen - eine "
            "Internetverbindung ist dafuer noetig (~2 GB)."
        ) from exc

    # Sequenzlaenge begrenzen (spart VRAM, beschleunigt das Indizieren)
    try:
        embeddings.client.max_seq_length = config.EMBEDDING_MAX_LENGTH
    except Exception:  # pragma: no cover - andere ST-Version
        logger.debug("max_seq_length konnte nicht gesetzt werden")
    return embeddings


def create_reranker(model_name: str | None = None, device: str | None = None) -> Any:
    """Laedt den Cross-Encoder-Reranker (Default: ``BAAI/bge-reranker-large``)."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover
        raise RagEngineError(
            "sentence-transformers fehlt - bitte 'pip install sentence-transformers'."
        ) from exc

    model_name = model_name or config.RERANKER_MODEL
    device = config.resolve_device(device)
    torch = _torch()

    kwargs: dict[str, Any] = {"max_length": config.RERANKER_MAX_LENGTH, "device": device}
    if config.RERANKER_TRUST_REMOTE_CODE:
        kwargs["trust_remote_code"] = True
    if config.USE_FP16 and device.startswith("cuda") and torch is not None:
        # Parametername wechselte zwischen sentence-transformers v3 und v4/v5
        params = inspect.signature(CrossEncoder.__init__).parameters
        dtype_arg = _dtype_kwargs(torch.float16)
        if "model_kwargs" in params:
            kwargs["model_kwargs"] = dtype_arg
        elif "automodel_args" in params:
            kwargs["automodel_args"] = dtype_arg

    logger.info("Lade Reranker %s auf %s", model_name, device)
    try:
        return CrossEncoder(model_name, **kwargs)
    except Exception as exc:
        if _is_oom(exc):
            free_vram()
            raise RagEngineError(
                "Zu wenig VRAM fuer den Reranker. Alternative: kleineres Modell "
                "'BAAI/bge-reranker-base' via REG_SEARCH_RERANKER_MODEL setzen."
            ) from exc
        message = str(exc).lower()
        if "trust_remote_code" in message or "custom code" in message:
            raise RagEngineError(
                f"'{model_name}' bringt eigenen Modellcode mit, der beim Laden "
                "ausgefuehrt wird. Wenn du dem Anbieter vertraust, setze\n"
                "  REG_SEARCH_TRUST_REMOTE_CODE=1\n"
                "Andernfalls ein Modell ohne Fremdcode waehlen, z. B. "
                "BAAI/bge-reranker-large oder BAAI/bge-reranker-v2-m3."
            ) from exc
        if "einops" in message:
            raise RagEngineError(
                f"'{model_name}' benoetigt zusaetzlich das Paket einops:\n"
                "  pip install einops"
            ) from exc
        raise RagEngineError(
            f"Reranker '{model_name}' konnte nicht geladen werden: {exc}"
        ) from exc


def create_vectorstore(client: Any, embeddings: Any, collection_name: str | None = None) -> Any:
    """
    Verbindet LangChain mit der ChromaDB-Collection.

    Die Collection wird zuerst explizit angelegt, damit die Distanzmetrik
    (``hnsw:space``) deterministisch gesetzt ist.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover
        raise VectorStoreError(
            "langchain-chroma fehlt - bitte 'pip install langchain-chroma'."
        ) from exc

    collection_name = collection_name or config.CHROMA_COLLECTION
    try:
        client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE},
        )
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
    except Exception as exc:
        raise VectorStoreError(f"Collection '{collection_name}' nicht nutzbar: {exc}") from exc


def create_ollama_client(base_url: str | None = None, timeout: float | None = None) -> Any:
    """
    Erzeugt den Ollama-HTTP-Client.

    ``timeout`` bewusst parametrisierbar: die Generierung braucht lange
    (Modell-Ladezeit), ein Statuscheck in der UI darf dagegen nicht haengen.
    """
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover
        raise RagEngineError("ollama fehlt - bitte 'pip install ollama'.") from exc
    return ollama.Client(
        host=base_url or config.OLLAMA_BASE_URL,
        timeout=config.OLLAMA_TIMEOUT if timeout is None else timeout,
    )


# --------------------------------------------------------------------------- #
# Ollama-Hilfen
# --------------------------------------------------------------------------- #
def _model_names(list_response: Any) -> list[str]:
    """Extrahiert Modellnamen aus der Antwort von ``client.list()`` (v0.3 & v0.4+)."""
    models = getattr(list_response, "models", None)
    if models is None and isinstance(list_response, dict):
        models = list_response.get("models", [])
    names: list[str] = []
    for entry in models or []:
        name = getattr(entry, "model", None) or getattr(entry, "name", None)
        if name is None and isinstance(entry, dict):
            name = entry.get("model") or entry.get("name")
        if name:
            names.append(str(name))
    return sorted(names)


def check_ollama(base_url: str | None = None) -> dict[str, Any]:
    """
    Prueft die Erreichbarkeit von Ollama.

    Returns:
        ``{"ok": bool, "models": list[str], "error": str, "url": str}``
    """
    url = base_url or config.OLLAMA_BASE_URL
    try:
        # kurzer Timeout: der Check laeuft bei jedem Streamlit-Rerun
        client = create_ollama_client(url, timeout=5.0)
        return {"ok": True, "models": _model_names(client.list()), "error": "", "url": url}
    except Exception as exc:
        return {
            "ok": False,
            "models": [],
            "error": (
                f"Ollama ist unter {url} nicht erreichbar ({exc.__class__.__name__}). "
                "Bitte den Dienst starten - z. B. mit 'ollama serve' - und pruefen, "
                "ob die Adresse in der Konfiguration stimmt."
            ),
            "url": url,
        }


def loaded_models(base_url: str | None = None) -> list[dict[str, Any]]:
    """Aktuell im VRAM geladene Ollama-Modelle (fuer die Statusanzeige)."""
    try:
        response = create_ollama_client(base_url, timeout=5.0).ps()
    except Exception:
        return []
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    result: list[dict[str, Any]] = []
    for entry in models or []:
        name = getattr(entry, "model", None) or (
            entry.get("model") if isinstance(entry, dict) else None
        )
        size = getattr(entry, "size_vram", None) or (
            entry.get("size_vram") if isinstance(entry, dict) else None
        )
        if name:
            result.append({"model": name, "vram_gb": round((size or 0) / 1024**3, 1)})
    return result


def _event_field(event: Any, name: str) -> Any:
    """Feld aus einer Ollama-Antwort lesen - egal ob Pydantic-Objekt oder dict."""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name)
    return value


def pull_model(model: str, base_url: str | None = None) -> Iterator[dict[str, Any]]:
    """
    Laedt ein Modell per 'ollama pull' nach.

    Liefert je Fortschrittsmeldung ``{"status", "completed", "total"}``. Die
    Anzeige bleibt Sache des Aufrufers - die Engine kennt keine Oberflaeche.

    Kein eigener Timeout: der Download eines 9-GB-Modells dauert laenger als
    jede sinnvolle Gesamtfrist. ``OLLAMA_TIMEOUT`` wirkt hier als Lesefrist
    zwischen zwei Fortschrittsmeldungen, und die kommen im Sekundentakt.
    """
    client = create_ollama_client(base_url)
    try:
        stream = client.pull(model, stream=True)
    except Exception as exc:  # Dienst weg, Modellname unbekannt, kein Netz
        raise RagEngineError(f"'ollama pull {model}' fehlgeschlagen: {exc}") from exc
    try:
        for event in stream:
            total = int(_event_field(event, "total") or 0)
            yield {
                "status": str(_event_field(event, "status") or ""),
                "completed": int(_event_field(event, "completed") or 0),
                "total": total,
            }
    except Exception as exc:
        raise RagEngineError(f"'ollama pull {model}' abgebrochen: {exc}") from exc


def remove_model(model: str, base_url: str | None = None) -> None:
    """
    Entfernt ein Modell aus Ollama - das Gegenstueck zu 'ollama rm'.

    Geloescht wird nur die Kopie in Ollama; die Wissensbasis in ChromaDB
    bleibt unberuehrt.
    """
    try:
        create_ollama_client(base_url, timeout=30.0).delete(model)
    except Exception as exc:
        raise RagEngineError(f"'ollama rm {model}' fehlgeschlagen: {exc}") from exc


def _tail_overlap(text: str, tag: str) -> int:
    """Laenge des laengsten Suffixes von ``text``, das ein Praefix von ``tag`` ist."""
    for size in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


SHORT_TITLE_PROMPT = """Name the subject of this vehicle regulation in at most 10 English words.

Rules:
- Only the subject, no identifier, no "Regulation No.", no vehicle categories.
- Drop boilerplate such as "Uniform provisions concerning the approval of".
- Reply with the subject and nothing else.

Example
Title: Regulation No. 155 of the Economic Commission for Europe of the United Nations (UN/ECE) - Uniform provisions concerning the approval of vehicles with regards to cyber security and cyber security management system
Subject: Cyber security and management system

Title: {title}
Subject:"""


#: Woerter, die am Ende eines gekuerzten Titels haengen bleiben koennen
#: ("... measurement of") und ihn unfertig aussehen lassen.
_DANGLING_WORDS = frozenset(
    {"of", "and", "or", "for", "with", "to", "in", "on", "the", "a", "an", "at", "by"}
)


def _trim_at_word(text: str, max_chars: int) -> str:
    """Kuerzt auf ``max_chars`` - an einer Wortgrenze und ohne Bindewort am Ende."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    words = text[:max_chars].split(" ")[:-1]
    while words and words[-1].lower().strip(",;:") in _DANGLING_WORDS:
        words.pop()
    return " ".join(words).rstrip(" -–—,;:")


def clean_short_title(raw: str, max_chars: int) -> str:
    """
    Macht aus der Modellantwort einen anzeigbaren Kurztitel - oder nichts.

    Modelle liefern hier gern Beiwerk: Anfuehrungszeichen, ein wiederholtes
    "Label:", eine Begruendung in der zweiten Zeile, bei Reasoning-Modellen
    eine Gedankenkette. Was danach nicht wie ein Titel aussieht, wird
    verworfen - lieber der lange Originaltitel als ein erfundener kurzer.
    """
    text = "".join(strip_thinking(iter([raw or ""])))
    for line in text.splitlines():
        line = line.strip().strip("\"'` ")
        if line.lower().startswith("subject:"):
            line = line[len("subject:") :].strip()
        if not line:
            continue
        line = " ".join(line.split())
        # Eine Erlaeuterung statt eines Titels ("Here is the short title ...").
        # Bewusst eine feste Schranke und nicht ``max_chars``: die Anzeigebreite
        # entscheidet ueber das Kuerzen, nicht darueber, ob die Antwort taugt.
        if len(line) > 200:
            return ""
        return _trim_at_word(line, max_chars)
    return ""


def strip_thinking(stream: Iterator[str]) -> Iterator[str]:
    """
    Entfernt ``<think>...</think>``-Bloecke aus dem Token-Strom.

    Reasoning-Modelle (z. B. qwen3) geben ihre Gedankenkette je nach
    Ollama-Version im Feld ``thinking`` *oder* als Inline-Tags im Text aus.
    Im zweiten Fall wuerde die Gedankenkette sonst in der Antwort landen.
    Da Tags ueber Token-Grenzen hinweg zerschnitten ankommen, wird ein
    moeglicher Tag-Anfang zurueckgehalten, bis er eindeutig ist.
    """
    open_tag, close_tag = "<think>", "</think>"
    buffer = ""
    inside = False

    for token in stream:
        buffer += token
        while buffer:
            if inside:
                end = buffer.find(close_tag)
                if end == -1:
                    keep = _tail_overlap(buffer, close_tag)
                    buffer = buffer[len(buffer) - keep :] if keep else ""
                    break
                buffer = buffer[end + len(close_tag) :]
                inside = False
                continue

            start = buffer.find(open_tag)
            if start == -1:
                keep = _tail_overlap(buffer, open_tag)
                emit = buffer[: len(buffer) - keep] if keep else buffer
                if emit:
                    yield emit
                buffer = buffer[len(buffer) - keep :] if keep else ""
                break

            if start:
                yield buffer[:start]
            buffer = buffer[start + len(open_tag) :]
            inside = True

    if buffer and not inside:
        yield buffer


def _chunk_content(part: Any) -> str:
    """Liest den Token-Text aus einem Ollama-Streaming-Event (dict oder Pydantic)."""
    try:
        message = part["message"] if not hasattr(part, "message") else part.message
    except Exception:
        return ""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content or ""


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RegSearchEngine:
    """
    Buendelt Vektorspeicher, Reranker und LLM zu einer RAG-Pipeline.

    Die Instanz ist zustandsbehaftet und teuer (Modelle im VRAM) - in Streamlit
    daher genau einmal per ``@st.cache_resource`` erzeugen.
    """

    def __init__(
        self,
        chroma_client: Any | None = None,
        embeddings: Any | None = None,
        reranker: Any | None = None,
        llm_model: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.collection_name = collection_name or config.CHROMA_COLLECTION
        self.llm_model = llm_model or config.LLM_MODEL

        self.chroma_client = chroma_client or create_chroma_client()
        self.embeddings = embeddings or create_embeddings()
        self.vectorstore = create_vectorstore(
            self.chroma_client, self.embeddings, self.collection_name
        )
        # Der Reranker wird erst beim ersten Suchlauf geladen (spart Startzeit
        # und VRAM, falls nur indiziert wird).
        self._reranker = reranker
        self.ollama = create_ollama_client()

    # ---------------------------------------------------------------- Chroma
    @property
    def collection(self) -> Any:
        """Rohe ChromaDB-Collection (fuer count/get/delete mit Metadatenfiltern)."""
        try:
            return self.chroma_client.get_or_create_collection(name=self.collection_name)
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Collection nicht verfuegbar: {exc}") from exc

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            self._reranker = create_reranker()
        return self._reranker

    def count_chunks(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:  # pragma: no cover
            logger.warning("count() fehlgeschlagen: %s", exc)
            return 0

    def list_documents(self) -> list[dict[str, Any]]:
        """Aggregiert die Metadaten der Collection zu einer Dokumentenliste."""
        try:
            payload = self.collection.get(include=["metadatas"])
        except Exception as exc:  # pragma: no cover
            raise VectorStoreError(f"Metadaten konnten nicht gelesen werden: {exc}") from exc

        documents: dict[str, dict[str, Any]] = {}
        for metadata in payload.get("metadatas") or []:
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id", "unbekannt"))
            entry = documents.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source": metadata.get("source", "?"),
                    "source_path": metadata.get("source_path", ""),
                    "regulation": metadata.get("regulation", ""),
                    "doc_title": metadata.get("doc_title", ""),
                    "short_title": metadata.get("short_title", ""),
                    "n_pages": metadata.get("n_pages", 0),
                    "file_type": metadata.get("file_type", ""),
                    "ingested_at": metadata.get("ingested_at", ""),
                    "chunks": 0,
                    "annexes": set(),
                },
            )
            entry["chunks"] += 1
            if metadata.get("annex"):
                entry["annexes"].add(str(metadata["annex"]))

        result = []
        for entry in documents.values():
            entry["annexes"] = sorted(entry["annexes"], key=lambda a: (len(a), a))
            result.append(entry)
        return sorted(result, key=lambda item: str(item["source"]).lower())

    def stats(self) -> dict[str, Any]:
        """Kennzahlen fuer die Statusanzeige der Sidebar."""
        try:
            documents = self.list_documents()
        except VectorStoreError:
            documents = []
        return {
            "chunks": self.count_chunks(),
            "documents": len(documents),
            "sources": [str(doc["source"]) for doc in documents],
            "collection": self.collection_name,
            "path": str(config.CHROMA_DIR),
        }

    def document_exists(self, doc_id: str) -> bool:
        try:
            found = self.collection.get(where={"doc_id": doc_id}, limit=1, include=[])
            return bool(found.get("ids"))
        except Exception:  # pragma: no cover
            return False

    def delete_document(self, doc_id: str) -> int:
        """Entfernt alle Chunks eines Dokuments. Gibt die Anzahl zurueck."""
        try:
            existing = self.collection.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Loeschen fehlgeschlagen: {exc}") from exc

    def reset_collection(self) -> None:
        """Loescht die gesamte Collection (Index bleibt nutzbar, ist danach leer)."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Collection konnte nicht geleert werden: {exc}") from exc
        finally:
            self.vectorstore = create_vectorstore(
                self.chroma_client, self.embeddings, self.collection_name
            )

    # -------------------------------------------------------------- Ingestion
    def short_title(self, long_title: str, regulation: str = "", model: str | None = None) -> str:
        """
        Bildet aus dem Langtitel einen kurzen, englischen Anzeigetitel.

        UNECE-Titel sind Satzungetueme ("Regulation No 85 ... Uniform
        provisions concerning the approval of internal combustion engines or
        electric drive trains intended for the propulsion of motor vehicles of
        categories M and N with regard to the measurement of net power and the
        maximum 30 minutes power of electric drive trains"). In einer Liste
        unterscheidet davon nichts, weil alle Titel gleich anfangen.

        Das Modell liefert nur das *Sachthema*; die Kennung stammt aus den
        bereits geparsten Metadaten und wird hier davorgesetzt. So kann kein
        "Regulation No. -" ohne Nummer entstehen, wenn der Titel keine enthaelt.

        Gibt bei jedem Fehler "" zurueck: ein Upload darf nicht daran
        scheitern, dass Ollama gerade nicht laeuft.
        """
        title = " ".join((long_title or "").split())
        if not title or not config.SHORT_TITLE_ENABLED:
            return ""

        messages = [{"role": "user", "content": SHORT_TITLE_PROMPT.format(title=title)}]
        options = {"temperature": 0.0, "top_p": 1.0, "num_predict": 64}
        try:
            # think=False: Reasoning-Modelle wie qwen3 verbrauchen sonst das
            # gesamte Token-Budget mit der Gedankenkette und liefern nichts.
            response = self.ollama.chat(
                model=model or self.llm_model, messages=messages, options=options, think=False
            )
        except TypeError:  # aeltere ollama-Clients kennen 'think' nicht
            try:
                response = self.ollama.chat(
                    model=model or self.llm_model,
                    messages=messages,
                    options={**options, "num_predict": 512},
                )
            except Exception as exc:
                logger.warning("Kurztitel nicht erzeugbar: %s", exc)
                return ""
        except Exception as exc:
            logger.warning("Kurztitel nicht erzeugbar: %s", exc)
            return ""

        message = getattr(response, "message", None) or (
            response.get("message") if isinstance(response, dict) else None
        )
        content = getattr(message, "content", None) or (
            message.get("content") if isinstance(message, dict) else ""
        )
        subject = clean_short_title(str(content or ""), config.SHORT_TITLE_MAX_CHARS)
        if not subject:
            return ""

        identifier = " ".join((regulation or "").split())
        if not identifier:
            return _trim_at_word(subject, config.SHORT_TITLE_MAX_CHARS)
        return _trim_at_word(
            f"{identifier} — {subject}", config.SHORT_TITLE_MAX_CHARS
        )

    def index_file(
        self,
        path: str | Path,
        force: bool = False,
        progress_callback: Any | None = None,
    ) -> IngestResult:
        """
        Verarbeitet eine Datei und schreibt ihre Chunks nach ChromaDB.

        Doppelte Dateien werden anhand der inhaltsbasierten ``doc_id``
        erkannt und uebersprungen (``force=True`` erzwingt Neuindizierung).
        """
        started = time.perf_counter()
        path = Path(path)
        result = IngestResult(filename=path.name)

        try:
            processed: ProcessedDocument = process_document(path)
        except DocumentProcessingError as exc:
            result.status = "error"
            result.message = str(exc)
            return result

        result.doc_id = processed.doc_id
        result.warnings = processed.warnings

        if self.document_exists(processed.doc_id):
            if not force:
                result.status = "skipped"
                result.message = "Bereits indiziert (identischer Dateiinhalt)."
                return result
            self.delete_document(processed.doc_id)

        # Kurztitel fuer die Dokumentliste - rein zur Anzeige, er geht nicht
        # in die Einbettung ein und aendert damit keine Suchergebnisse.
        if config.SHORT_TITLE_ENABLED:
            title = self.short_title(
                str(processed.metadata.get("doc_title", "")),
                regulation=str(processed.metadata.get("regulation", "")),
            )
            if title:
                for chunk in processed.chunks:
                    chunk.metadata["short_title"] = title
            else:
                result.warnings = list(result.warnings) + [
                    "Kurztitel konnte nicht erzeugt werden - angezeigt wird der Originaltitel."
                ]

        try:
            from langchain_core.documents import Document
        except ImportError as exc:  # pragma: no cover
            raise RagEngineError("langchain-core fehlt.") from exc

        documents = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in processed.chunks
        ]
        ids = [chunk.metadata["chunk_id"] for chunk in processed.chunks]

        batch_size = config.INGEST_BATCH_SIZE
        written = 0
        index = 0
        while index < len(documents):
            batch = documents[index : index + batch_size]
            batch_ids = ids[index : index + batch_size]
            try:
                self.vectorstore.add_documents(documents=batch, ids=batch_ids)
            except Exception as exc:
                if _is_oom(exc) and batch_size > 1:
                    # VRAM-Druck: Batch halbieren und erneut versuchen
                    free_vram()
                    batch_size = max(1, batch_size // 2)
                    self._shrink_embedding_batch()
                    logger.warning("OOM beim Embedden - Batchgroesse -> %s", batch_size)
                    continue
                result.status = "error"
                result.message = (
                    f"Schreiben in ChromaDB fehlgeschlagen: {exc}"
                    + (
                        "\nTipp: GPU-Speicher freigeben ('ollama stop <modell>') "
                        "oder REG_SEARCH_DEVICE=cpu setzen."
                        if _is_oom(exc)
                        else ""
                    )
                )
                return result

            written += len(batch)
            index += len(batch)
            if progress_callback:
                try:
                    progress_callback(written, len(documents))
                except Exception:  # pragma: no cover - UI darf nie den Ingest killen
                    pass

        result.n_chunks = written
        result.duration_s = round(time.perf_counter() - started, 2)
        result.message = f"{written} Chunks indiziert."
        return result

    def _shrink_embedding_batch(self) -> None:
        """Halbiert die Encode-Batchgroesse des Embedding-Modells nach einem OOM."""
        try:
            encode_kwargs = getattr(self.embeddings, "encode_kwargs", None)
            if isinstance(encode_kwargs, dict):
                encode_kwargs["batch_size"] = max(1, int(encode_kwargs.get("batch_size", 8)) // 2)
        except Exception:  # pragma: no cover
            pass

    # -------------------------------------------------------------- Retrieval
    def _lookup_reference(
        self, query: str, where: dict[str, Any] | None, limit: int
    ) -> list[RetrievedChunk]:
        """
        Holt Chunks, deren Fundstelle in der Frage ausdruecklich genannt wird.

        Dichte Embeddings kodieren Bedeutung, keine Bezeichner: "5.2.3.3.3",
        "5.2.3.3.6" und "5.2.3.2.3" liegen im Vektorraum praktisch aufeinander.
        Eine Frage nach einer bestimmten Klausel trifft deshalb zufaellig -
        beobachtet an "Ist R85 §5.2.3.3.3 relevant fuer BEVs?": der Absatz war
        sauber indiziert, lag aber nicht einmal unter den Top-20, und das
        Modell musste die Auskunft verweigern. Die Nummer steht als Metadatum
        bereit, wird hier direkt abgefragt und der Vektorsuche vorangestellt.
        """
        numbers, annex = parse_citation_reference(query)
        if not numbers:
            return []

        conditions: list[dict[str, Any]] = [
            {"paragraph": numbers[0] if len(numbers) == 1 else {"$in": numbers}}
        ]
        if annex:
            conditions.append({"annex": annex})
        if where:
            conditions.append(where)
        criteria = conditions[0] if len(conditions) == 1 else {"$and": conditions}

        try:
            found = self.collection.get(
                where=criteria, limit=limit, include=["documents", "metadatas"]
            )
        except Exception as exc:  # pragma: no cover - Filtersyntax je Chroma-Version
            logger.warning("Direktabfrage der Fundstelle fehlgeschlagen: %s", exc)
            return []

        chunks = [
            RetrievedChunk(text=text, metadata=dict(meta or {}), exact_match=True)
            for text, meta in zip(found.get("documents") or [], found.get("metadatas") or [])
        ]
        chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        if chunks:
            logger.info(
                "Fundstelle direkt aufgeloest: %s -> %s Chunk(s)", ", ".join(numbers), len(chunks)
            )
        return chunks

    def _parent_sections(
        self, selected: Sequence[RetrievedChunk], where: dict[str, Any] | None
    ) -> list[RetrievedChunk]:
        """
        Laedt die uebergeordneten Abschnitte der Treffer nach.

        In UNECE-Texten steht der Rahmen im Elternabschnitt: 5.3.2 beschreibt
        die Messung der 30-Minuten-Leistung, aber der Aufbau (Ausruestung nach
        Annex 6, DC-Quelle mit hoechstens 5 % Spannungsabfall) steht in 5.3.
        Der Cross-Encoder findet solche Rahmenabschnitte nicht - sie enthalten
        die Frageworte nicht. Ueber die Nummer sind sie dagegen exakt
        bestimmbar, ohne Modell und ohne Zufall.
        """
        if not config.RETRIEVAL_PARENT_CONTEXT:
            return []

        present = {
            (c.metadata.get("annex", ""), c.metadata.get("appendix", ""), c.metadata.get("paragraph", ""))
            for c in selected
        }
        # Nur zu den *fuehrenden* Treffern: deren Rahmen braucht die Antwort.
        # Der Elternabschnitt eines schwachen Treffers auf Platz 5 waere meist
        # eine blosse Ueberschrift ("2. Definitions") und verduennt den Kontext.
        wanted: list[tuple[str, str, str]] = []
        for chunk in selected[: config.RETRIEVAL_MAX_PARENTS]:
            paragraph = str(chunk.metadata.get("paragraph") or "")
            if "." not in paragraph:
                continue
            key = (
                str(chunk.metadata.get("annex", "")),
                str(chunk.metadata.get("appendix", "")),
                paragraph.rsplit(".", 1)[0],
            )
            if key not in present and key not in wanted:
                wanted.append(key)

        parents: list[RetrievedChunk] = []
        for annex, appendix, paragraph in wanted[: config.RETRIEVAL_MAX_PARENTS]:
            # Nur nach der Nummer filtern: leere Metadaten werden beim
            # Indizieren verworfen (sanitize_metadata), ein Filter auf
            # annex="" traefe deshalb nichts. Die Zuordnung zu Annex und
            # Appendix passiert danach in Python - dort ist ein fehlender
            # Schluessel gleichbedeutend mit "Hauptteil".
            criteria: dict[str, Any] = (
                {"$and": [{"paragraph": paragraph}, where]} if where else {"paragraph": paragraph}
            )
            try:
                found = self.collection.get(
                    where=criteria, limit=8, include=["documents", "metadatas"]
                )
            except Exception as exc:  # pragma: no cover - Filtersyntax je Chroma-Version
                logger.warning("Elternabschnitt %s nicht ladbar: %s", paragraph, exc)
                continue
            for text, meta in zip(found.get("documents") or [], found.get("metadatas") or []):
                meta = dict(meta or {})
                if str(meta.get("annex", "")) != annex or str(meta.get("appendix", "")) != appendix:
                    continue
                parents.append(RetrievedChunk(text=text, metadata=meta, parent_context=True))
                break
        if parents:
            logger.info(
                "Elternkontext ergaenzt: %s",
                ", ".join(str(c.metadata.get("paragraph")) for c in parents),
            )
        return parents

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_n: int | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Zweistufige Suche: Vektorsuche (Top-K) -> Cross-Encoder-Rerank (Top-N).

        Args:
            sources: optionale Einschraenkung auf bestimmte Dateinamen.
        """
        query = (query or "").strip()
        if not query:
            return []

        top_k = top_k or config.RETRIEVAL_TOP_K
        top_n = top_n or config.RERANK_TOP_N

        total_chunks = self.count_chunks()
        if total_chunks == 0:
            return []
        # Chroma warnt, wenn mehr Treffer angefordert werden als vorhanden sind
        top_k = max(1, min(top_k, total_chunks))

        where: dict[str, Any] | None = None
        if sources:
            where = (
                {"source": sources[0]}
                if len(sources) == 1
                else {"source": {"$in": list(sources)}}
            )

        # --- Stufe 0: ausdruecklich genannte Fundstelle ---------------------
        exact = self._lookup_reference(query, where, max(top_n, 3))
        exact_ids = {
            c.metadata.get("chunk_id") for c in exact if c.metadata.get("chunk_id")
        }

        # --- Stufe 1: Dense Retrieval ---------------------------------------
        try:
            hits = self.vectorstore.similarity_search_with_score(query, k=top_k, filter=where)
        except Exception as exc:
            if _is_oom(exc):
                free_vram()
                raise RagEngineError(
                    "GPU-Speicher erschoepft bei der Vektorsuche. Bitte das LLM "
                    "entladen ('ollama stop <modell>') und erneut versuchen."
                ) from exc
            raise VectorStoreError(f"Vektorsuche fehlgeschlagen: {exc}") from exc

        if not hits:
            for position, chunk in enumerate(exact, start=1):
                chunk.rank = position
            return exact

        candidates = [
            RetrievedChunk(
                text=document.page_content,
                metadata=dict(document.metadata or {}),
                # Chroma liefert bei 'cosine' eine Distanz in [0, 2]
                vector_score=round(1.0 - float(distance), 4),
            )
            for document, distance in hits
        ]
        candidates = exact + [
            c for c in candidates if c.metadata.get("chunk_id") not in exact_ids
        ]

        # Leere Formularzeilen belegen nichts. Sie fliegen raus, solange etwas
        # anderes uebrig bleibt - sonst waere eine Frage, die tatsaechlich auf
        # ein Pruefbericht-Muster zielt, gar nicht mehr beantwortbar.
        substantive = [c for c in candidates if not is_form_field(c.body)]
        if substantive:
            candidates = substantive

        # --- Stufe 2: Cross-Encoder-Reranking -------------------------------
        try:
            scores = self._rerank_scores(query, [c.text for c in candidates])
        except RagEngineError:
            raise
        except Exception as exc:  # pragma: no cover
            logger.warning("Reranking fehlgeschlagen (%s) - nutze Vektor-Ranking", exc)
            scores = [c.vector_score for c in candidates]

        for chunk, score in zip(candidates, scores):
            chunk.rerank_score = round(float(score), 4)

        # Hat der Cross-Encoder ueberhaupt etwas unterschieden? Liegen alle
        # Kandidaten gleichauf, ist seine Sortierung Rauschen (typisch bei
        # deutscher Frage auf englischen Text) - dann bleibt die Reihenfolge
        # der mehrsprachigen Vektorsuche stehen.
        spread = max(scores) - min(scores) if len(scores) > 1 else 0.0

        if spread < config.RERANK_MIN_SPREAD:
            logger.info(
                "Reranker ohne Trennschaerfe (Spreizung %.4f) - Vektor-Ranking "
                "wird beibehalten fuer: %.60s",
                spread,
                query,
            )
            selected = sorted(candidates, key=lambda c: c.vector_score, reverse=True)[:top_n]
            for chunk in selected:
                chunk.low_confidence = True
        else:
            ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
            selected = [c for c in ranked if c.rerank_score >= config.RERANK_MIN_SCORE][:top_n]
            if not selected:
                # Nichts ueber dem Schwellwert: etwas Kontext mitgeben, aber
                # als unsicher kennzeichnen - Prompt und UI weisen darauf hin.
                selected = ranked[: min(top_n, 3)]
                for chunk in selected:
                    chunk.low_confidence = True
                logger.info("Retrieval mit geringer Konfidenz fuer: %.60s", query)

        if exact:
            # Der Schwellwert entscheidet ueber *semantische* Treffer. Eine
            # ausdruecklich genannte Fundstelle ist keine Schaetzung - sie
            # bleibt drin, steht vorn und gilt nicht als unsicher.
            rest = [c for c in selected if c.metadata.get("chunk_id") not in exact_ids]
            for chunk in exact:
                chunk.low_confidence = False
            selected = exact + rest[: max(0, top_n - len(exact))]

        selected = selected + self._parent_sections(selected, where)

        for position, chunk in enumerate(selected, start=1):
            chunk.rank = position
        return selected

    def _rerank_scores(self, query: str, passages: Sequence[str]) -> list[float]:
        """Berechnet Cross-Encoder-Scores, mit Batch-Verkleinerung bei OOM."""
        pairs = [(query, passage) for passage in passages]
        batch_size = config.RERANKER_BATCH_SIZE
        while True:
            try:
                scores = self.reranker.predict(
                    pairs, batch_size=batch_size, show_progress_bar=False
                )
                return [float(score) for score in scores]
            except Exception as exc:
                if _is_oom(exc) and batch_size > 1:
                    free_vram()
                    batch_size = max(1, batch_size // 2)
                    logger.warning("OOM beim Reranking - Batchgroesse -> %s", batch_size)
                    continue
                if _is_oom(exc):
                    free_vram()
                    raise RagEngineError(
                        "GPU-Speicher erschoepft beim Reranking. Bitte ein kleineres "
                        "LLM waehlen oder REG_SEARCH_DEVICE=cpu setzen."
                    ) from exc
                raise

    # ------------------------------------------------------------- Generation
    @staticmethod
    def build_context(chunks: Sequence[RetrievedChunk], max_chars: int = 12000) -> str:
        """Formatiert die Treffer als nummerierte, zitierbare Kontextbloecke."""
        blocks: list[str] = []
        used = 0
        for index, chunk in enumerate(chunks, start=1):
            header = f"[{index}] {chunk.citation} (Datei: {chunk.metadata.get('source', '?')})"
            block = f"{header}\n{chunk.body}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def _messages(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        for turn in list(history or [])[-config.CHAT_HISTORY_TURNS * 2 :]:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:4000]})
        prompt = config.ANSWER_PROMPT.format(
            context=self.build_context(chunks), question=question
        )
        if any(chunk.low_confidence for chunk in chunks):
            prompt = config.LOW_CONFIDENCE_NOTICE + "\n\n" + prompt
        messages.append({"role": "user", "content": prompt})
        return messages

    def stream_answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict[str, str]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """
        Streamt die Antwort tokenweise (fuer die UI).

        Eine eventuelle Gedankenkette von Reasoning-Modellen wird herausgefiltert.
        """
        if not chunks:
            yield config.NO_CONTEXT_MESSAGE
            return
        yield from strip_thinking(
            self._raw_stream(question, chunks, history, model, temperature)
        )

    def _raw_stream(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict[str, str]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Rohe Token von Ollama, ohne Nachbearbeitung."""
        model = model or self.llm_model
        options = {
            "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
            "top_p": config.LLM_TOP_P,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
        }

        try:
            stream = self.ollama.chat(
                model=model,
                messages=self._messages(question, chunks, history),
                stream=True,
                options=options,
                keep_alive=config.LLM_KEEP_ALIVE,
            )
            for part in stream:
                token = _chunk_content(part)
                if token:
                    yield token
        except Exception as exc:
            raise self._translate_ollama_error(exc, model) from exc

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        top_n: int | None = None,
        sources: Sequence[str] | None = None,
        history: Sequence[dict[str, str]] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Kompletter RAG-Durchlauf ohne Streaming (CLI/Tests)."""
        started = time.perf_counter()
        chunks = self.retrieve(question, top_k=top_k, top_n=top_n, sources=sources)
        text = "".join(
            self.stream_answer(
                question, chunks, history=history, model=model, temperature=temperature
            )
        )
        return {
            "answer": text,
            "sources": chunks,
            "duration_s": round(time.perf_counter() - started, 2),
            "model": model or self.llm_model,
        }

    @staticmethod
    def _translate_ollama_error(exc: Exception, model: str) -> RagEngineError:
        """Uebersetzt technische Ollama-Fehler in handlungsleitende Meldungen."""
        message = str(exc).lower()
        if "not found" in message or "no such model" in message or "404" in message:
            return ModelNotAvailableError(
                f"Das Modell '{model}' ist in Ollama nicht installiert.\n"
                f"Bitte im Terminal ausfuehren:  ollama pull {model}"
            )
        if any(
            token in message
            for token in ("connection", "connect", "refused", "timeout", "timed out")
        ):
            return OllamaUnavailableError(
                f"Ollama ({config.OLLAMA_BASE_URL}) antwortet nicht: {exc}\n"
                "Laeuft der Dienst? Test:  ollama list"
            )
        if "memory" in message or "vram" in message:
            free_vram()
            return RagEngineError(
                f"Ollama meldet Speicherprobleme beim Laden von '{model}': {exc}\n"
                "Tipp: kleineres Modell (z. B. llama3.1:8b) waehlen oder "
                "REG_SEARCH_NUM_CTX verkleinern."
            )
        return RagEngineError(f"Fehler bei der Antwortgenerierung: {exc}")

    # ------------------------------------------------------------------ Misc
    def warmup(self) -> None:
        """Laedt Embedding- und Reranker-Modell vorab (macht die erste Frage schnell)."""
        try:
            self.embeddings.embed_query("warmup")
            self.reranker.predict([("warmup", "warmup")], show_progress_bar=False)
        except Exception as exc:  # pragma: no cover
            logger.debug("Warmup uebersprungen: %s", exc)

    def available_models(self) -> list[str]:
        return check_ollama().get("models", [])


__all__ = [
    "RegSearchEngine",
    "RetrievedChunk",
    "IngestResult",
    "RagEngineError",
    "OllamaUnavailableError",
    "ModelNotAvailableError",
    "VectorStoreError",
    "create_chroma_client",
    "create_embeddings",
    "create_reranker",
    "create_vectorstore",
    "check_ollama",
    "loaded_models",
    "strip_thinking",
    "gpu_status",
    "free_vram",
]

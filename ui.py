"""
ui.py - Erscheinungsbild der Reg-Search-Oberflaeche.

Gestaltungsprinzipien
---------------------
Die Oberflaeche ist ein Einstellungs-Dashboard im Dark Mode, wie es bei
Entwickler-Werkzeugen ueblich ist. Die Graustufen liegen bewusst eng
beieinander; getrennt wird ueber Rahmen, nicht ueber Fuellung.

Im Einzelnen:

* die dunklere Sidebar (#131313) vor hellerer Arbeitsflaeche (#212121) -
  ungewoehnlich herum, haelt die Navigation aber optisch im Hintergrund,
* Karten ohne eigene Fuellung: gleiche Farbe wie die Seite, nur ein Rahmen
  (#3C3C3C). Eingaben bekommen einen helleren Rahmen (#595959),
* die Navigation als schlichte Zeilenliste, aktiv nur durch eine graue
  Flaeche (#303030) markiert,
* Seitentitel klein und zurueckhaltend (1.5rem), direkt darunter die
  Reiter, darunter der Inhalt,
* eine einzige Schrift fuer die gesamte Oberflaeche.

Bewusste Setzungen
------------------
* Genau **eine** Farbe im Layout: das Lime ``#72C616``. Es markiert
  Primaerbuttons, den aktiven Reiter, Fokusrahmen und Fortschritt - alles
  Uebrige bleibt grau, damit ein Akzent auch etwas bedeutet.
* Geist Mono bleibt fuer Code, Pfade und Modellnamen. Ein UNECE-Dateipfad in
  einer Proportionalschrift ist schlecht lesbar.

Arbeitsteilung mit .streamlit/config.toml
-----------------------------------------
Farben, Radien, Schriften und Statustoene stehen als ``[theme]`` in der
``config.toml``. Dieses Modul ergaenzt nur, was die Theme-Optionen nicht
abdecken: Layout, Navigation, Reiter und die eigenen Bauteile.

Hinweis zu den Selektoren
-------------------------
Angesprochen werden ``data-testid``-Attribute (von Streamlit als stabile
Testanker gepflegt), ``data-baseweb``-Attribute und ``st-key-*``-Klassen, die
Streamlit fuer Elemente mit ``key=`` vergibt - keine generierten
Emotion-Klassennamen, die sich mit jedem Release aendern.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# Design-Tokens (identisch mit [theme] in .streamlit/config.toml)
# --------------------------------------------------------------------------- #
SIDEBAR = "#131313"
SURFACE = "#212121"
RAISED = "#303030"
BORDER = "#3C3C3C"
BORDER_INPUT = "#595959"
FOREGROUND = "#DCDCDC"
FOREGROUND_MUTED = "#AFAFAF"
ACCENT = "#72C616"

_CSS = """
<style>
/* --- Tokens ------------------------------------------------------------- */
:root {
  --rs-sidebar:      #131313;
  --rs-surface:      #212121;
  --rs-raised:       #303030;      /* Nav aktiv, Hover */
  --rs-sunken:       #1A1A1A;      /* Code, Eingabe-Fuellung in der Sidebar */
  --rs-border:       #3C3C3C;      /* Karten, Trennlinien */
  --rs-border-input: #595959;      /* Eingaben - bewusst heller */
  --rs-fg:           #DCDCDC;
  --rs-fg-bright:    #FFFFFF;
  --rs-fg-muted:     #AFAFAF;

  --rs-accent:        #72C616;
  --rs-accent-bright: #8AD82A;
  --rs-accent-ink:    #10180A;     /* Text auf Lime-Flaechen */
  --rs-accent-soft:   rgba(114, 198, 22, .14);
  --rs-accent-line:   rgba(114, 198, 22, .35);

  --rs-radius:    .375rem;         /* Eingaben, Buttons */
  --rs-radius-lg: .5rem;           /* Karten */

  --rs-nav-width: 216px;
  --rs-content:   980px;

  --rs-font: "DM Sans", "Segoe UI", system-ui, sans-serif;
  --rs-mono: "Geist Mono", "Cascadia Code", Consolas, monospace;
}

/* Der Container des eingefuegten <style> wuerde sonst als Leerzeile wirken. */
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
  display: none !important;
}

/* --- Kopfleiste: unsichtbar, aber nicht entfernt ------------------------- */
/* Wichtig: der einzige Knopf zum Wiederaufklappen einer eingeklappten
   Sidebar (stExpandSidebarButton) sitzt in dieser Leiste. Sie wird deshalb
   nur transparent und klickdurchlaessig gemacht - nicht ausgeblendet. */
[data-testid="stHeader"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  pointer-events: none;
}
[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {
  pointer-events: auto;
}
/* stToolbar hier NICHT ausblenden: der Aufklapp-Knopf der eingeklappten
   Sidebar liegt darin. Ausgeblendet wird nur stToolbarActions - das ist der
   Block mit Deploy-Button und Hamburger-Menue. */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {
  display: none !important;
}
[data-testid="stStatusWidget"] { display: none !important; }

/* --- Grundflaechen ------------------------------------------------------ */
[data-testid="stAppViewContainer"] { background: var(--rs-surface); }

/* Inhalt linksbuendig, nicht zentriert - wie in der Vorlage.
   Der scrollende Bereich ist ein Flex-Container mit align-items:center.
   Achtung: sobald eine Chat-Eingabe auf der Seite liegt, heisst sein
   data-testid nicht mehr "stMain", sondern "stAppScrollToBottomContainer" -
   die Klasse .stMain ist aber in beiden Faellen gesetzt. */
section.stMain { align-items: flex-start !important; }
[data-testid="stMainBlockContainer"] {
  max-width: var(--rs-content);
  padding: 1.5rem 2rem 4rem 1.75rem;
}

/* Die Chat-Eingabe haengt in einem eigenen Container am unteren Rand und
   wird dort ebenfalls zentriert. align-self wirkt unabhaengig davon, wie der
   umgebende Flex-Container ausrichtet. */
[data-testid="stBottomBlockContainer"] {
  max-width: var(--rs-content) !important;
  align-self: flex-start !important;
  padding-left: 1.75rem !important;
  padding-right: 2rem !important;
}

/* Formularelemente bleiben schmal - die Vorlage zieht Eingaben nie ueber die
   volle Breite. Gilt nur ausserhalb von Spalten, damit mehrspaltige Zeilen
   (Chat-Kopf, Dokumentkarten) unberuehrt bleiben. */
[data-testid="stTabPanel"] [data-testid="stSelectbox"],
[data-testid="stTabPanel"] [data-testid="stTextInput"],
[data-testid="stTabPanel"] [data-testid="stNumberInput"],
[data-testid="stTabPanel"] [data-testid="stSlider"],
[data-testid="stTabPanel"] [data-testid="stProgress"],
[data-testid="stTabPanel"] [data-testid="stFileUploader"] {
  max-width: 26rem;
}
[data-testid="stColumn"] [data-testid="stSelectbox"],
[data-testid="stColumn"] [data-testid="stTextInput"],
[data-testid="stColumn"] [data-testid="stNumberInput"] {
  max-width: none;
}

/* --- Sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] {
  background: var(--rs-sidebar);
  border-right: 1px solid #2A2A2A;
}
/* Feste Nav-Breite nur im ausgeklappten Zustand. Streamlit schiebt die
   eingeklappte Sidebar per transform aus dem Bild, behaelt aber ihren Platz
   im Layout - eine erzwungene Breite liesse also eine leere Spalte stehen. */
[data-testid="stSidebar"][aria-expanded="true"] {
  width: var(--rs-nav-width) !important;
  min-width: var(--rs-nav-width) !important;
  max-width: var(--rs-nav-width) !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
  width: 0 !important;
  min-width: 0 !important;
}
/* Bei eingeklappter Sidebar sitzt der Aufklapp-Knopf oben links im Inhalt und
   laege sonst auf dem Seitentitel. Der Platz wird nur in diesem Zustand
   freigeraeumt - Sidebar und Inhaltsbereich sind Geschwister, daher der
   Nachbarschafts-Selektor. */
[data-testid="stSidebar"][aria-expanded="false"] ~ div [data-testid="stMainBlockContainer"] {
  padding-top: 4.25rem;
}
[data-testid="stSidebarContent"] { background: var(--rs-sidebar); }
/* Streamlit polstert die Sidebar doppelt: 17.5 px auf stSidebarContent und
   noch einmal auf stSidebarUserContent. Die Vorlage setzt ihre Nav-Zeilen
   dicht an den Rand. */
[data-testid="stSidebarContent"] {
  padding-left: .5rem !important;
  padding-right: .5rem !important;
}
[data-testid="stSidebarUserContent"] { padding: .35rem .25rem 1rem !important; }
/* Nav-Zeilen dicht stapeln statt mit dem Standardabstand fuer Widgets. */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: .125rem; }
[data-testid="stSidebarHeader"] { padding-bottom: 0; height: 2.4rem; }
/* Der Ziehgriff zum Verbreitern passt nicht zur festen Nav-Breite. */
[data-testid="stSidebarResizeHandle"] { display: none !important; }

/* --- Navigation --------------------------------------------------------- */
/* Nav-Zeilen sind tertiaere Buttons. Aktiv/inaktiv unterscheiden sich ueber
   den key-Praefix (rs-navon- / rs-nav-), den app.py setzt. */
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"] {
  display: flex !important;
  width: 100% !important;
  justify-content: flex-start;
  text-align: left;
  padding: .34rem .6rem;
  min-height: 1.75rem;
  border: 1px solid transparent;
  border-radius: var(--rs-radius);
  color: var(--rs-fg);
  font-size: .8125rem;
  font-weight: 500;
  line-height: 1.35;
  transition: background .12s ease, color .12s ease;
}
/* Streamlit schachtelt die Beschriftung in zwei zentrierende Flex-Ebenen
   (button > div > span). Ohne diese Regel steht der Text mittig in der Zeile,
   egal was am Button selbst gesetzt ist. */
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"] > div,
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"] > div > span {
  width: 100%;
  justify-content: flex-start !important;
}
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"]
  [data-testid="stMarkdownContainer"] {
  width: 100%;
  text-align: left;
}
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"] p {
  font-size: .875rem;          /* 14 px - wie die Nav der Vorlage */
  font-weight: inherit;
  text-align: left;
}
/* Nach einem Mausklick keinen Ring stehen lassen - bei Tastaturbedienung
   (:focus-visible) bleibt er erhalten. */
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"]:focus:not(:focus-visible) {
  outline: none !important;
  box-shadow: none;
}
[class*="st-key-rs-nav"] [data-testid="stBaseButton-tertiary"]:focus-visible {
  outline: 2px solid #6E6E6E !important;
  outline-offset: -2px;
}
[class*="st-key-rs-nav-"] [data-testid="stBaseButton-tertiary"]:hover {
  background: #1E1E1E;
  color: var(--rs-fg-bright);
}
/* Aktiv: nur die graue Flaeche, genau wie in der Vorlage. Kein Lime hier -
   ein farbiger Streifen liest sich am gerundeten Rand wie ein Rahmen und
   nimmt dem aktiven Reiter die Aufmerksamkeit. */
[class*="st-key-rs-navon-"] [data-testid="stBaseButton-tertiary"] {
  background: var(--rs-raised);
  color: var(--rs-fg-bright);
  font-weight: 600;
}
[class*="st-key-rs-nav"] [data-testid="stElementContainer"] { margin-bottom: 0; }

/* --- Seitentitel und Reiter --------------------------------------------- */
/* Der Seitentitel ist bewusst zurueckhaltend, mit dem Streamlit-Default aber
   zu klein: gemessen 22 statt der angestrebten 28 Geraetepixel Versalhoehe.
   1.5rem bei 16 px Basis trifft die Zielgroesse. */
.rs-title {
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -.012em;
  line-height: 1.25;
  color: var(--rs-fg);
  margin: 0 0 .15rem;
}
.rs-subtitle {
  font-size: .8125rem;
  line-height: 1.5;
  color: var(--rs-fg-muted);
  margin: 0;
  max-width: 68ch;
}

/* Reiter. Achtung: in Streamlit 1.62 sind das <div role="tab"
   data-testid="stTab">, keine <button> - ein button-Selektor greift nie.
   Der Unterstrich des aktiven Reiters ist ein Kind-<div> mit der
   Primaerfarbe als Hintergrund, bleibt also von selbst lime. */
[data-testid="stTabs"] [role="tablist"] {
  gap: 1.15rem;
  border-bottom: none !important;
  background: transparent;
}
[data-testid="stTabs"] [role="tablist"]::after,
[data-testid="stTabs"] [role="tablist"]::before { display: none !important; }
[data-testid="stTab"] {
  padding: .3rem 0 .45rem;
  font-size: .8125rem;
  background: transparent;
  color: var(--rs-fg-muted);
}
[data-testid="stTab"] [data-testid="stMarkdownContainer"] p {
  font-size: .8125rem;
  font-weight: 500;
  color: inherit;
}
[data-testid="stTab"]:hover { color: var(--rs-fg); }
/* Aktiv: weisse Beschriftung wie in der Vorlage - Streamlit faerbt sie sonst
   in der Primaerfarbe, was auf Lime unruhig wirkt. */
[data-testid="stTab"][aria-selected="true"] { color: var(--rs-fg-bright); }
[data-testid="stTab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p {
  color: var(--rs-fg-bright);
  font-weight: 600;
}
[data-testid="stTabPanel"] { padding-top: 1.1rem; }

/* --- Beschriftungen und Fliesstext -------------------------------------- */
[data-testid="stWidgetLabel"] p {
  font-size: .8125rem;
  font-weight: 600;
  color: var(--rs-fg);
}
/* Das Fragezeichen der Hilfe sitzt in einem Flex-Kind mit justify-content:
   flex-end. Ueber die volle Breite eines Widgets driftet es sonst weit vom
   Beschriftungstext ab; fit-content zieht es direkt daneben. */
[data-testid="stWidgetLabel"] { width: fit-content; }
[data-testid="stCaptionContainer"] p {
  font-size: .75rem;
  line-height: 1.5;
  color: var(--rs-fg-muted);
}
code, kbd, pre { font-family: var(--rs-mono); }
[data-testid="stMarkdown"] code {
  background: var(--rs-sunken);
  border: 1px solid var(--rs-border);
  border-radius: .3rem;
  padding: .06rem .3rem;
  font-size: .8em;
}
hr { border-color: var(--rs-border); opacity: 1; }

/* --- Buttons ------------------------------------------------------------ */
/* Primaer: an der Stelle, an der die Vorlage einen weissen Button setzt.
   Dunkler Text auf Lime - weiss auf Lime kaeme nur auf 1,9:1. */
[data-testid="stBaseButton-primary"] {
  background: var(--rs-accent);
  border: 1px solid var(--rs-accent);
  color: var(--rs-accent-ink);
  font-size: .8125rem;
  font-weight: 600;
  padding: .3rem .85rem;
  min-height: 2rem;
  transition: background .15s ease;
}
[data-testid="stBaseButton-primary"] * { color: inherit; }
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
  background: var(--rs-accent-bright);
  border-color: var(--rs-accent-bright);
}

[data-testid="stBaseButton-secondary"] {
  background: #2C2C2C;
  border: 1px solid var(--rs-border);
  color: var(--rs-fg);
  font-size: .8125rem;
  font-weight: 600;
  padding: .3rem .85rem;
  min-height: 2rem;
  transition: background .15s ease, border-color .15s ease;
}
[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
  background: var(--rs-raised);
  border-color: #4A4A4A;
  color: var(--rs-fg-bright);
}

/* --- Eingaben ----------------------------------------------------------- */
[data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"],
[data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--rs-surface);
  border-color: var(--rs-border-input);
  border-radius: var(--rs-radius);
}
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div {
  border-color: var(--rs-accent);
  box-shadow: 0 0 0 2px var(--rs-accent-soft);
}
[data-testid="stMultiSelectTagsContainer"] span[data-baseweb="tag"] {
  background: var(--rs-raised) !important;
  color: var(--rs-fg) !important;
  border: 1px solid var(--rs-border);
  border-radius: var(--rs-radius);
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--rs-surface);
  border: 1px dashed var(--rs-border-input);
  border-radius: var(--rs-radius-lg);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #6E6E6E; }

/* --- Karten: gleiche Fuellung wie die Seite, nur Rahmen ----------------- */
[class*="st-key-rs-card"] {
  background: transparent;
  border: 1px solid var(--rs-border) !important;
  border-radius: var(--rs-radius-lg) !important;
  padding: .85rem 1rem !important;
}
[class*="st-key-rs-card"]:hover { border-color: #4A4A4A !important; }

/* --- Expander ----------------------------------------------------------- */
[data-testid="stExpander"] details {
  background: transparent;
  border: 1px solid var(--rs-border);
  border-radius: var(--rs-radius-lg);
}
[data-testid="stExpander"] summary { font-size: .8125rem; }
[data-testid="stExpander"] summary:hover { color: var(--rs-fg-bright); }

/* --- Metriken ----------------------------------------------------------- */
[data-testid="stMetricValue"] {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--rs-fg);
}
[data-testid="stMetricLabel"] p {
  font-size: .75rem;
  font-weight: 400;
  color: var(--rs-fg-muted);
}

/* --- Chat --------------------------------------------------------------- */
[data-testid="stChatMessage"] { background: transparent; padding: .3rem 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #1B1B1B;
  border: 1px solid var(--rs-border);
  border-radius: var(--rs-radius-lg);
  padding: .7rem .9rem;
}
[data-testid="stChatMessageAvatarAssistant"] {
  background: var(--rs-accent-soft) !important;
  border: 1px solid var(--rs-accent-line);
  color: var(--rs-accent-bright) !important;
}
[data-testid="stChatInput"] {
  background: var(--rs-surface);
  border: 1px solid var(--rs-border-input);
  border-radius: var(--rs-radius-lg);
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--rs-accent);
  box-shadow: 0 0 0 2px var(--rs-accent-soft);
}

/* --- Eigene Bauteile ---------------------------------------------------- */
/* Markenzeile am Kopf der Sidebar. */
.rs-brand {
  display: flex;
  align-items: center;
  gap: .5rem;
  padding: .15rem .35rem .7rem;
}
.rs-brand-mark {
  flex: 0 0 auto;
  width: 1.4rem;
  height: 1.4rem;
  display: grid;
  place-items: center;
  border-radius: .3rem;
  background: var(--rs-accent);
  color: var(--rs-accent-ink);
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: -.02em;
}
.rs-brand-name {
  font-size: .8125rem;
  font-weight: 600;
  color: var(--rs-fg);
  letter-spacing: -.005em;
}

/* Fusszeile der Sidebar - Gegenstueck zur Organisationszeile der Vorlage. */
.rs-navfoot {
  margin-top: .9rem;
  padding: .6rem .6rem 0;
  border-top: 1px solid #2A2A2A;
  font-size: .7rem;
  line-height: 1.5;
  color: var(--rs-fg-muted);
}

/* Gruppenueberschrift innerhalb einer Seite. */
.rs-group {
  font-size: .875rem;
  font-weight: 600;
  color: var(--rs-fg);
  margin: 1.5rem 0 .1rem;
}
.rs-group:first-child { margin-top: 0; }
.rs-grouphint {
  font-size: .75rem;
  line-height: 1.5;
  color: var(--rs-fg-muted);
  margin: 0 0 .55rem;
  max-width: 72ch;
}

/* Metadaten an den Fundstellen. */
.rs-badge {
  display: inline-flex;
  align-items: center;
  padding: .1rem .45rem;
  margin: 0 .3rem .3rem 0;
  border-radius: .3rem;
  font-size: .7rem;
  font-weight: 500;
  background: var(--rs-raised);
  color: var(--rs-fg);
  border: 1px solid var(--rs-border);
  white-space: nowrap;
}
.rs-badge.grey {
  background: transparent;
  color: var(--rs-fg-muted);
}

/* Woertliches Zitat aus der Regelung. */
.rs-quote {
  border-left: 2px solid var(--rs-accent);
  border-radius: 0 var(--rs-radius) var(--rs-radius) 0;
  background: #1B1B1B;
  padding: .6rem .85rem;
  margin: .4rem 0 .6rem;
  font-size: .8125rem;
  line-height: 1.65;
  white-space: pre-wrap;
  color: #C8C8C8;
}

/* Statuszeile: farbiger Punkt plus Text, statt einer bunten Meldungsflaeche. */
.rs-status {
  display: flex;
  align-items: center;
  gap: .45rem;
  font-size: .8125rem;
  color: var(--rs-fg);
  margin: .1rem 0 .45rem;
}
.rs-dot {
  width: .5rem;
  height: .5rem;
  border-radius: 50%;
  flex: 0 0 auto;
}
.rs-dot.ok   { background: var(--rs-accent); }
.rs-dot.warn { background: #E3B341; }
.rs-dot.bad  { background: #F2726F; }

/* --- Bildlaufleisten ---------------------------------------------------- */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
  background: #3A3A3A;
  border: 2px solid var(--rs-surface);
  border-radius: 999px;
}
*::-webkit-scrollbar-thumb:hover { background: #4A4A4A; }

/* --- Bewegung ----------------------------------------------------------- */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


def inject() -> None:
    """
    Bindet das Stylesheet ein. Einmal pro Skriptlauf aufrufen.

    Bewusst ueber ``st.markdown``: ``st.html`` reicht seinen Inhalt durch
    DOMPurify, und dessen HTML-Profil kennt ``<style>`` nicht - das Stylesheet
    wuerde kommentarlos entfernt. Fuer reines Markup (unten) ist ``st.html``
    dagegen der passende Weg.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar-Bausteine
# --------------------------------------------------------------------------- #
def brand(name: str) -> None:
    """Markenzeile am Kopf der Navigation."""
    initial = name[:1].upper()
    st.html(
        f"""
        <div class="rs-brand">
          <div class="rs-brand-mark">{initial}</div>
          <div class="rs-brand-name">{name}</div>
        </div>
        """
    )


def nav(items: list[tuple[str, str]], current: str) -> str:
    """
    Zeichnet die Navigation und gibt den gewaehlten Schluessel zurueck.

    ``items`` ist eine Liste aus ``(schluessel, beschriftung)``. Der aktive
    Eintrag bekommt den key-Praefix ``rs-navon-``, alle anderen ``rs-nav-``;
    daran haengt in ``_CSS`` die Unterscheidung aktiv/inaktiv.
    """
    selected = current
    for key, label in items:
        prefix = "rs-navon" if key == current else "rs-nav"
        if st.button(label, key=f"{prefix}-{key}", type="tertiary", width="stretch"):
            selected = key
    return selected


def nav_footer(text: str) -> None:
    """Kleingedrucktes am Fuss der Navigation (Version, Betriebsart)."""
    st.html(f'<div class="rs-navfoot">{text}</div>')


# --------------------------------------------------------------------------- #
# Inhalts-Bausteine
# --------------------------------------------------------------------------- #
def page_title(title: str, subtitle: str = "") -> None:
    """Seitentitel. Direkt darunter folgen die Reiter."""
    extra = f'<p class="rs-subtitle">{subtitle}</p>' if subtitle else ""
    st.html(f'<div class="rs-pagehead"><div class="rs-title">{title}</div>{extra}</div>')


def group(label: str, hint: str = "") -> None:
    """Gruppenueberschrift innerhalb einer Seite, optional mit Erklaerzeile."""
    extra = f'<p class="rs-grouphint">{hint}</p>' if hint else ""
    st.html(f'<div class="rs-group">{label}</div>{extra}')


def status(text: str, tone: str = "ok") -> None:
    """
    Statuszeile mit farbigem Punkt.

    Ersetzt ``st.success``/``st.error`` dort, wo eine ganze farbige Flaeche
    zu laut waere - die Vorlage arbeitet durchgehend mit solchen Punkten.
    ``tone`` ist ``ok``, ``warn`` oder ``bad``.
    """
    st.html(f'<div class="rs-status"><span class="rs-dot {tone}"></span><span>{text}</span></div>')


def dim_options(label: str, indices: list[int]) -> None:
    """
    Graut einzelne Eintraege eines ``st.selectbox`` aus.

    Streamlit kann Optionen nicht einzeln abschalten. Das aufgeklappte Menue
    haengt ausserdem als Portal am ``<body>`` und damit ausserhalb des Widgets.
    Die Zuordnung laeuft deshalb ueber zwei Anker, die Streamlit selbst setzt:
    ``aria-label`` der Liste traegt die Beschriftung des Feldes, ``data-key``
    den Index der Option. ``label`` muss also der Beschriftung entsprechen,
    ``indices`` sind die Positionen in ``options``.

    Ausgegraut heisst hier wirklich nur blass - anklickbar bleiben die
    Eintraege, sonst liesse sich das aktive Modell nicht mehr umstellen.
    """
    if not indices:
        return
    rules = ", ".join(
        '[data-testid="stSelectboxVirtualDropdown"] '
        f'[role="listbox"][aria-label="{label}"] [role="option"][data-key="{index}"]'
        for index in indices
    )
    st.markdown(
        f"<style>{rules} {{ opacity: .45; }}</style>",
        unsafe_allow_html=True,
    )


def badge(text: str, grey: bool = False) -> str:
    """Metadaten-Chip als HTML-Schnipsel (wird in Markdown eingebettet)."""
    css = "rs-badge grey" if grey else "rs-badge"
    return f"<span class='{css}'>{text}</span>"


def quote(text: str) -> str:
    """Woertliches Zitat mit Lime-Kante. Maskiert spitze Klammern aus dem Regeltext."""
    return f"<div class='rs-quote'>{text.replace('<', '&lt;')}</div>"

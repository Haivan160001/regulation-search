"""
tools/fetch_fonts.py - laedt die Schriften der Marken-Oberflaeche herunter.

Reg-Search verwendet DM Sans fuer die gesamte Oberflaeche und Geist Mono
fuer Code, Pfade und Modellnamen. Beide stehen unter der SIL Open Font
License und duerfen mitgeliefert werden.

Warum lokal statt ``@import`` von Google Fonts?
----------------------------------------------
Reg-Search verarbeitet vertrauliche Regelungsentwuerfe und laeuft bewusst
offline. Ein Webfont-Import wuerde bei jedem Seitenaufruf eine Anfrage an
fonts.googleapis.com ausloesen (inkl. IP-Adresse des Nutzers) und ohne
Internetverbindung stumm auf Systemschriften zurueckfallen. Die Dateien werden
deshalb einmalig nach ``static/fonts/`` geladen und von Streamlit selbst
ausgeliefert (``server.enableStaticServing = true``).

Aufruf
------
    python tools/fetch_fonts.py            # fehlende Dateien laden
    python tools/fetch_fonts.py --force    # vorhandene ueberschreiben

Ohne die Dateien startet die App normal; die Oberflaeche faellt dann auf die
in ``.streamlit/config.toml`` hinterlegten Systemschriften zurueck.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

# Variable Fonts: Google liefert pro Subset *eine* Datei fuer den gesamten
# Gewichtsbereich - daher genuegen zwei Dateien je Familie.
FAMILIES: dict[str, str] = {
    "DM Sans": "wght@100..1000",
    "Geist Mono": "wght@100..900",
}

# latin deckt Deutsch/Englisch ab, latin-ext die restlichen europaeischen
# Sonderzeichen (z. B. in Namen von UNECE-Vertragsparteien).
SUBSETS = ("latin", "latin-ext")

CSS_URL = "https://fonts.googleapis.com/css2?{families}&display=swap"

# Ohne modernen User-Agent liefert Google TTF statt WOFF2 (rund 4x groesser).
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

TARGET_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"

# "/* latin */" gefolgt vom zugehoerigen @font-face-Block.
_BLOCK_RE = re.compile(r"/\*\s*([\w-]+)\s*\*/\s*@font-face\s*\{(.*?)\}", re.DOTALL)
_URL_RE = re.compile(r"url\((https://[^)]+\.woff2)\)")
_UNICODE_RE = re.compile(r"unicode-range:\s*([^;]+);")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def slug(family: str) -> str:
    return family.lower().replace(" ", "-")


def parse_css(css: str) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Ordnet ``(Familie, Subset)`` auf ``(woff2-URL, unicode-range)``.

    Der zurueckgegebene unicode-range wandert unveraendert in die
    ``[[theme.fontFaces]]``-Eintraege, damit der Browser latin-ext nur bei
    Bedarf nachlaedt.
    """
    found: dict[tuple[str, str], tuple[str, str]] = {}
    family = ""
    for subset, body in _BLOCK_RE.findall(css):
        match = re.search(r"font-family:\s*'([^']+)'", body)
        if match:
            family = match.group(1)
        if subset not in SUBSETS or not family:
            continue
        url_match = _URL_RE.search(body)
        range_match = _UNICODE_RE.search(body)
        if url_match and range_match:
            found[(family, subset)] = (url_match.group(1), range_match.group(1).strip())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="vorhandene Dateien ersetzen")
    args = parser.parse_args()

    families = "&".join(
        f"family={family.replace(' ', '+')}:{axis}" for family, axis in FAMILIES.items()
    )
    url = CSS_URL.format(families=families)

    print(f"Lade Font-Metadaten von {url.split('?')[0]} ...")
    try:
        css = _fetch(url).decode("utf-8")
    except Exception as exc:  # Netzwerk/Proxy - Abbruch mit klarer Meldung
        print(f"FEHLER: Stylesheet nicht erreichbar: {exc}", file=sys.stderr)
        return 1

    fonts = parse_css(css)
    missing = [
        (family, subset)
        for family in FAMILIES
        for subset in SUBSETS
        if (family, subset) not in fonts
    ]
    if missing:
        names = ", ".join(f"{f}/{s}" for f, s in missing)
        print(f"WARNUNG: kein Treffer fuer {names}", file=sys.stderr)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    ranges: list[str] = []

    for (family, subset), (font_url, unicode_range) in sorted(fonts.items()):
        target = TARGET_DIR / f"{slug(family)}-{subset}.woff2"
        if target.exists() and not args.force:
            print(f"  uebersprungen (vorhanden): {target.name}")
        else:
            try:
                target.write_bytes(_fetch(font_url))
            except Exception as exc:
                print(f"FEHLER bei {target.name}: {exc}", file=sys.stderr)
                return 1
            print(f"  geladen: {target.name} ({target.stat().st_size / 1024:.0f} KB)")
        ranges.append(f"# {target.name}\n#   unicodeRange = \"{unicode_range}\"")

    print(f"\nFertig - {len(fonts)} Dateien in {TARGET_DIR}")
    print("Die unicode-ranges stehen bereits in .streamlit/config.toml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

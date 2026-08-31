"""
tools/check_i18n.py - prueft die Uebersetzungstabelle in i18n.py.

Meldet drei Dinge:

1. **Fehlende Sprachen** - ein Eintrag ohne "de" oder "en". ``t()`` faellt dann
   stumm auf Deutsch zurueck; im englischen UI stuende unerwartet Deutsch.
2. **Unbenutzte Schluessel** - stehen in ``STRINGS``, werden aber nirgends per
   ``t("...")`` aufgerufen.
3. **Unbekannte Schluessel** - werden aufgerufen, fehlen aber in ``STRINGS``.
   ``t()`` gibt dann den Schluessel selbst aus, was in der Oberflaeche auffaellt.

Zusaetzlich wird geprueft, ob beide Sprachfassungen dieselben ``{platzhalter}``
verwenden - sonst bleibt beim Sprachwechsel eine Luecke im Text.

Aufruf
------
    python tools/check_i18n.py

Rueckgabewert 1, wenn etwas fehlt - damit taugt das Skript fuer CI.
"""

from __future__ import annotations

import re
import string
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import i18n  # noqa: E402  (Pfad muss vorher stehen)

#: Dateien, die t("...") aufrufen duerfen.
SOURCES = ["app.py", "ui.py"]

_CALL_RE = re.compile(r"""\bt\(\s*["']([\w.]+)["']""")
#: Schluessel, die dynamisch zusammengesetzt werden - z. B. t(f"nav.{key}").
DYNAMIC_PREFIXES = ("nav.", "chat.example_")


def placeholders(text: str) -> set[str]:
    """Namen aller ``{platzhalter}`` in einem Text."""
    return {
        name
        for _, name, _, _ in string.Formatter().parse(text)
        if name
    }


def main() -> int:
    problems = 0
    languages = set(i18n.LANGUAGES)

    # -- 1. Vollstaendigkeit und passende Platzhalter -----------------------
    for key, entry in sorted(i18n.STRINGS.items()):
        missing = languages - set(entry)
        if missing:
            print(f"FEHLT   {key}: keine Fassung fuer {', '.join(sorted(missing))}")
            problems += 1
            continue
        sets = {lang: placeholders(entry[lang]) for lang in languages}
        first = next(iter(sets.values()))
        if any(s != first for s in sets.values()):
            detail = "  ".join(f"{lang}={sorted(s) or '-'}" for lang, s in sorted(sets.items()))
            print(f"PLATZH. {key}: unterschiedliche Platzhalter -> {detail}")
            problems += 1

    # -- 2./3. Abgleich mit den Aufrufstellen -------------------------------
    used: set[str] = set()
    for name in SOURCES:
        path = BASE_DIR / name
        if not path.exists():
            print(f"HINWEIS {name} nicht gefunden - uebersprungen")
            continue
        used |= set(_CALL_RE.findall(path.read_text(encoding="utf-8")))

    unknown = sorted(used - set(i18n.STRINGS))
    for key in unknown:
        print(f"UNBEK.  {key}: wird aufgerufen, fehlt aber in STRINGS")
        problems += 1

    unused = sorted(
        key
        for key in set(i18n.STRINGS) - used
        if not key.startswith(DYNAMIC_PREFIXES)
    )
    for key in unused:
        print(f"UNGENUTZT {key}")

    print(
        f"\n{len(i18n.STRINGS)} Schluessel, {len(languages)} Sprachen, "
        f"{len(used)} feste Aufrufe, {len(unused)} ungenutzt, {problems} Problem(e)."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

# Schriften

Die Oberflaeche verwendet **eine** Schrift; Geist Mono ist Code, Pfaden und
Modellnamen vorbehalten, weil ein Dateipfad in einer Proportionalschrift
schlecht lesbar ist.

| Datei | Familie | Verwendung |
|---|---|---|
| `dm-sans-*.woff2` | DM Sans | gesamte Oberflaeche |
| `geist-mono-*.woff2` | Geist Mono | Code, Pfade, Modellnamen |

Beide stehen unter der **SIL Open Font License 1.1** und duerfen mitgeliefert
werden. Die Lizenz verlangt, dass ihr Text jeder Weitergabe der Schriftdateien
beiliegt - er steht deshalb vollstaendig in [`OFL.txt`](OFL.txt) neben den
Dateien. Bezugsquelle ist Google Fonts:

* <https://fonts.google.com/specimen/DM+Sans/license>
* <https://fonts.google.com/specimen/Geist+Mono/license>

Es sind Variable Fonts - eine Datei deckt den gesamten Gewichtsbereich ab.
Pro Familie gibt es zwei Subsets: `latin` und `latin-ext`; letzteres laedt der
Browser nur, wenn ein Zeichen daraus vorkommt.

## Warum lokal statt Google Fonts?

Reg-Search verarbeitet vertrauliche Regelungsentwuerfe und laeuft bewusst
offline. Ein `@import` von `fonts.googleapis.com` wuerde bei jedem Seitenaufruf
eine Anfrage an Google ausloesen und ohne Internetverbindung stumm auf
Systemschriften zurueckfallen. Streamlit liefert die Dateien deshalb selbst
aus (`server.enableStaticServing = true` in `.streamlit/config.toml`).

## Neu laden

    python tools/fetch_fonts.py           # fehlende Dateien holen
    python tools/fetch_fonts.py --force   # vorhandene ersetzen

Fehlen die Dateien, startet die App normal - die Stacks in
`.streamlit/config.toml` und `ui.py` fallen dann auf Segoe UI bzw.
Consolas zurueck.

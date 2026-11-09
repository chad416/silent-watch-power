"""Export DOSSIER.md (and docs/RESULTS.md as an annex) to docs/DOSSIER.pdf.

Markdown -> HTML with the `markdown` package, then a headless Chromium-family browser prints it.
Usage: python tools/export_pdf.py [--browser PATH]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.5pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; border-bottom: 2px solid #2a6fb0; padding-bottom: 4px; }
h2 { font-size: 13pt; color: #2a6fb0; margin-top: 18px; page-break-after: avoid; }
h3 { font-size: 11pt; margin-top: 12px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8pt; page-break-inside: auto; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; }
th { background: #eef3f8; }
tr { page-break-inside: avoid; }
code, pre { font-family: Consolas, monospace; font-size: 8pt; }
pre { background: #f5f5f5; padding: 6px; white-space: pre; overflow: hidden; }
img { max-width: 100%; page-break-inside: avoid; }
blockquote { border-left: 3px solid #2a6fb0; margin: 6px 0; padding: 2px 10px; color: #333; }
hr { border: 0; border-top: 1px solid #ccc; }
"""
CANDIDATES = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              "chromium", "chromium-browser", "google-chrome"]


LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.) ")


def gfm_lists(md: str) -> str:
    """python-markdown needs a blank line before a list and 4-space nesting; GitHub needs neither."""
    out, prev, in_code = [], "", False
    for line in md.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code:
            m = LIST_RE.match(line)
            if m:
                ind = len(m.group(1))
                if 0 < ind < 4:
                    line = " " * 4 + line[ind:]
                if prev.strip() and not LIST_RE.match(prev) and not prev.startswith("    "):
                    out.append("")
            elif line.startswith("   ") and not line.startswith("    ") and LIST_RE.match(prev or ""):
                line = " " + line
        out.append(line)
        prev = line
    return "\n".join(out)


def build_html() -> Path:
    md = gfm_lists((ROOT / "DOSSIER.md").read_text(encoding="utf-8"))
    annex = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body += '<div style="page-break-before: always"></div>'
    body += markdown.markdown(annex.replace("# Results (generated)", "# Annex: results tables (generated)"),
                              extensions=["tables", "fenced_code"])
    out = ROOT / "build" / "DOSSIER.html"
    out.parent.mkdir(exist_ok=True)
    html = (f"<!doctype html><html><head><meta charset='utf-8'><base href='{ROOT.as_uri()}/'>"
            f"<style>{CSS}</style><title>Engineering dossier</title></head><body>{body}</body></html>")
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    a = ap.parse_args()
    html = build_html()
    browser = a.browser or next((c for c in CANDIDATES if Path(c).exists() or shutil.which(c)), None)
    if browser is None:
        print("no headless browser found; HTML written to", html)
        return 1
    pdf = ROOT / "docs" / "DOSSIER.pdf"
    pdf.unlink(missing_ok=True)
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, timeout=180)
    # some browsers hand the job to a running instance and return early: wait for a stable file
    last = -1
    for _ in range(120):
        size = pdf.stat().st_size if pdf.exists() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(1)
    print("wrote", pdf.relative_to(ROOT), f"({last} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Export the ST function block as Siemens SCL source for TIA Portal (copied from the author's weld-cell project).

The body is copied as it is, with every declared variable written as #Name, TON written as
TON_TIME and the block name quoted. The output is generated text: it was not compiled in TIA
Portal here.
"""
from __future__ import annotations

import re
from pathlib import Path

from .stlang import compile_st

COMMENT_RE = re.compile(r"\(\*.*?\*\)|//[^\n]*", re.DOTALL)
SECTIONS = {"input": "VAR_INPUT", "output": "VAR_OUTPUT", "inout": "VAR_IN_OUT", "local": "VAR",
            "constant": "VAR CONSTANT"}
TIA_TYPE = {"BOOL": "Bool", "INT": "Int", "DINT": "DInt", "REAL": "Real", "TIME": "Time", "TON": "TON_TIME",
            "R_TRIG": "R_TRIG"}


def _fmt_init(v) -> str:
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)


def _time_lit(ms: int) -> str:
    return f"T#{ms}MS" if ms % 1000 else f"T#{ms // 1000}S"


def to_scl(src: str, version: str = "0.1") -> str:
    c = compile_st(src)
    # body: between the last END_VAR and END_FUNCTION_BLOCK
    body = src[src.upper().rfind("END_VAR") + len("END_VAR"):src.upper().rfind("END_FUNCTION_BLOCK")]
    names = sorted((v.name for v in c.vars.values()), key=len, reverse=True)
    pat = re.compile(r"(?<![#.\w])(" + "|".join(map(re.escape, names)) + r")(?!\w)", re.IGNORECASE)
    canon = {v.name.lower(): v.name for v in c.vars.values()}

    def sub_code(seg: str) -> str:
        return pat.sub(lambda m: "#" + canon[m.group(1).lower()], seg)

    parts, last = [], 0
    for m in COMMENT_RE.finditer(body):
        parts.append(sub_code(body[last:m.start()]))
        parts.append(m.group())
        last = m.end()
    parts.append(sub_code(body[last:]))
    out = [f'FUNCTION_BLOCK "{c.name}"', "{ S7_Optimized_Access := 'TRUE' }", f"VERSION : {version}"]
    for section, kw in SECTIONS.items():
        vs = [v for v in c.vars.values() if v.section == section]
        if not vs:
            continue
        out.append(f"   {kw}")
        for v in vs:
            if v.type == "ARRAY":
                t = f"Array[{v.lo}..{v.hi}] of {TIA_TYPE[v.elem]}"
            else:
                t = TIA_TYPE[v.type]
            init = ""
            if v.init is not None:
                init = " := " + (_time_lit(v.init) if v.type == "TIME" else _fmt_init(v.init))
            out.append(f"      {v.name} : {t}{init};")
        out.append("   END_VAR")
    out.append("")
    out.append("BEGIN")
    out.append("".join(parts).strip("\n"))
    out.append("END_FUNCTION_BLOCK")
    return "\n".join(out) + "\n"


def export(st_path: Path, scl_path: Path) -> Path:
    scl_path.parent.mkdir(parents=True, exist_ok=True)
    scl_path.write_text(to_scl(Path(st_path).read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    return scl_path

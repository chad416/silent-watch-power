"""Derive the cold-temperature R0 points of each battery block from its specification cranking rating.

For each rating (T, I, V_end, 30 s) the R0 at T is the value for which a full block, soaked at T,
ends a 30 s discharge at I exactly at V_end. With that R0 the model block *just* meets the
specification at that point. `--check` compares with data/batteries.yaml (used in CI).
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swpower import data  # noqa: E402
from swpower.battery import Pack, crank_test  # noqa: E402

RATINGS = {
    # block: [(degC, A, V_end)] - 30 s each
    "agm_6t": [(-18.0, 1100.0, 7.2), (-40.0, 400.0, 7.2)],        # MIL-PRF-32143C Table I
    "li_6t": [(-18.0, 1100.0, 14.4), (-32.0, 600.0, 14.4), (-40.0, 400.0, 14.4)],  # MIL-PRF-32565 Table I, Type 3
}


def solve_point(blk, t_c: float, i: float, v_end: float) -> float:
    lo, hi = 1e-4, 0.2
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        b = copy.deepcopy(blk)
        b.r0_table = [p for p in b.r0_table if abs(p[0] - t_c) > 1e-9] + [(t_c, mid)]
        b.r0_table.sort()
        ok, _ = crank_test(Pack(b, 1, 1, soc=1.0, t_c=t_c), i, 30.0, 1, 0.0, t_c, v_end)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


def derive() -> dict[str, list[tuple[float, float]]]:
    out = {}
    blocks = data.blocks()
    for name, pts in RATINGS.items():
        blk = blocks[name]
        table = dict(blk.r0_table)
        for t_c, i, v_end in pts:
            work = copy.deepcopy(blk)
            work.r0_table = sorted(table.items())
            table[t_c] = round(solve_point(work, t_c, i, v_end), 6)
        out[name] = sorted(table.items())
    return out


def write_tables(d: dict) -> None:
    path = ROOT / "data" / "batteries.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    out, block, i = [], None, 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("  ") and not ln.startswith("    ") and ln.strip().endswith(":"):
            block = ln.strip()[:-1]
        out.append(ln)
        if ln.strip().startswith("r0_table:") and block in d:
            i += 1
            while i < len(lines) and lines[i].strip().startswith("- ["):
                i += 1
            out += [f"      - [{t:.1f}, {r:.6f}]" for t, r in d[block]]
            continue
        i += 1
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true", help="rewrite the r0_table blocks in data/batteries.yaml")
    a = ap.parse_args()
    d = derive()
    if a.write:
        for _ in range(4):          # self-heating couples neighbouring points: iterate to a fixed point
            write_tables(d)
            d = derive()
        write_tables(d)
    blocks = data.blocks()
    bad = False
    for name, table in d.items():
        cur = dict(blocks[name].r0_table)
        for t_c, r in table:
            flag = abs(cur.get(t_c, -1) - r) > 2e-6
            bad |= flag
            print(f"{name:8s} {t_c:6.1f} degC  R0 = {r:.6f} ohm   (yaml {cur.get(t_c, float('nan')):.6f}){'  <-- update' if flag else ''}")
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())

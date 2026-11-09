"""Command line: python -m swpower <command>."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import data, studies
from .analysis import endurance, validate_against_product, validate_battery_models
from .scl import export

ROOT = Path(__file__).resolve().parents[1]


def _h(x: float) -> str:
    return "  >run " if math.isinf(x) else f"{x:6.2f}"


def cmd_budget(_a) -> None:
    r = studies.budget_study()
    print("Power budget (W): CBE / with growth / allocated incl. system margin / coincident peak")
    for m, v in r["modes"].items():
        print(f"  {m:13s} {v['cbe_w']:7.0f} {v['with_growth_w']:7.0f} {v['allocated_w']:7.0f} {v['coincident_peak_w']:7.0f}")
    print("Silent watch by priority (W, with growth): " + ", ".join(f"P{k} {v:.0f}" for k, v in r["by_priority_sw_w"].items()))
    print(f"Mission-compartment heat from equipment (silent watch): {r['equipment_heat_w']:.0f} W")
    print("Thermal budget per climate:   T_amb  uncooled  cooling   ECU draw")
    for k, v in r["thermal"].items():
        print(f"  {k:15s} {v['t_amb']:6.0f} {v['t_uncooled']:8.0f} {v['q_cool_w']:8.0f} {v['ecu_w']:9.0f}")


def cmd_validate(_a) -> None:
    print("Battery models against their specification ratings:")
    for r in validate_battery_models():
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['test']:50s} model {r['model']:>9s}  required {r['required']}")
    print("Spec-minimum 6T AGM model against a qualified product's run-time table (25 degC, to 10.5 V):")
    for r in validate_against_product():
        print(f"  {r['current_a']:6.1f} A  product {r['product_min']:5.0f} min  model {r['model_min']:6.1f} min  ratio {r['ratio']:.2f}")


def cmd_watch(a) -> None:
    r = endurance(a.arch, a.climate, a.np, a.apu, hours=a.hours, dt=a.dt, silent_req=not a.allow_apu)
    print(f"{a.arch}, {a.climate}, mission strings/blocks np={r['np']}, mass {r['mass_kg']:.0f} kg")
    print(f"  full suite {_h(r['t_full_h'])} h | P1-P3 {_h(r['t_l2_h'])} h | essential P1-P2 {_h(r['t_essential_h'])} h"
          f" | vital P1 {_h(r['t_vital_h'])} h")
    print(f"  engine start after the watch: {'OK' if r['crank_ok'] else 'FAIL'} (lowest {r['crank_vmin']:.1f} V)"
          f" | fuel {r['fuel_l']:.1f} L | APU {r['apu_h']:.1f} h")
    for e in r["result"].events:
        print("  " + e)


def cmd_scl(a) -> None:
    p = export(ROOT / "plc" / "FB_PowerMgmt.st", Path(a.out))
    print(f"wrote {p}")


def cmd_all(a) -> None:
    res = studies.run_all(quick=a.quick)
    print("wrote results/*.json")
    print(json.dumps(res["aoa"]["score"]["fom"], indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="swpower")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("budget").set_defaults(fn=cmd_budget)
    sub.add_parser("validate").set_defaults(fn=cmd_validate)
    w = sub.add_parser("watch")
    w.add_argument("--arch", default="A5_split_li_apu", choices=list(data.architectures()))
    w.add_argument("--climate", default="A2_basic_hot", choices=list(data.climates()))
    w.add_argument("--np", type=int, default=None)
    w.add_argument("--apu", default=None)
    w.add_argument("--allow-apu", action="store_true")
    w.add_argument("--hours", type=float, default=16.0)
    w.add_argument("--dt", type=float, default=10.0)
    w.set_defaults(fn=cmd_watch)
    s = sub.add_parser("scl")
    s.add_argument("--out", default="build/FB_PowerMgmt.scl")
    s.set_defaults(fn=cmd_scl)
    r = sub.add_parser("all")
    r.add_argument("--quick", action="store_true")
    r.set_defaults(fn=cmd_all)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

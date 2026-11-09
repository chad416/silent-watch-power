"""Load the YAML data set (data/*.yaml) into model objects, with checks.

Every number in data/ carries a `basis` (sourced / derived / assumed) and, unless assumed, a
source key that resolves in data/references.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .battery import BlockParams, Pack, block_from_dict
from .loads import Budget, load_from_dict, validate
from .sim import Architecture, Crank, Phase
from .sources import alternator_from_dict, apu_from_dict
from .thermal import zone_from_dict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _y(name: str) -> dict:
    return yaml.safe_load((DATA / f"{name}.yaml").read_text(encoding="utf-8"))


def references() -> dict:
    return _y("references")


def check_sources() -> list[str]:
    refs = references()
    bad = []
    for name in ("loads", "batteries", "sources", "environment", "mission"):
        doc = _y(name)

        def walk(x, path, name=name):
            if isinstance(x, dict):
                for k in x.get("src", []) or []:
                    if k not in refs:
                        bad.append(f"{name}:{path}: unknown source {k}")
                for k, v in x.items():
                    walk(v, f"{path}.{k}", name)
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{path}[{i}]", name)
        walk(doc, "")
    return bad


def budget() -> Budget:
    d = _y("loads")
    loads = [load_from_dict(x) for x in d["loads"]]
    errs = validate(loads)
    if errs:
        raise ValueError("loads.yaml: " + "; ".join(errs))
    return Budget(loads, d["growth_allowance"], d["system_margin"])


def blocks() -> dict[str, BlockParams]:
    return {k: block_from_dict(v) for k, v in _y("batteries")["blocks"].items()}


def alternator():
    return alternator_from_dict(_y("sources")["alternator"])


def apu(name: str = "apu5"):
    return apu_from_dict(_y("sources")["apus"][name])


def engine() -> dict:
    return _y("sources")["engine"]


def crank(case: str = "cold") -> Crank:
    c = _y("sources")["crank"][case]
    return Crank(c["i_a"], c["seconds"], c["attempts"], c["rest_s"], c["v_min"])


def zone():
    return zone_from_dict(_y("environment")["zone"])


def climates() -> dict:
    return _y("environment")["climates"]


def mission_profiles() -> dict[str, list[Phase]]:
    d = _y("mission")["profiles"]
    return {k: [Phase(**p) for p in v] for k, v in d.items()}


def architectures() -> dict[str, dict]:
    return _y("mission")["architectures"]


def build_arch(name: str, mission_np: int | None = None, with_apu: str | None = None) -> Architecture:
    """with_apu: None = as the architecture defines, "" = none, else an APU name from sources.yaml."""
    a = architectures()[name]
    b = blocks()
    m = a["mission"]
    mission = Pack(b[m["block"]], m["ns"], mission_np if mission_np is not None else m["np"])
    start = None
    if a.get("start"):
        s = a["start"]
        start = Pack(b[s["block"]], s["ns"], s["np"])
    eng = engine()
    use_apu = a.get("apu") if with_apu is None else with_apu
    floor = mission.p.dod_floor
    table = None
    if a["soc_reserve"] == "crank":
        table = reserve_table(m["block"], m["ns"], mission.np_, floor)
    return Architecture(name=name, mission=mission, start=start, alternator=alternator(),
                        apu=apu(use_apu) if use_apu else None, soc_reserve=floor,
                        keepalive_w=eng["keepalive_w"], engine_idle_lph=eng["idle_lph"], reserve_table=table)


RESERVE_T = (-46.0, -32.0, -18.0, 0.0, 25.0, 49.0)
_RESERVE_CACHE: dict = {}


def reserve_table(block: str, ns: int, np_: int, floor: float, margin: float = 0.05) -> list[tuple[float, float]]:
    """Controller floor vs temperature for a common bank: the crank-reserve SOC + margin, never below
    the DoD floor; 1.0 where the bank cannot start the engine at all (it must never be discharged)."""
    import math

    from .analysis import crank_reserve_soc
    key = (block, ns, np_, floor, margin)
    if key not in _RESERVE_CACHE:
        rows = []
        for t in RESERVE_T:
            r = crank_reserve_soc(block, ns, np_, t)
            rows.append((t, 1.0 if math.isnan(r) else min(1.0, max(floor, r + margin))))
        _RESERVE_CACHE[key] = rows
    return _RESERVE_CACHE[key]

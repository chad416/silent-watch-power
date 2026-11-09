"""The studies that answer the central question (DOSSIER 9-13). `run_all()` writes results/*.json,
which the figures, the tests and the DOSSIER tables use."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import yaml

from . import aoa, data, rma
from .analysis import crank_reserve_soc, endurance, equipment_heat, validate_against_product, validate_battery_models
from .sspc import FaultCase, SSPCRating, Wire, coordination, fault_coverage
from .transient import FrontEnd, simulate, waveform_from_dict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def cfg() -> dict:
    return yaml.safe_load((ROOT / "data" / "studies.yaml").read_text(encoding="utf-8"))


def _fin(x: float, cap: float = 24.0) -> float:
    return cap if math.isinf(x) else float(x)


# --- power budget and thermal budget ------------------------------------------------------------
def budget_study() -> dict:
    b = data.budget()
    z = data.zone()
    modes = {m: b.by_mode(m) for m in ("drive", "halt", "silent_watch")}
    by_prio = {}
    for p in (1, 2, 3, 4):
        by_prio[p] = sum(ld.p_avg("silent_watch") * (1 + b.growth[ld.maturity]) for ld in b.loads if ld.priority == p)
    rows = [{"id": ld.id, "name": ld.name, "priority": ld.priority, "channel": ld.channel, "basis": ld.basis,
             "src": ld.src, "sw_avg_w": ld.p_avg("silent_watch"), "sw_peak_w": ld.p_peak("silent_watch"),
             "drive_avg_w": ld.p_avg("drive")} for ld in b.loads]
    q = equipment_heat(b, "silent_watch")
    thermal = {}
    for name, c in data.climates().items():
        pe, qc = z.ecu_electric_w(c["t_amb"], c["g_solar"], q)
        thermal[name] = {"t_amb": c["t_amb"], "g_solar": c["g_solar"], "t_uncooled": z.t_uncooled(c["t_amb"], c["g_solar"], q),
                         "q_cool_w": qc, "ecu_w": pe,
                         "hours_to_limit_no_cooling": z.time_to_limit(z.t_limit_c - 5, c["t_amb"], c["g_solar"], q)}
    return {"modes": modes, "by_priority_sw_w": by_prio, "loads": rows, "equipment_heat_w": q, "thermal": thermal}


# --- engine-start reserve -------------------------------------------------------------------------
def reserve_study() -> dict:
    out = {}
    for label, blk, ns, np_ in [("AGM 2s1p", "agm_6t", 2, 1), ("AGM 2s2p", "agm_6t", 2, 2), ("AGM 2s3p", "agm_6t", 2, 3),
                                ("Li 1s1p", "li_6t", 1, 1), ("Li 1s2p", "li_6t", 1, 2)]:
        out[label] = {t: crank_reserve_soc(blk, ns, np_, t) for t in (25.0, 0.0, -18.0, -32.0, -46.0)}
    return out


# --- sizing ------------------------------------------------------------------------------------------
def sizing_study(dt: float = 20.0) -> dict:
    c = cfg()
    out = {}
    for arch, nps in c["sizing"].items():
        out[arch] = {}
        for cl in ("A2_basic_hot", "temperate", "C1_basic_cold", "A1_hot_dry", "C2_cold"):
            rows = []
            for n in nps:
                r = endurance(arch, cl, n, hours=20, dt=dt)
                rows.append({"np": n, "full_h": r["t_full_h"], "essential_h": r["t_essential_h"],
                             "vital_h": r["t_vital_h"], "crank_ok": bool(r["crank_ok"])})   # inf = still on at run end
            out[arch][cl] = rows
    return out


# --- alternatives -----------------------------------------------------------------------------------
def _storage(arch) -> tuple[float, float, float]:
    packs = [arch.mission] + ([arch.start] if arch.start is not None else [])
    mass = sum(p.mass_kg for p in packs)
    vol = sum(p.volume_l for p in packs)
    cost = sum(p.blocks * p.p.cost_eur for p in packs)
    if arch.apu is not None:
        mass += arch.apu.mass_kg
        vol += arch.apu.volume_l
        cost += arch.apu.cost_eur
    return mass, vol, cost


def rbd_blocks(arch, n_ess_channels: int, k_needed: int) -> list[rma.Block]:
    rb = cfg()["rbd"]["blocks"]

    def b(key, n=1, k=1, name=None):
        d = rb[key]
        return rma.Block(key, name or d["name"], d["fpmh"], n, k, d["mcmt_h"], "assumed", d.get("src", []))

    m = arch.mission
    blocks = []
    if m.p.chem == "li_ion":
        blocks.append(b("li_block", m.np_, min(k_needed, m.np_), "Li-ion mission bank (k of n blocks)"))
    else:
        # one failed block takes its whole series string out: model strings as items
        d = rb["agm_block"]
        blocks.append(rma.Block("agm_string", "AGM mission bank (k of n strings)", d["fpmh"] * m.ns, m.np_,
                                min(k_needed, m.np_), d["mcmt_h"]))
    if arch.start is not None:
        blocks.append(b("combiner"))
    blocks += [b("main_contactor"), b("monitor"), b("pmc"), b("busbar"),
               b("sspc_ch", n_ess_channels, n_ess_channels, f"SSPC channels of the essential suite ({n_ess_channels})"),
               b("ecu")]
    if arch.apu is not None:
        blocks.append(b("apu") if arch.apu.name.startswith("5 kW") else
                      rma.Block("engine_idle", "Alternator + engine as charging source", rb["alternator"]["fpmh"], 1, 1,
                                rb["alternator"]["mcmt_h"]))
    return blocks


def evaluate_alternative(key: str, spec: dict, dt: float = 20.0) -> dict:
    c = cfg()
    arch = data.build_arch(spec["arch"], spec["np"], spec["apu"])
    hot = endurance(spec["arch"], "A2_basic_hot", spec["np"], "", hours=24, dt=dt)       # silent: no APU
    cold = endurance(spec["arch"], "C1_basic_cold", spec["np"], "", hours=24, dt=dt)
    w12 = endurance(spec["arch"], "A2_basic_hot", spec["np"], spec["apu"], hours=12, dt=dt, silent_req=False)
    mass, vol, cost = _storage(arch)
    base = data.blocks()[c["baseline_storage"]["block"]]
    nb = c["baseline_storage"]["blocks"]
    sw_hot, sw_cold = _fin(hot["t_essential_h"]), _fin(cold["t_essential_h"])
    ess12 = min(12.0, _fin(w12["t_essential_h"]))
    apu_used = w12["apu_h"] > 0
    dba = arch.apu.noise_dba_7m if (arch.apu is not None and apu_used) else c["background_dba"]
    budget = data.budget()
    ess_ch = len({ld.channel for ld in budget.loads if ld.priority <= 2})
    k_needed = arch.mission.np_ if sw_hot <= c["kpp"]["threshold_h"] else \
        max(1, math.ceil(arch.mission.np_ * c["kpp"]["threshold_h"] / sw_hot))
    blocks = rbd_blocks(arch, ess_ch, k_needed)
    r12 = rma.series_reliability(blocks, 12.0)
    return {
        "key": key, "label": spec["label"], "arch": spec["arch"], "np": spec["np"], "apu": spec["apu"],
        "blocks_mission": arch.mission.blocks, "blocks_start": arch.start.blocks if arch.start else 0,
        "values": {"sw_hot": sw_hot, "sw_cold": sw_cold, "watch12_dba": dba, "watch12_fuel": w12["fuel_l"],
                   "added_mass": mass - nb * base.mass_kg, "added_volume": vol - nb * base.volume_l,
                   "added_cost": (cost - nb * base.cost_eur) / 1000.0, "r_watch": r12, "watch12_ess": ess12},
        "crank_after_cold_watch": bool(cold["crank_ok"]), "crank_vmin_cold": cold["crank_vmin"],
        "full_hot_h": _fin(hot["t_full_h"]), "vital_hot_h": _fin(hot["t_vital_h"]), "apu_hours_12h": w12["apu_h"],
        "k_needed": k_needed,
    }


def aoa_study(dt: float = 20.0) -> dict:
    c = cfg()
    crit = [aoa.criterion_from_dict(x) for x in c["criteria"]]
    evals = {k: evaluate_alternative(k, v, dt) for k, v in c["alternatives"].items()}
    table = {k: e["values"] for k, e in evals.items()}
    # the cold-start screen (KPP side condition): an alternative that cannot crank after the cold watch is out
    sc = aoa.score(crit, table)
    for k, e in evals.items():
        if not e["crank_after_cold_watch"]:
            sc["screened_out"].setdefault(k, []).append("crank_after_cold_watch")
            sc["fom"].pop(k, None)
            sc["utility"].pop(k, None)
    sc["ranking"] = sorted(sc["fom"], key=lambda a: -sc["fom"][a])
    surv = {k: v for k, v in table.items() if k in sc["fom"]}
    sens = aoa.sensitivity_10pct(crit, surv) if len(surv) > 1 else []
    mc = aoa.monte_carlo_weights(crit, surv, n=2000) if len(surv) > 1 else {}
    return {"criteria": c["criteria"], "weights": aoa.weights(crit).round(4).tolist(), "alternatives": evals,
            "score": sc, "sensitivity": sens, "mc_weights": mc}


# --- SSPC / wire coordination -------------------------------------------------------------------------
def _wire(csa: float, d_mm: float, w: dict) -> Wire:
    rho = 1.72e-8 * (1 + 0.00393 * (w["t_max_c"] - 20.0))
    r_per_m = rho / (csa * 1e-6)
    q_per_m = w["h_w_m2k"] * math.pi * d_mm * 1e-3 * (w["t_max_c"] - w["t_amb_c"])
    return Wire(csa, math.sqrt(q_per_m / r_per_m), w["k_adiabatic"])


def sspc_study() -> dict:
    c = cfg()
    s, w = c["sspc"], c["wire"]
    budget = data.budget()
    chans = {}
    for ld in budget.loads:
        peak = max(ld.states.values()) * ld.qty
        chans.setdefault(ld.channel, {"loads": [], "peak_w": 0.0, "prio": 9})
        chans[ld.channel]["loads"].append(ld.id)
        chans[ld.channel]["peak_w"] += peak
        chans[ld.channel]["prio"] = min(chans[ld.channel]["prio"], ld.priority)
    zone = data.zone()
    chans[[ld.channel for ld in budget.loads if ld.id == "ECU"][0]]["peak_w"] = zone.ecu_q_rated_w / zone.cop(49.0) + zone.ecu_fan_w
    rows = []
    worst = math.inf
    for ch in sorted(chans):
        d = chans[ch]
        i_peak = d["peak_w"] / 20.0                     # lowest steady bus voltage (MIL-STD-1275E 5.1.1.1)
        rating = next(r for r in s["ratings"] if r >= s["rating_factor"] * i_peak) if i_peak > 0 else s["ratings"][0]
        r = SSPCRating(rating, s["k_nt"], s["m_ref"], s["t_ref"], s["k_inst"], s["t_inst"])
        wire = None
        for csa, dmm in w["sizes"]:
            cand = _wire(csa, dmm, w)
            if coordination(r, cand, np.logspace(0, 1.5, 60))["coordinated"]:
                wire = cand
                break
        co = coordination(r, wire, np.logspace(0, 1.5, 60))
        worst = min(worst, co["min_ratio"])
        rows.append({"channel": ch, "loads": d["loads"], "priority": d["prio"], "peak_w": d["peak_w"], "i_peak_a": i_peak,
                     "rating_a": rating, "wire_mm2": wire.csa_mm2, "wire_iss_a": wire.i_ss_a, "min_damage_to_trip": co["min_ratio"]})
    # fault coverage on one representative channel (10 A, its wire), DOSSIER 9.3
    r10 = SSPCRating(10.0, s["k_nt"], s["m_ref"], s["t_ref"], s["k_inst"], s["t_inst"])
    w10 = next(_wire(a, b, w) for a, b in w["sizes"] if coordination(r10, _wire(a, b, w), np.logspace(0, 1.5, 60))["coordinated"])
    r_wire = 2 * w["run_m"] * 1.72e-8 * 1.0 / (w10.csa_mm2 * 1e-6)
    cases = [
        FaultCase("F1", "Bolted short at the load end", 28.0 / r_wire / 10.0, "trip"),
        FaultCase("F2", "Short to chassis mid-run (chafe)", 28.0 / (r_wire / 2) / 10.0, "trip"),
        FaultCase("F3", "Stalled fan motor (3 x rated)", 3.0, "trip"),
        FaultCase("F4", "Overload 150 %", 1.5, "trip"),
        FaultCase("F5", "Overload 120 % (inside the no-trip band of a fuse-like curve)", 1.2, "trip"),
        FaultCase("F6", "Open circuit / connector pulled", None, "undercurrent"),
        FaultCase("F7", "Series arc / high-resistance joint (current falls)", 0.8, "none"),
        FaultCase("F8", "Intermittent parallel arc (5 ms bursts at 8 x)", None, "none"),
    ]
    cov = fault_coverage(r10, w10, cases, undercurrent_a=0.1)
    covered = sum(1 for x in cov if x["covered"])
    return {"channels": rows, "worst_damage_to_trip": worst, "fault_cases": cov,
            "coverage": f"{covered}/{len(cov)}", "rep_wire_mm2": w10.csa_mm2,
            "curve": [(m, r10.trip_time(m * 10.0), w10.damage_time(m * 10.0)) for m in np.logspace(0, 1.6, 50)]}


# --- MIL-STD-1275E front end ---------------------------------------------------------------------------
FRONT_ENDS = {
    "v1_tvs36": {"v_br": 36.0, "r_dyn": 0.15, "tvs_pulse_j": 3.0, "fet_vds": 100.0},
    "v2_tvs122": {"v_br": 122.0, "r_dyn": 0.05, "tvs_pulse_j": 15.0, "fet_vds": 200.0},
    "v3_tvs122_2fet": {"v_br": 122.0, "r_dyn": 0.05, "tvs_pulse_j": 15.0, "fet_vds": 200.0, "rth": 0.3, "cth": 0.16,
                       "v_clamp": 36.0},
    # v4: the 1275F surge tail (decaying to 33 V at 500 ms) outlasts v3's 0.2 s fault timer, and the 1275F
    # 8.2 uF spike drives the input to ~172 V: fault timer 0.8 s, 250 V FETs
    "v4_final": {"v_br": 122.0, "r_dyn": 0.05, "tvs_pulse_j": 15.0, "fet_vds": 250.0, "rth": 0.3, "cth": 0.16,
                 "v_clamp": 36.0, "t_fault": 0.8},
}
PASS_RULES = {"reverse_polarity": "no_damage"}   # 1275E 4.1: protect itself; operation not required while reversed


def transient_study(cases: list[str] | None = None, fronts: list[str] | None = None) -> dict:
    d = yaml.safe_load((ROOT / "data" / "transients.yaml").read_text(encoding="utf-8"))
    out = {}
    for fe_name in fronts or FRONT_ENDS:
        spec = FRONT_ENDS[fe_name]
        fe = FrontEnd(**{k: v for k, v in spec.items() if k in FrontEnd.__dataclass_fields__})
        out[fe_name] = {}
        for name, c in d["cases"].items():
            if cases and name not in cases:
                continue
            r = simulate(waveform_from_dict(name, c), fe)
            checks = {
                "tvs_energy_ok": r.e_tvs <= fe.tvs_pulse_j,
                "fet_tj_ok": r.tj_peak <= fe.tj_max,
                "fet_vds_ok": float(np.max(r.va - np.minimum(r.vb, r.va))) <= spec["fet_vds"] * 0.8,
                "dcdc_vmax_ok": r.vb_max <= fe.v_dcdc_max,
                "no_dropout": r.dropouts == 0 or PASS_RULES.get(name) == "no_damage",
            }
            out[fe_name][name] = {"e_tvs_j": r.e_tvs, "e_fet_j": r.e_fet, "tj_peak_c": r.tj_peak, "vb_max": r.vb_max,
                                  "vb_min": r.vb_min, "va_max": float(r.va.max()), "dropouts": r.dropouts,
                                  "fet_trips": r.fet_trips, "checks": checks, "pass": all(checks.values())}
    return out


# --- arc-fault detection --------------------------------------------------------------------------------
def arc_study(n_each: int = 30, seed: int = 1000) -> dict:
    """Score FB_ArcDetect on held-out seeds (thresholds were tuned on seeds 0-99)."""
    from . import arcfault
    from .stlang import FBInstance, compile_file
    c = compile_file(ROOT / "plc" / "FB_ArcDetect.st")
    out: dict = {}
    for cs in arcfault.dataset(seed, n_each):
        fb = FBInstance(c)
        tr, dt, kind = arcfault.run_detector(cs, fb)
        key = cs.kind if cs.kind != "normal" else "nuisance: " + cs.name
        d = out.setdefault(key, {"n": 0, "tripped": 0, "times_ms": []})
        d["n"] += 1
        d["tripped"] += int(tr)
        if dt is not None:
            d["times_ms"].append(dt * 1000.0)
    for d in out.values():
        t = d.pop("times_ms")
        d["max_ms"] = max(t) if t else None
        d["median_ms"] = float(np.median(t)) if t else None
    return {"seed": seed, "n_each": n_each, "classes": out}


# --- reliability / availability -------------------------------------------------------------------------
def rma_study(alt_key: str = "A5") -> dict:
    c = cfg()
    spec = c["alternatives"][alt_key]
    arch = data.build_arch(spec["arch"], spec["np"], spec["apu"])
    budget = data.budget()
    ess_ch = len({ld.channel for ld in budget.loads if ld.priority <= 2})
    out = {}
    for label, k in (("k = n (no spare block)", arch.mission.np_), ("k = n - 1 (one spare block)", arch.mission.np_ - 1)):
        blocks = rbd_blocks(arch, ess_ch, k)
        out[label] = {"R_4h": rma.series_reliability(blocks, 4.0), "R_12h": rma.series_reliability(blocks, 12.0),
                      "importance_12h": rma.importance(blocks, 12.0)}
    blocks = rbd_blocks(arch, ess_ch, arch.mission.np_ - 1)
    av = rma.availability(blocks, c["rbd"]["ldt_h"], c["rbd"]["adt_h"])
    sens = {}
    for f in (0.5, 1.0, 3.0):
        bb = copy.deepcopy(blocks)
        for x in bb:
            x.fpmh *= f
        sens[f"x{f}"] = rma.series_reliability(bb, 12.0)
    return {"alternative": alt_key, "rbd": out, "availability": av, "rate_sensitivity_R12": sens,
            "blocks": [{"id": b.id, "name": b.name, "fpmh": b.fpmh, "n": b.n, "k": b.k} for b in blocks]}


# --- Monte Carlo on the uncertain inputs --------------------------------------------------------------
def monte_carlo(alt_key: str = "A2", climate: str = "A2_basic_hot", n: int = 120, seed: int = 11, dt: float = 30.0) -> dict:
    rng = np.random.default_rng(seed)
    c = cfg()
    spec = c["alternatives"][alt_key]
    hours = []
    for _ in range(n):
        b = data.budget()
        z = data.zone()
        tx = rng.uniform(0.05, 0.25)                         # command-net transmit duty
        for ld in b.loads:
            if ld.id in ("VHF1",):
                for mode in ("halt", "silent_watch"):
                    ld.duty[mode] = {"rx": 1 - tx, "tx": tx}
        g = rng.uniform(0.0, 0.15)                           # realised growth (instead of the allowance)
        b.growth = {k: g for k in b.growth}
        z.ecu_cop = [(t, cop * rng.uniform(0.85, 1.15)) for t, cop in z.ecu_cop]
        z.alpha_roof = rng.uniform(0.5, 0.85)
        z.crew = int(rng.integers(3, 6))
        arch = data.build_arch(spec["arch"], spec["np"], "")
        cap = rng.uniform(1.0, 1.2)                          # products exceed their spec minimum by 0-20 %
        arch.mission.p = copy.deepcopy(arch.mission.p)
        arch.mission.p.c_rated_ah *= cap
        r = endurance(spec["arch"], climate, spec["np"], "", hours=20, dt=dt, budget=b, zone=z, arch=arch)
        hours.append(_fin(r["t_essential_h"], 20.0))
    h = np.array(hours)
    thr = c["kpp"]["threshold_h"]
    return {"alternative": alt_key, "climate": climate, "n": n, "mean_h": float(h.mean()), "p05_h": float(np.percentile(h, 5)),
            "p50_h": float(np.median(h)), "p95_h": float(np.percentile(h, 95)), "p_meets_threshold": float((h >= thr).mean()),
            "samples": h.round(3).tolist()}


# --- the MIL-STD-1275 (16 V) vs MIL-PRF-32565C (14.4 V) cranking conflict ------------------------------
def conflict_study() -> dict:
    """Lowest bank voltage while cranking, full bank, per start-bank option and temperature.

    Cases: the vehicle starting-current profile of MIL-PRF-32565C Fig. 10 (2000 A for 0.1 s, ramp to
    1100 A at 0.6 s, held to 30 s) at 25 degC, and the class-B cranking ratings at -18 / -32 / -40 degC
    (1100 / 600 / 400 A for 30 s) applied as the engine demand.
    """
    from .battery import Pack
    blocks = data.blocks()
    banks = [("AGM 2s1p", "agm_6t", 2, 1), ("AGM 2s2p", "agm_6t", 2, 2), ("Li 1s1p", "li_6t", 1, 1),
             ("Li 1s2p", "li_6t", 1, 2)]
    cases = [("Fig. 10 profile, 25 degC", 25.0, None), ("1100 A, -18 degC", -18.0, 1100.0),
             ("600 A, -32 degC", -32.0, 600.0), ("400 A, -40 degC", -40.0, 400.0)]
    rows = []
    for label, blk, ns, np_ in banks:
        r = {"bank": label}
        for cname, t, i_const in cases:
            p = Pack(blocks[blk], ns, np_, soc=1.0, t_c=t)
            vmin, tt, dt = 1e9, 0.0, 0.01
            while tt < 30.0:
                i = i_const if i_const else (2000.0 if tt < 0.1 else (2000.0 - 1800.0 * (tt - 0.1) if tt < 0.6 else 1100.0))
                v = p.terminal_v(i) if p.soc > 1e-6 else 0.0
                vmin = min(vmin, v)
                p.step(i, dt, t)
                tt += dt
            r[cname] = vmin
        rows.append(r)
    return {"cases": [c[0] for c in cases], "rows": rows, "floor_1275": 16.0, "floor_32565c": 14.4}


# --- 4 h or 12 h: decision support ----------------------------------------------------------------
POSTURE_CONDITIONS = [("night", -32.0, 0.0), ("night", -18.0, 0.0), ("night", 0.0, 0.0), ("night", 15.0, 0.0),
                      ("night", 25.0, 0.0), ("day", 25.0, 1120.0), ("day", 35.0, 1120.0), ("day", 43.0, 1120.0),
                      ("day", 49.0, 1120.0)]


def postures() -> dict[str, set[int] | None]:
    b = data.budget()
    p12 = {ld.channel for ld in b.loads if ld.priority <= 2}
    p1 = {ld.channel for ld in b.loads if ld.priority == 1}
    keep = {ld.channel for ld in b.loads if ld.id in ("ECU", "CBRN")}
    return {"full (P1-P4 requested)": None, "essential (P1-P2 requested)": p12,
            "minimum (P1 + cooling + CBRN)": p1 | keep}


def decision_study(dt: float = 30.0, banks: tuple[int, ...] = (10, 12)) -> dict:
    """Hours each crew posture lasts, engine and APU off, vs ambient, for 10 and 12 Li-ion 6T blocks."""
    out = {"conditions": POSTURE_CONDITIONS, "rows": []}
    for name, req in postures().items():
        for n in banks:
            hours = []
            for _tag, t, g in POSTURE_CONDITIONS:
                r = endurance("A2_split_li", "temperate", n, "", hours=24, dt=dt, requested=req,
                              climate_override={"t_amb": t, "g_solar": g})
                h = r["t_full_h"] if req is None else r["t_essential_h"]
                hours.append(min(_fin(h, 24.0), 24.0))
            out["rows"].append({"posture": name, "blocks": n, "hours": hours})
    return out


# --- everything -----------------------------------------------------------------------------------------
def run_all(quick: bool = False) -> dict:
    OUT.mkdir(exist_ok=True)
    res = {
        "battery_validation": {"spec": validate_battery_models(), "product": validate_against_product()},
        "budget": budget_study(),
        "reserve": reserve_study(),
        "sizing": sizing_study(dt=30.0 if quick else 20.0),
        "aoa": aoa_study(dt=30.0 if quick else 20.0),
        "sspc": sspc_study(),
        "rma": rma_study(),
        "transients": transient_study(),
        "monte_carlo": monte_carlo(n=40 if quick else 120),
        "decision": decision_study(),
        "conflict": conflict_study(),
        "arc": arc_study(),
    }
    for k, v in res.items():
        (OUT / f"{k}.json").write_text(json.dumps(_clean(v), indent=1), encoding="utf-8", newline="\n")
    return res


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if k != "result"}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) else ("inf" if math.isinf(f) else round(f, 6))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x

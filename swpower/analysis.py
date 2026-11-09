"""The analyses behind the DOSSIER results (sections 7-13). Each returns plain data for the CLI,
the figures and the tests."""
from __future__ import annotations

import copy
import math

import numpy as np

from . import data
from .battery import Pack, crank_test
from .sim import Phase, run


# --- thermal coupling -----------------------------------------------------------------------
def equipment_heat(budget, mode: str, enabled: set[int] | None = None) -> float:
    return sum(ld.heat_avg(mode) * (1 + budget.growth[ld.maturity]) for ld in budget.loads
               if ld.in_bay and (enabled is None or ld.channel in enabled) and not ld.id.startswith("ECU"))


class ThermalCoupling:
    """extra_w hook for the simulator: cooling-unit draw and cold-climate heater electrics, and the
    mission-compartment air temperature (fed to the controller as TBay)."""

    def __init__(self, budget, zone, g_solar: float, dt: float, heater_w: float = 90.0):
        self.budget, self.zone, self.g, self.dt = budget, zone, g_solar, dt
        ecu = [ld for ld in budget.loads if ld.id.startswith("ECU")]
        cab = [ld for ld in budget.loads if ld.id.startswith("CABIN")]
        self.ch_ecu = ecu[0].channel if ecu else None
        self.ch_cab = cab[0].channel if cab else None
        self.heater_w = heater_w
        self.t_zone = None
        self.q_cool = 0.0

    def __call__(self, ph: Phase, enabled: set[int]) -> float:
        z = self.zone
        if self.t_zone is None:
            self.t_zone = min(z.t_limit_c, max(ph.t_amb, 15.0))
        q = equipment_heat(self.budget, ph.mode, enabled)
        p = 0.0
        q_cool = 0.0
        if self.ch_ecu is not None and self.ch_ecu in enabled:
            need = max(0.0, z.gains_at(z.t_limit_c, ph.t_amb, self.g, q)) if self.t_zone >= z.t_limit_c - 0.5 else 0.0
            q_cool = min(need, z.ecu_q_rated_w)
            if q_cool > 0:
                p += q_cool / z.cop(ph.t_amb) + z.ecu_fan_w
        if ph.t_amb < 5.0 and self.ch_cab is not None and self.ch_cab in enabled:
            p += self.heater_w            # fuel-fired air heater: fan, pump and glow electrics (ASSUMED 90 W)
        self.t_zone = z.step(self.t_zone, self.dt, ph.t_amb, self.g, q, q_cool)
        self.q_cool = q_cool
        return p


def ecu_hook(budget, zone, g_solar: float, dt: float = 10.0) -> ThermalCoupling:
    return ThermalCoupling(budget, zone, g_solar, dt)


# --- silent-watch endurance -------------------------------------------------------------------
def endurance(arch_name: str, climate: str, mission_np: int | None = None, with_apu: str | None = None,
              hours: float = 24.0, dt: float = 10.0, cbrn: bool = False, soc0: float = 1.0,
              silent_req: bool = True, budget=None, zone=None, arch=None, climate_override: dict | None = None,
              requested: set[int] | None = None) -> dict:
    """Silent watch from a charged state until the mission bank is cut off (or `hours`).

    silent_req=True forbids any automatic APU / engine start (a true silent watch)."""
    cl = dict(data.climates()[climate])
    if climate_override:
        cl.update(climate_override)
    budget = budget or data.budget()
    zone = zone or data.zone()
    arch = arch or data.build_arch(arch_name, mission_np, with_apu)
    soak = Phase("soak", 0.25, "halt", "idle", cl["t_amb"], rpm=cl.get("idle_rpm", 700))
    sw = Phase("silent watch", hours, "silent_watch", "off", cl["t_amb"], apu="auto" if arch.apu else "off",
               cbrn=cbrn, silent_req=silent_req)
    res = run(arch, budget, [soak, sw], data.crank("cold"), dt=dt, soc0=soc0,
              extra_w=ecu_hook(budget, zone, cl["g_solar"], dt), requested=requested)
    t0 = soak.hours
    sel = res.t_h >= t0

    def first(mask):
        v = res.first_time(mask & sel, t0)
        return (v - t0) if v is not None else math.inf

    t_l1 = first(res.level >= 1)
    t_l2 = first(res.level >= 2)
    t_l3 = first(res.level >= 3)
    t_lvd = first(res.lvd)
    # engine start after the watch: crank the start bank (split) or the common bank at its end state
    end = _bank_state_at(arch, res, -1)
    ok, vmin = crank_test(end, *_crank_args(data.crank("cold"), cl["t_amb"]))
    p_sw = float(np.mean(res.p_load[sel][: max(1, int(3600 / dt))]))
    return {"arch": arch_name, "climate": climate, "np": arch.mission.np_, "t_full_h": t_l1, "t_l2_h": t_l2,
            "t_essential_h": t_l3, "t_vital_h": t_lvd, "crank_ok": ok, "crank_vmin": vmin,
            "p_first_hour_w": p_sw, "mass_kg": arch.mission.mass_kg + (arch.start.mass_kg if arch.start else 0),
            "fuel_l": res.fuel_l - soak.hours * arch.engine_idle_lph, "apu_h": float(res.apu_on.sum() * dt / 3600), "result": res}


def _crank_args(c, t_amb):
    return (c.i_a, c.seconds, c.attempts, c.rest_s, t_amb, c.v_min)


def _bank_state_at(arch, res, idx) -> Pack:
    """Reconstruct the cranking bank's state at the end of a run (SOC and temperature)."""
    if arch.start is not None:
        bank = copy.deepcopy(arch.start)
        bank.soc = float(res.soc_s[idx])
        bank.t_c = float(res.t_batt_s[idx])
    else:
        bank = copy.deepcopy(arch.mission)
        bank.soc = float(res.soc_m[idx])
        bank.t_c = float(res.t_batt[idx])
    return bank


# --- engine-start reserve ---------------------------------------------------------------------
def crank_reserve_soc(block_name: str, ns: int, np_: int, t_amb: float, case: str = "cold") -> float:
    """Lowest SOC from which the bank still passes the crank requirement at t_amb (bisection)."""
    b = data.blocks()[block_name]
    c = data.crank(case)
    lo, hi = 0.0, 1.0
    p = Pack(b, ns, np_, soc=1.0, t_c=t_amb)
    if not crank_test(copy.deepcopy(p), *_crank_args(c, t_amb))[0]:
        return math.nan  # cannot start even when full
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        q = Pack(b, ns, np_, soc=mid, t_c=t_amb)
        if crank_test(q, *_crank_args(c, t_amb))[0]:
            hi = mid
        else:
            lo = mid
    return hi


# --- model validation against the specification sheets ---------------------------------------
def validate_battery_models() -> list[dict]:
    """Reproduce the specification ratings with the block models (DOSSIER 10.3)."""
    out = []
    b = data.blocks()
    agm = b["agm_6t"]
    # 20 h capacity: 6 A to 10.5 V at 25 degC
    out.append(_cap_test(agm, 6.0, 25.0, 20 * 60, "C20: 6 A for >= 20 h (120 Ah)"))
    out.append(_cap_test(agm, 25.0, 27.0, 230, "Reserve capacity: 25 A for >= 230 min"))
    out.append(_crank_spec(agm, 1100, -18.0, 7.2, "CCA: 1100 A, 30 s, -18 degC, >= 7.2 V"))
    out.append(_crank_spec(agm, 400, -40.0, 7.2, "Low temp: 400 A, 30 s, -40 degC, >= 7.2 V"))
    li = b["li_6t"]
    out.append(_cap_test(li, 90.0, 22.0, 60, "Capacity: 90 A for >= 1 h (90 Ah at 1 h rate)"))
    out.append(_crank_spec(li, 1100, -18.0, 14.4, "Cranking: 1100 A, 30 s, -18 degC, >= 14.4 V"))
    out.append(_crank_spec(li, 600, -32.0, 14.4, "Cranking: 600 A, 30 s, -32 degC, >= 14.4 V"))
    out.append(_crank_spec(li, 400, -40.0, 14.4, "Cranking: 400 A, 30 s, -40 degC, >= 14.4 V"))
    return out


# independent data: a qualified 6T AGM product's constant-current run-time table [AGM-DS1]
PRODUCT_RUNTIME = [(633.5, 2), (292.0, 10), (138.2, 30), (80.7, 60), (45.4, 120), (25.0, 240), (20.5, 300),
                  (11.0, 600), (6.0, 1200)]


def validate_against_product() -> list[dict]:
    """Model (spec-minimum block) vs a qualified product's run-time table, 25 degC, to 10.5 V."""
    agm = data.blocks()["agm_6t"]
    rows = []
    for i, minutes in PRODUCT_RUNTIME:
        p = Pack(agm, 1, 1, soc=1.0, t_c=25.0)
        t, dt = 0.0, max(1.0, minutes * 60 / 2000)
        while t < 3 * minutes * 60 and not p.at_cutoff(i):
            p.step(i, dt, 25.0)
            p.t_c = 25.0
            t += dt
        rows.append({"current_a": i, "product_min": minutes, "model_min": t / 60,
                     "ratio": (t / 60) / minutes})
    return rows


def _cap_test(blk, i, t_c, minutes_req, label) -> dict:
    p = Pack(blk, 1, 1, soc=1.0, t_c=t_c)
    t = 0.0
    while t < 48 * 3600:
        if p.at_cutoff(i):
            break
        p.step(i, 10.0, t_c)
        p.t_c = t_c                      # chamber-held temperature, as in the test
        t += 10.0
    return {"test": label, "model": f"{t / 60:.0f} min", "required": f">= {minutes_req} min",
            "pass": t / 60 >= minutes_req * 0.98}


def _crank_spec(blk, i, t_c, v_end, label) -> dict:
    p = Pack(blk, 1, 1, soc=1.0, t_c=t_c)
    ok, vmin = crank_test(p, i, 30.0, 1, 0.0, t_c, v_end - 0.005)
    return {"test": label, "model": f"{vmin:.2f} V", "required": f">= {v_end} V", "pass": ok}

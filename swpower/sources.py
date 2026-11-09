"""Charging sources: engine-driven alternator and auxiliary power unit (DOSSIER 8.3).

Both are modelled as a regulated voltage source with a current limit:

    I_src(V) = clip((V_set - V) / R_reg, 0, I_max)

* Alternator: I_max follows the published output-versus-speed curve (table, alternator rpm), with
  the alternator speed = engine speed x pulley ratio and a linear hot-ambient derating.
  V_set is temperature compensated for lead-acid charging.
* APU: a 28 V DC generator set with a rated output; fuel use is linear in load (Willans line),
  fitted to the published idle and rated-load figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_REG = 0.0002  # ohm: regulator stiffness used by the bus solver (numerical, not a physical part)


@dataclass
class Alternator:
    name: str
    curve: list[tuple[float, float]]       # (alternator rpm, max output A) at reference temperature
    pulley_ratio: float
    v_set_25: float                         # V regulation set point at 25 degC
    v_comp_per_k: float                     # V/K temperature compensation (negative)
    v_set_min: float
    v_set_max: float
    derate_from_c: float                    # ambient above which output derates
    derate_per_k: float                     # fraction of output lost per K above derate_from_c
    efficiency: float                       # mechanical -> electrical, at rated output
    src: list[str] = field(default_factory=list)

    def i_max(self, engine_rpm: float, t_amb: float) -> float:
        n_alt = engine_rpm * self.pulley_ratio
        r, i = zip(*self.curve)
        out = float(np.interp(n_alt, r, i, left=0.0))
        if t_amb > self.derate_from_c:
            out *= max(0.0, 1.0 - self.derate_per_k * (t_amb - self.derate_from_c))
        return out

    def v_set(self, t_batt: float) -> float:
        v = self.v_set_25 + self.v_comp_per_k * (t_batt - 25.0)
        return min(self.v_set_max, max(self.v_set_min, v))


@dataclass
class APU:
    name: str
    p_rated_w: float
    v_set: float
    fuel_idle_lph: float                    # L/h at no electrical load
    fuel_rated_lph: float                   # L/h at rated output
    mass_kg: float
    volume_l: float
    noise_dba_7m: float
    mtbf_h: float = 0.0
    cost_eur: float = 0.0
    src: list[str] = field(default_factory=list)

    @property
    def i_max(self) -> float:
        return self.p_rated_w / self.v_set

    def fuel_lph(self, p_out_w: float) -> float:
        x = min(max(p_out_w / self.p_rated_w, 0.0), 1.0)
        return self.fuel_idle_lph + x * (self.fuel_rated_lph - self.fuel_idle_lph)


def source_current(v_bus: float, v_set: float, i_max: float) -> float:
    return min(max((v_set - v_bus) / R_REG, 0.0), i_max)


def alternator_from_dict(d: dict) -> Alternator:
    kw = {k: v for k, v in d.items() if k in Alternator.__dataclass_fields__}
    kw["curve"] = [tuple(x) for x in d["curve"]]
    return Alternator(**kw)


def apu_from_dict(d: dict) -> APU:
    return APU(**{k: v for k, v in d.items() if k in APU.__dataclass_fields__})

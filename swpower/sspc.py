"""SSPC channel protection versus wire damage (DOSSIER section 9.3).

SSPC trip model (I^2t with a no-trip band and an instantaneous trip):

    no trip while I <= k_nt * I_n
    thermal memory  E(t) = integral (I^2 - (k_nt I_n)^2) dt,  trip when E >= K_t
    instantaneous trip when I >= k_inst * I_n after t_inst

K_t is set from one point of the published trip curve (trip time t_ref at m_ref x I_n).

Wire model (first-order, one conductor):  C' dT/dt = I^2 R'(T) - (T - T_a) / R'_th.
Its damage curve is the current that brings the conductor to T_max in time t:

    I_dmg(t) = I_ss / sqrt(1 - exp(-t / tau)),   I_ss^2 R' R'_th = T_max - T_a,
    and for t << tau the adiabatic limit  I^2 t = k^2 S^2  (IEC 60364-4-43 form), so tau = k^2 S^2 / I_ss^2.

Protection is coordinated when the SSPC trips before the wire reaches T_max at every current.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SSPCRating:
    i_n: float                 # rated current, A
    k_nt: float                # must-not-trip multiple (e.g. 1.15)
    m_ref: float               # reference overload multiple (e.g. 2.0)
    t_ref: float               # trip time at m_ref, s
    k_inst: float              # instantaneous trip multiple
    t_inst: float              # instantaneous trip time, s

    @property
    def k_thermal(self) -> float:
        return ((self.m_ref * self.i_n) ** 2 - (self.k_nt * self.i_n) ** 2) * self.t_ref

    def trip_time(self, i: float) -> float:
        """Trip time (s) at a constant current i from a cold channel; inf = no trip."""
        if i >= self.k_inst * self.i_n:
            return self.t_inst
        excess = i * i - (self.k_nt * self.i_n) ** 2
        if excess <= 0:
            return math.inf
        return min(self.k_thermal / excess, 1e6)


@dataclass
class Wire:
    csa_mm2: float
    i_ss_a: float              # steady current that just reaches t_max (free air, given ambient)
    k_adiabatic: float         # A s^0.5 / mm2 (115 PVC, 143 XLPE/EPR in IEC 60364-4-43)

    @property
    def tau_s(self) -> float:
        return (self.k_adiabatic * self.csa_mm2) ** 2 / self.i_ss_a ** 2

    def damage_time(self, i: float) -> float:
        if i <= self.i_ss_a:
            return math.inf
        x = 1.0 - (self.i_ss_a / i) ** 2
        return -self.tau_s * math.log(1.0 - x) if x < 1 else 0.0


def coordination(r: SSPCRating, w: Wire, multiples: np.ndarray) -> dict:
    """Trip time vs wire damage time over a range of current multiples of I_n."""
    rows = []
    worst = math.inf
    for m in multiples:
        i = m * r.i_n
        tt, td = r.trip_time(i), w.damage_time(i)
        ratio = td / tt if math.isfinite(tt) and tt > 0 else (math.inf if math.isinf(td) else 0.0)
        if math.isinf(tt) and math.isfinite(td):
            ratio = 0.0  # wire damaged with no trip: not coordinated
        worst = min(worst, ratio)
        rows.append((m, i, tt, td, ratio))
    return {"rows": rows, "min_ratio": worst, "coordinated": worst >= 1.0}


@dataclass
class FaultCase:
    id: str
    name: str
    current_multiple: float | None   # steady fault current as a multiple of I_n (None = no overcurrent)
    detected_by: str                  # what is supposed to detect it


def fault_coverage(r: SSPCRating, w: Wire, cases: list[FaultCase], undercurrent_a: float) -> list[dict]:
    """For each fault: does the SSPC clear it before the wire is damaged, or flag it at all?"""
    out = []
    for c in cases:
        if c.current_multiple is None:
            # open circuit / load loss: only the SSPC on-state undercurrent flag can see it
            out.append({"id": c.id, "name": c.name, "trip_s": None, "damage_s": None,
                        "covered": c.detected_by == "undercurrent", "how": c.detected_by})
            continue
        i = c.current_multiple * r.i_n
        tt, td = r.trip_time(i), w.damage_time(i)
        covered = math.isfinite(tt) and tt < td
        how = "trip" if math.isfinite(tt) else ("no trip, wire safe" if math.isinf(td) else "NOT COVERED")
        if math.isinf(tt) and math.isinf(td):
            covered = c.detected_by in ("none-needed", "undercurrent") or i < undercurrent_a
        out.append({"id": c.id, "name": c.name, "trip_s": tt, "damage_s": td, "covered": covered, "how": how})
    return out

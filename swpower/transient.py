"""MIL-STD-1275E transients and a protection front end for one mission-equipment input (DOSSIER 8.5).

Circuit (one equipment input, behind its SSPC channel):

    Vs(t) --Rs--+--[A]--+-- pass FET (ideal diode + surge stopper + current limit) --+--[B]--+-- DC/DC (P const)
                |       |                                                            |       |
               TVS     C_a                                                         C_hold   UVLO
               (bidirectional)

* TVS: I = sign(V) * max(|V| - V_br, 0) / R_dyn
* pass FET: blocks reverse current (reverse polarity, input dips); limits the output to V_clamp
  (linear mode, the FET takes V_A - V_B); limits the current to I_lim; a fault timer turns it off
  after t_fault in clamp or limit, and it stays off for t_cool.
* DC/DC: constant power P while on; on at V_on, off below V_off (UVLO hysteresis).
* FET junction: one-pole thermal model  dTj/dt = (P_fet - (Tj - T_case)/Rth) / Cth.

Node A is solved implicitly each step (backward Euler, bisection), node B by backward Euler too,
so the large source-impedance ratios do not make the step unstable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrontEnd:
    c_a: float = 2.2e-6          # F, input ceramic
    v_br: float = 36.0           # V, TVS breakdown
    r_dyn: float = 0.15          # ohm, TVS dynamic resistance
    tvs_pulse_j: float = 3.0     # J, TVS rating for a short pulse (assumed part class)
    r_on: float = 0.010          # ohm, pass FET on resistance
    v_clamp: float = 33.0        # V, surge-stopper output limit
    i_lim: float = 20.0          # A, current limit
    t_fault: float = 0.2         # s, time allowed in clamp or limit before the FET turns off
    t_cool: float = 1.0          # s, off time after a fault
    c_hold: float = 2.2e-3       # F, hold-up capacitance behind the ideal diode
    p_load: float = 150.0        # W, equipment input power
    v_on: float = 9.0            # V, DC/DC start
    v_off: float = 8.0           # V, DC/DC stop
    v_dcdc_max: float = 40.0     # V, DC/DC absolute maximum input
    rth: float = 0.6             # K/W junction-case (transient one-pole equivalent)
    cth: float = 0.08            # J/K
    tj_max: float = 175.0        # degC
    t_case: float = 85.0         # degC worst-case heatsink temperature


@dataclass
class Waveform:
    name: str
    points: list[tuple[float, float]]   # (t s, V) piecewise linear
    r_s: float                          # source impedance, ohm
    t_end: float
    dt: float
    note: str = ""
    src: list[str] = field(default_factory=list)
    cap_f: float = 0.0           # >0: energy-limited spike from a charged capacitor (1275F 8.2 uF circuit)
    cap_v0: float = 0.0          # its initial voltage (sign = polarity)
    cap_t_on: float = 0.0

    def v(self, t: float) -> float:
        tt, vv = zip(*self.points)
        return float(np.interp(t, tt, vv))

    def samples(self) -> np.ndarray:
        n = int(round(self.t_end / self.dt))
        tt, vv = zip(*self.points)
        return np.interp(np.arange(n) * self.dt, tt, vv)


@dataclass
class TransientResult:
    name: str
    t: np.ndarray
    vs: np.ndarray
    va: np.ndarray
    vb: np.ndarray
    on: np.ndarray
    p_fet: np.ndarray
    tj: np.ndarray
    e_fet: float
    e_tvs: float
    tj_peak: float
    vb_max: float
    vb_min: float
    dropouts: int
    fet_trips: int

    @property
    def passed(self) -> bool:
        return self.dropouts == 0 and self.vb_max <= 40.0 and self.tj_peak <= 175.0


def _tvs_i(v: float, fe: FrontEnd) -> float:
    a = abs(v) - fe.v_br
    return math.copysign(a / fe.r_dyn, v) if a > 0 else 0.0


def _solve_a(vs: float, va_prev: float, i_pass_of, fe: FrontEnd, wf: Waveform, dt: float) -> float:
    """Backward-Euler node A: C_a (V - Va_prev)/dt = (Vs - V)/Rs - I_tvs(V) - I_pass(V).

    I_pass(V) is non-decreasing in V, so the residual is strictly decreasing: bisection is safe.
    """
    def f(v: float) -> float:
        return (vs - v) / wf.r_s - _tvs_i(v, fe) - i_pass_of(v) - fe.c_a * (v - va_prev) / dt

    lo, hi = -400.0, 400.0
    for _ in range(56):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(wf: Waveform, fe: FrontEnd, v0: float = 28.0) -> TransientResult:
    n = int(round(wf.t_end / wf.dt))
    dt = wf.dt
    va, vb = v0, v0 - 0.05
    on = vb >= fe.v_on
    tj = fe.t_case
    fet_ok, t_in_limit, t_off = True, 0.0, 0.0
    e_fet = e_tvs = 0.0
    out = np.zeros((n, 7))
    dropouts = trips = 0
    g = dt / (fe.c_hold * fe.r_on)
    vs_all = wf.samples()
    v_cap = 0.0
    for k in range(n):
        t = k * dt
        if wf.cap_f > 0 and k == int(round(wf.cap_t_on / dt)):
            v_cap = wf.cap_v0
        vs = float(vs_all[k]) + v_cap
        i_load = fe.p_load / vb if on and vb > 1.0 else 0.0
        if fet_ok:
            # fully on: I = (Va - Vb_new)/R_on with C_hold (Vb_new - Vb)/dt = I - I_load
            def i_full(v: float, vb: float = vb, i_load: float = i_load) -> float:
                vb_new = (vb + g * v - dt * i_load / fe.c_hold) / (1.0 + g)
                return max(0.0, (v - vb_new) / fe.r_on)
            i_cap = min(fe.i_lim, max(0.0, i_load + fe.c_hold * (fe.v_clamp - vb) / dt))

            def i_pass_of(v: float, i_cap: float = i_cap, i_full=i_full) -> float:
                return min(i_full(v), i_cap)
            va = _solve_a(vs, va, i_pass_of, fe, wf, dt)
            i_pass = i_pass_of(va)
            limiting = i_full(va) > i_cap + 1e-9
        else:
            va = _solve_a(vs, va, lambda v: 0.0, fe, wf, dt)
            i_pass, limiting = 0.0, False
            t_off += dt
            if t_off >= fe.t_cool:
                fet_ok, t_off = True, 0.0
        vb = max(vb + (i_pass - i_load) * dt / fe.c_hold, 0.0)
        if wf.cap_f > 0 and v_cap != 0.0:
            v_cap -= (vs - va) / wf.r_s * dt / wf.cap_f      # the spike capacitor discharges into the input
        p_fet = max(va - vb, 0.0) * i_pass
        e_fet += p_fet * dt
        e_tvs += abs(_tvs_i(va, fe) * va) * dt
        tj += (p_fet - (tj - fe.t_case) / fe.rth) / fe.cth * dt
        if fet_ok and limiting and p_fet > 1.0:
            t_in_limit += dt
            if t_in_limit >= fe.t_fault:
                fet_ok, t_in_limit, trips = False, 0.0, trips + 1
        else:
            t_in_limit = 0.0
        if on and vb < fe.v_off:
            on, dropouts = False, dropouts + 1
        elif not on and vb >= fe.v_on:
            on = True
        out[k] = (t, vs, va, vb, on, p_fet, tj)
    return TransientResult(wf.name, out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4].astype(bool),
                           out[:, 5], out[:, 6], e_fet, e_tvs, float(out[:, 6].max()), float(out[:, 3].max()),
                           float(out[:, 3].min()), dropouts, trips)


def waveform_from_dict(name: str, d: dict) -> Waveform:
    """Build a 1275E case from data/transients.yaml (explicit points or a parametric shape)."""
    kind = d.get("kind", "points")
    t_end, dt = float(d["t_end"]), float(d["dt"])
    if kind == "cap_spike":
        pts = [(0.0, d["v_nom"]), (t_end, d["v_nom"])]
        return Waveform(name, pts, float(d["r_s"]), t_end, dt, d.get("note", ""), d.get("src", []),
                        cap_f=float(d["cap_f"]), cap_v0=float(d["cap_v0"]), cap_t_on=float(d["t_on"]))
    if kind == "surge_f":
        t = np.arange(0.0, t_end, dt)
        vn, vp, t_on = d["v_nom"], d["v_peak"], d["t_on"]
        tau = (d["t2"] - d["t1"]) / math.log((vp - vn) / (d["v_t2"] - vn))
        x = t - t_on
        v = np.where(x < 0, vn, np.where(x < d["t_rise"], vn + (vp - vn) * x / d["t_rise"],
                     np.where(x < d["t1"], vp, np.where(x < d["t2"], vn + (vp - vn) * np.exp(-(x - d["t1"]) / tau), vn))))
        pts = list(zip(t.tolist(), v.tolist()))
        return Waveform(name, pts, float(d["r_s"]), t_end, dt, d.get("note", ""), d.get("src", []))
    if kind == "points":
        pts = [tuple(p) for p in d["points"]]
    else:
        t = np.arange(0.0, t_end, dt)
        vn = d["v_nom"]
        if kind == "surge_exp":
            t_on, tr, tw = d["t_on"], d["t_rise"], d["t_width"]
            tau = (tw - tr) / math.log(10.0)          # decays to V10% at t_width
            x = t - t_on
            v = np.where(x < 0, vn, np.where(x < tr, vn + (d["v_peak"] - vn) * x / tr,
                                             vn + (d["v_peak"] - vn) * np.exp(-(x - tr) / tau)))
        elif kind == "spike":
            t_on, tr = d["t_on"], d["t_rise"]
            tau = d["t_env"] / math.log(10.0)
            x = t - t_on
            env = np.where(x < 0, 0.0, np.minimum(1.0, x / tr) * np.exp(-np.maximum(x, 0) / tau))
            v = vn + (d["v_peak"] - vn * np.sign(d["v_peak"])) * env * np.cos(2 * math.pi * d["f_ring"] * np.maximum(x, 0))
        elif kind == "ripple":
            v = vn + 0.5 * d["v_pp"] * np.sin(2 * math.pi * d["f"] * t)
        else:
            raise ValueError(f"{name}: unknown kind {kind}")
        pts = list(zip(t.tolist(), v.tolist()))
    return Waveform(name, pts, float(d["r_s"]), t_end, dt, d.get("note", ""), d.get("src", []))


def holdup_capacitance(p_w: float, t_s: float, v_start: float, v_min: float, eta: float = 1.0) -> float:
    """C such that a constant-power load p_w can run t_s from v_start down to v_min."""
    return 2.0 * p_w * t_s / (eta * (v_start ** 2 - v_min ** 2))

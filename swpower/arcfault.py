"""DC arc-fault detection for SSPC channels (DOSSIER 9.3).

I2t SSPCs miss two fault types (FMEA FM12): series arcs at loose terminals (the current does not
rise) and intermittent parallel arcs (bursts too short for the I2t memory). This module provides:

* a signal model of one channel's current at fs = 200 kS/s:
  - normal loads (resistive or constant-power DC/DC), with nuisance features: radio transmit keying
    with in-band modulation ripple, a brushed fan with commutation ripple and start inrush,
    capacitive switch-on inrush, MIL-STD-1275F bus ripple (4 V p-p up to 5 kHz) and the 12 V
    initial engagement surge seen through a constant-power load;
  - a series arc: a ~15 V arc in series [ARC-WARWICK], repeated extinction and re-strike every
    ~2 ms (mean arc life 1.98-2.13 ms in 28 V loose-terminal tests [ARC-WARWICK]) and broadband
    noise; with a resistive load the current falls, with a constant-power load it rises;
  - a parallel arc: short bursts of > 50 A at about 20 V arc voltage [ARC-WARWICK], limited by
    the source and wire resistance, repeated at random intervals (vibration);
* the detector front end that would run in the SSPC's signal processor: a 5-20 kHz band-pass and
  envelope as in the NASA 28 V detector [ARC-NASA], current-dip counting, and burst timing against
  the channel's running mean, reduced to one feature frame per millisecond;
* the decision logic is the ST block plc/FB_ArcDetect.st, scanned once per frame.

The signal shapes are synthetic: amplitudes of the noise and of the nuisance ripples are ASSUMED
and stated per case. The thresholds were tuned on seeds 0-99 and are scored on seeds 1000+.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

FS = 200_000.0           # samples per second
FRAME = 200              # samples per 1 ms feature frame
V_BUS = 28.0
R_WIRE = 0.02            # ohm, channel wire, assumed
SOS_BAND = signal.butter(4, [5_000, 20_000], btype="bandpass", fs=FS, output="sos")
SOS_ENV = signal.butter(2, 1_000, btype="lowpass", fs=FS, output="sos")


@dataclass
class Case:
    name: str
    kind: str                # "normal", "series", "parallel"
    i_rated: float           # SSPC channel rating, A
    current: np.ndarray      # A, at FS
    t_fault: float | None    # s, fault start (None for normal)


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * FS)) / FS


def _band_noise(rng, n: int, rel: float) -> np.ndarray:
    w = rng.standard_normal(n)
    b = signal.sosfilt(signal.butter(2, [1_000, 60_000], btype="bandpass", fs=FS, output="sos"), w)
    return rel * b / (np.std(b) + 1e-12)


def load_current(rng, kind: str, p_w: float, v: np.ndarray) -> np.ndarray:
    """Current of a resistive or constant-power load fed with v(t)."""
    if kind == "resistive":
        r = V_BUS ** 2 / p_w
        return v / r
    return p_w / np.maximum(v, 8.0)


def make_normal(rng, scenario: str, dur: float = 1.0) -> Case:
    t = _t(dur)
    n = len(t)
    v = np.full(n, V_BUS)
    if scenario == "radio_tx":
        # 65 W receive, 450 W transmit for 0.2-0.6 s keyed at a random time; transmit ripple from the
        # waveform envelope at 2-8 kHz, 10-20 % of the transmit current (ASSUMED)
        p = np.full(n, 65.0)
        a = rng.uniform(0.1, 0.3)
        b = min(dur, a + rng.uniform(0.2, 0.6))
        on = (t >= a) & (t < b)
        f = rng.uniform(2_000, 8_000)
        rip = 1 + rng.uniform(0.10, 0.20) * np.sin(2 * np.pi * f * t)
        p = np.where(on, 450.0 * rip, p)
        i = load_current(rng, "cp", 1.0, v) * p
        return Case("radio transmit keying", "normal", 35.0, i + _band_noise(rng, n, 0.02), None)
    if scenario == "fan_start":
        # brushed fan: 5 x inrush decaying over 0.3 s, then running with commutation ripple at 1-6 kHz, 15-30 %
        i_run = 300.0 / V_BUS
        a = rng.uniform(0.05, 0.2)
        inrush = np.where(t >= a, 4.0 * np.exp(-(t - a) / 0.08), 0.0)
        f = rng.uniform(1_000, 6_000)
        rip = rng.uniform(0.15, 0.30) * np.sin(2 * np.pi * f * t)
        i = np.where(t >= a, i_run * (1 + inrush + rip), 0.0)
        return Case("fan motor start + commutation ripple", "normal", 20.0, i + _band_noise(rng, n, 0.02), None)
    if scenario == "cap_inrush":
        # switch-on of a DC/DC: 10 x rated for 0.5 ms, then 150 W
        a = rng.uniform(0.1, 0.5)
        i = np.where(t >= a, 150.0 / V_BUS, 0.0)
        i = i + np.where((t >= a) & (t < a + 5e-4), 10 * 10.0, 0.0)
        return Case("capacitive switch-on inrush", "normal", 10.0, i + _band_noise(rng, n, 0.02), None)
    if scenario == "bus_ripple":
        # MIL-STD-1275F 5.1.1.2.1: 4 V p-p at 30 Hz-5 kHz, here near the band edge
        f = rng.uniform(1_000, 5_000)
        v = 24.0 + 2.0 * np.sin(2 * np.pi * f * t)
        i = load_current(rng, "cp", 150.0, v)
        return Case("1275F bus ripple through a DC/DC", "normal", 10.0, i + _band_noise(rng, n, 0.02), None)
    if scenario == "ies":
        # 12 V initial engagement surge for 0.3-0.6 s through a constant-power load
        a = rng.uniform(0.1, 0.3)
        v = np.where((t >= a) & (t < a + rng.uniform(0.3, 0.6)), 12.0, V_BUS)
        v = signal.sosfilt(signal.butter(1, 2_000, fs=FS, output="sos"), v - V_BUS) + V_BUS
        i = load_current(rng, "cp", 150.0, v)
        return Case("12 V initial engagement surge", "normal", 10.0, i + _band_noise(rng, n, 0.02), None)
    raise ValueError(scenario)


def make_series_arc(rng, dur: float = 1.0) -> Case:
    """Loose terminal starts arcing at t0: 15 V arc, extinction / re-strike, broadband noise."""
    t = _t(dur)
    n = len(t)
    load = rng.choice(["resistive", "cp"])
    p = rng.uniform(100.0, 600.0)
    i_rated = 10.0 if p < 250 else 35.0
    t0 = rng.uniform(0.15, 0.4)
    v_arc = rng.uniform(13.0, 17.0)                    # ~15 V [ARC-WARWICK], >= 13 V copper minimum [ARC-RELAY]
    arcing = t >= t0
    v_load = np.where(arcing, V_BUS - v_arc, V_BUS)
    i = load_current(rng, load, p, v_load)
    # extinction and re-strike: each arc lives ~2 ms, the gap is 0.05-0.4 ms with near-zero current
    dips = np.zeros(n, dtype=bool)
    k = int(t0 * FS)
    while k < n:
        k += int(rng.normal(2.05e-3, 0.3e-3) * FS)
        w = int(rng.uniform(0.05e-3, 0.4e-3) * FS)
        dips[k:k + w] = True
        k += w
    i = np.where(arcing & dips, i * 0.05, i)
    i = i + np.where(arcing, _band_noise(rng, n, rng.uniform(0.05, 0.25)) * i, 0.0)   # arc noise, ASSUMED level
    i = i + _band_noise(rng, n, 0.02)
    return Case(f"series arc, {load} load {p:.0f} W", "series", i_rated, i, t0)


def make_parallel_arc(rng, dur: float = 1.0) -> Case:
    """Chafed wire touching structure: bursts of 2-4 ms at (28 - 20 V) / R_loop, at random intervals."""
    t = _t(dur)
    n = len(t)
    i_rated = rng.choice([20.0, 35.0, 75.0])
    i_load = 0.5 * i_rated
    r_loop = rng.uniform(0.04, 0.12)                    # source + wire to the fault, ASSUMED
    i_burst = (V_BUS - 20.0) / r_loop                   # ~20 V arc [ARC-WARWICK] -> 67-200 A
    t0 = rng.uniform(0.1, 0.3)
    i = np.full(n, i_load)
    k = int(t0 * FS)
    while k < n:
        w = int(rng.uniform(2e-3, 4e-3) * FS)
        i[k:k + w] += i_burst * (1 + 0.2 * rng.standard_normal(min(w, n - k)))
        k += w + int(rng.uniform(5e-3, 80e-3) * FS)
    return Case(f"parallel arc, {i_burst:.0f} A bursts on a {i_rated:.0f} A channel", "parallel", i_rated,
                i + _band_noise(rng, n, 0.02), t0)


def features(c: Case) -> dict[str, np.ndarray]:
    """Per-millisecond features computed in the SSPC signal processor."""
    i = c.current
    band = signal.sosfilt(SOS_BAND, i)
    env = signal.sosfilt(SOS_ENV, np.abs(band))
    nf = len(i) // FRAME
    fr = i[: nf * FRAME].reshape(nf, FRAME)
    mean = np.maximum(np.abs(fr.mean(axis=1)), 0.2)
    hf = env[: nf * FRAME].reshape(nf, FRAME).mean(axis=1) / mean
    # dips: samples below 30 % of the frame mean, counted as runs of >= 4 samples (20 us)
    low = fr < 0.3 * mean[:, None]
    # a dip = a run of >= 4 low samples (20 us): count positions where a 4-sample low run ends
    lw = low.astype(np.int8)
    run4 = lw[:, 3:] & lw[:, 2:-1] & lw[:, 1:-2] & lw[:, :-3]
    ends = run4 & np.concatenate((1 - lw[:, 4:], np.ones((nf, 1), dtype=np.int8)), axis=1)
    dips = ends.sum(axis=1).astype(int)
    # bursts: above twice the channel's 100 ms running mean and above 30 A (parallel-arc current is set
    # by the loop resistance, not by the channel rating, so a rating-relative threshold misses it)
    w = int(0.1 * FS)
    cs = np.concatenate(([0.0], np.cumsum(i)))
    idx = np.arange(1, len(i) + 1)
    base = (cs[idx] - cs[np.maximum(idx - w, 0)]) / np.minimum(idx, w)    # trailing 100 ms mean
    thr = np.maximum(2.0 * np.abs(base), 30.0)
    burst_ms = (i[: nf * FRAME] > thr[: nf * FRAME]).reshape(nf, FRAME).sum(axis=1) / FS * 1e3
    return {"hf": hf, "dips": dips, "burst_ms": burst_ms, "i_mean": mean}


def dataset(seed0: int, n_each: int = 20) -> list[Case]:
    rng = np.random.default_rng(seed0)
    cases = []
    for _ in range(n_each):
        cases.append(make_series_arc(rng))
        cases.append(make_parallel_arc(rng))
        for sc in ("radio_tx", "fan_start", "cap_inrush", "bus_ripple", "ies"):
            cases.append(make_normal(rng, sc))
    return cases


def run_detector(c: Case, fb) -> tuple[bool, float | None, int]:
    """Scan FB_ArcDetect once per 1 ms frame. Returns (tripped, time from fault start to trip, kind)."""
    f = features(c)
    fb.Reset = False
    for k in range(len(f["hf"])):
        fb.HfNorm = float(f["hf"][k])
        fb.Dips = int(f["dips"][k])
        fb.BurstMs = float(f["burst_ms"][k])
        fb.IMean = float(f["i_mean"][k])
        fb.scan(1)
        if fb.Trip:
            t_trip = (k + 1) / 1000.0
            dt = None if c.t_fault is None else t_trip - c.t_fault
            return True, dt, int(fb.Kind)
    return False, None, 0

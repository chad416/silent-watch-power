"""Time-domain mission simulation (DOSSIER section 10, M&S plan).

The vehicle 28 V system is a start bank and a mission bank joined by a combiner, or one common bank
(architecture option). Each time step:

1. the phase sets engine state, engine speed, ambient temperature and the mode of operation;
2. the load-management controller (plc/FB_PowerMgmt.st, run through the ST translator) reads the
   bus and battery state and returns the SSPC channel commands, the combiner command and the APU
   request;
3. the bus solver finds the bus voltage(s) for the enabled loads and the on-line sources;
4. the packs advance (SOC, polarisation, temperature, heater).

The engine-start check runs the crank profile on the bank that must start the engine, at the
ambient temperature of that moment, after the silent watch (DOSSIER 8.4).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .battery import Pack, crank_test
from .bus import solve_bus
from .loads import Budget
from .sources import APU, Alternator
from .stlang import FBInstance, compile_file

ROOT = Path(__file__).resolve().parents[1]
ST_PATH = ROOT / "plc" / "FB_PowerMgmt.st"
_COMPILED = None


def controller() -> FBInstance:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = compile_file(ST_PATH)
    return FBInstance(_COMPILED)


@dataclass
class Phase:
    name: str
    hours: float
    mode: str                     # drive / halt / silent_watch / cold_start
    engine: str                   # off / idle / drive
    t_amb: float
    rpm: float = 0.0
    apu: str = "off"              # off / auto / on
    crank_at_start: bool = False
    cbrn: bool = False
    silent_req: bool = False


@dataclass
class Crank:
    i_a: float                    # cranking current (A) at the battery
    seconds: float
    attempts: int
    rest_s: float
    v_min: float                  # lowest acceptable terminal voltage during cranking


@dataclass
class Architecture:
    name: str
    mission: Pack
    start: Pack | None            # None = one common bank
    alternator: Alternator
    apu: APU | None
    soc_reserve: float            # controller floor for the mission bank (used when no table)
    keepalive_w: float            # start-bus loads while the engine is off
    engine_idle_lph: float
    bay_heat_extra_w: float = 0.0
    reserve_table: list | None = None   # [(degC, SOC)]: temperature-scheduled floor (common bank)


@dataclass
class Result:
    t_h: np.ndarray
    v_bus: np.ndarray
    soc_m: np.ndarray
    soc_s: np.ndarray
    t_batt: np.ndarray
    p_load: np.ndarray
    p_src: np.ndarray
    level: np.ndarray
    lvd: np.ndarray
    apu_on: np.ndarray
    phase_idx: np.ndarray
    t_batt_s: np.ndarray
    t_zone: np.ndarray
    fuel_l: float
    cranks: list[dict] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def first_time(self, mask: np.ndarray, after_h: float = 0.0) -> float | None:
        idx = np.nonzero(mask & (self.t_h >= after_h))[0]
        return float(self.t_h[idx[0]]) if len(idx) else None


def _enabled_channels(fb: FBInstance) -> set[int]:
    return {i + 1 for i, on in enumerate(fb.ChOn) if on}


def run(arch: Architecture, budget: Budget, phases: list[Phase], crank: Crank, dt: float = 5.0,
        soc0: float = 1.0, extra_w=None, auto_start: bool = True, trips=None,
        requested: set[int] | None = None) -> Result:
    """Simulate the phases in order. extra_w(phase, enabled) -> W adds computed loads (e.g. cooling).

    trips: optional callable (t_h, channel) -> bool forcing an SSPC trip (fault injection).
    requested: channels the crew switches on (None = every configured channel).
    """
    arch = copy.deepcopy(arch)
    m, s = arch.mission, arch.start
    m.soc = soc0
    if s is not None:
        s.soc = soc0
    m.t_c = phases[0].t_amb
    if s is not None:
        s.t_c = phases[0].t_amb
    fb = controller()
    prio = budget.channel_priority()
    fb.Prio = [prio.get(i, 0) for i in range(1, 17)]
    fb.ChReq = [i in prio and (requested is None or i in requested) for i in range(1, 17)]
    fb.CycleMs = int(dt * 1000)
    fb.SocReserve = arch.soc_reserve
    fb.UsableWh = m.blocks * m.p.v_nom * m.p.capacity_ah(25.0)
    cbrn_ch = [ld.channel for ld in budget.loads if ld.id.startswith("CBRN")]
    fb.CbrnCh = cbrn_ch[0] if cbrn_ch else 0
    fb.AutoStartEn = auto_start and arch.apu is not None
    tripped: set[int] = set()

    rows = []
    fuel = 0.0
    t = 0.0
    cranks, events = [], []
    apu_running = False
    for k, ph in enumerate(phases):
        if ph.crank_at_start:
            bank = copy.deepcopy(s if s is not None else m)
            ok, vmin = crank_test(bank, crank.i_a, crank.seconds, crank.attempts, crank.rest_s, ph.t_amb,
                                  crank.v_min)
            cranks.append({"phase": ph.name, "t_amb": ph.t_amb, "soc": bank.soc, "ok": ok, "v_min": vmin,
                           "t_h": t / 3600})
            # charge the real bank with one attempt's worth of energy
            real = s if s is not None else m
            real.step(crank.i_a, crank.seconds, ph.t_amb)
        n = int(round(ph.hours * 3600 / dt))
        for _ in range(n):
            engine_on = ph.engine != "off"
            apu_cmd = ph.apu == "on" or (ph.apu == "auto" and fb.ApuStartReq)
            apu_running = arch.apu is not None and apu_cmd and not engine_on
            charge_ok = engine_on or apu_running
            # controller scan with the present measurements
            fb.ChargeOK = charge_ok
            fb.CbrnAlarm = ph.cbrn
            fb.SilentReq = ph.silent_req
            fb.ChTrip = [i in tripped for i in range(1, 17)]
            enabled = _enabled_channels(fb)
            p_load = budget.p_mode(ph.mode, enabled) + (extra_w(ph, enabled) if extra_w else 0.0)
            # sources
            v_set, i_max = None, 0.0
            if engine_on:
                v_set = arch.alternator.v_set(m.t_c)
                i_max = arch.alternator.i_max(ph.rpm, ph.t_amb)
            elif apu_running:
                v_set, i_max = arch.apu.v_set, arch.apu.i_max
            p_keep = arch.keepalive_w
            heat_m = m.heater_power()
            if s is not None and fb.CombinerCmd:
                packs = [m, s]
                res = solve_bus(packs, p_load + p_keep + heat_m + s.heater_power(), v_set, i_max)
                i_m, i_s = res.i_packs
                v_bus = res.v
            elif s is not None:
                res = solve_bus([m], p_load + heat_m, v_set, i_max)
                i_m = res.i_packs[0]
                v_bus = res.v
                rs = solve_bus([s], p_keep + s.heater_power(), v_set if engine_on else None,
                               i_max if engine_on else 0.0)
                i_s = rs.i_packs[0]
            else:
                res = solve_bus([m], p_load + p_keep + heat_m, v_set, i_max)
                i_m = res.i_packs[0]
                i_s = 0.0
                v_bus = res.v
            if arch.reserve_table:
                tt, rr = zip(*arch.reserve_table)
                fb.SocReserve = float(np.interp(m.t_c, tt, rr))
            fb.VBus = v_bus
            fb.SocMB = m.soc
            fb.TBay = getattr(extra_w, "t_zone", None) or 30.0
            fb.PLoadW = p_load
            fb.scan(int(dt * 1000))
            # SSPC side: trip injection and reset
            if trips is not None:
                for ch in range(1, 17):
                    if fb.ChResetCmd[ch - 1]:
                        tripped.discard(ch)
                    if ch in _enabled_channels(fb) and trips(t / 3600, ch):
                        tripped.add(ch)
            # advance
            m.step(i_m, dt, ph.t_amb)
            if s is not None:
                s.step(i_s, dt, ph.t_amb)
            if engine_on and ph.engine == "idle":
                fuel += arch.engine_idle_lph * dt / 3600
            if apu_running:
                fuel += arch.apu.fuel_lph(res.i_src * v_bus) * dt / 3600
            rows.append((t / 3600, v_bus, m.soc, s.soc if s is not None else np.nan, m.t_c, p_load,
                         res.i_src * v_bus, fb.ShedLevel, fb.LvdActive, apu_running, k,
                         s.t_c if s is not None else np.nan, getattr(extra_w, "t_zone", None) or np.nan))
            t += dt
        events.append(f"{t / 3600:6.2f} h end of {ph.name}: SOC_m {m.soc:.2f}"
                      + (f", SOC_s {s.soc:.2f}" if s is not None else "") + f", level {fb.ShedLevel}")
    a = np.array(rows, dtype=float)
    return Result(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7].astype(int),
                  a[:, 8].astype(bool), a[:, 9].astype(bool), a[:, 10].astype(int), a[:, 11], a[:, 12], fuel, cranks, events)

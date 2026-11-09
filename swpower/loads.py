"""Load inventory and power budget (DOSSIER section 7, Wasson ch. 31).

Each load entry carries its source, its maturity (which sets the growth allowance), its priority
(which sets the load-shed order) and its electrical states. The average power in a mode is the
duty-weighted sum of its state powers:

    P_mode = qty * sum_s duty[mode][s] * P_state[s]

The budget carries two layers on top of the current best estimate (CBE), as Wasson ch. 31 asks:

* growth allowance per item by maturity (off-the-shelf / modified / new design), and
* a system-level margin held by the power-system owner (not allocated to any item).
"""
from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("drive", "halt", "silent_watch", "cold_start")


@dataclass
class Load:
    id: str
    name: str
    qty: int
    priority: int                    # 1 vital ... 4 comfort (shed first)
    channel: int                     # SSPC channel
    maturity: str                    # cots / modified / new
    in_bay: bool                     # dissipates into the electronics bay
    states: dict[str, float]         # W per unit at 28 V, per state
    duty: dict[str, dict[str, float]]  # mode -> state -> fraction (sums to 1)
    rf_out_w: dict[str, float] = field(default_factory=dict)  # radiated RF per state (not heat)
    basis: str = "sourced"           # sourced / derived / assumed
    src: list[str] = field(default_factory=list)
    note: str = ""

    def p_avg(self, mode: str) -> float:
        d = self.duty.get(mode, {})
        return self.qty * sum(frac * self.states[s] for s, frac in d.items())

    def p_peak(self, mode: str) -> float:
        d = self.duty.get(mode, {})
        return self.qty * max((self.states[s] for s, f in d.items() if f > 0), default=0.0)

    def heat_avg(self, mode: str) -> float:
        """Average heat released (W): electrical input minus radiated RF."""
        d = self.duty.get(mode, {})
        rf = self.qty * sum(frac * self.rf_out_w.get(s, 0.0) for s, frac in d.items())
        return self.p_avg(mode) - rf


@dataclass
class Budget:
    loads: list[Load]
    growth: dict[str, float]          # maturity -> growth allowance fraction
    system_margin: float              # fraction held at system level

    def by_mode(self, mode: str, max_priority: int = 4) -> dict:
        cbe = sum(ld.p_avg(mode) for ld in self.loads if ld.priority <= max_priority)
        grown = sum(ld.p_avg(mode) * (1 + self.growth[ld.maturity]) for ld in self.loads
                    if ld.priority <= max_priority)
        peak = sum(ld.p_peak(mode) for ld in self.loads if ld.priority <= max_priority)
        return {"mode": mode, "cbe_w": cbe, "with_growth_w": grown,
                "allocated_w": grown * (1 + self.system_margin), "coincident_peak_w": peak}

    def p_mode(self, mode: str, enabled: set[int] | None = None, use_growth: bool = True) -> float:
        tot = 0.0
        for ld in self.loads:
            if enabled is not None and ld.channel not in enabled:
                continue
            g = (1 + self.growth[ld.maturity]) if use_growth else 1.0
            tot += ld.p_avg(mode) * g
        return tot

    def channels_by_priority(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for ld in self.loads:
            out.setdefault(ld.priority, []).append(ld.channel)
        return out

    def channel_priority(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for ld in self.loads:
            out[ld.channel] = min(out.get(ld.channel, 9), ld.priority)
        return out


def load_from_dict(d: dict) -> Load:
    return Load(**{k: v for k, v in d.items() if k in Load.__dataclass_fields__})


def validate(loads: list[Load]) -> list[str]:
    errs = []
    seen = set()
    for ld in loads:
        if ld.id in seen:
            errs.append(f"{ld.id}: duplicate id")
        seen.add(ld.id)
        if not 1 <= ld.priority <= 4:
            errs.append(f"{ld.id}: priority {ld.priority} not in 1..4")
        if not 1 <= ld.channel <= 16:
            errs.append(f"{ld.id}: channel {ld.channel} not in 1..16")
        for mode, d in ld.duty.items():
            if mode not in MODES:
                errs.append(f"{ld.id}: unknown mode {mode}")
            s = sum(d.values())
            if d and abs(s - 1.0) > 1e-6:
                errs.append(f"{ld.id}: duty in {mode} sums to {s:.3f}")
            for st in d:
                if st not in ld.states:
                    errs.append(f"{ld.id}: unknown state {st} in {mode}")
        if ld.basis not in ("sourced", "derived", "assumed"):
            errs.append(f"{ld.id}: basis {ld.basis}")
        if ld.basis != "assumed" and not ld.src:
            errs.append(f"{ld.id}: basis {ld.basis} but no source")
    return errs

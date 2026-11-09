"""28 V bus solver (DOSSIER 8.1).

Nodes: one bus with any number of packs, one regulated source (alternator or APU, or none) and a
constant-power load (the DC/DC converters of the mission equipment hold their power as the voltage
moves). The bus voltage V solves the current balance

    I_src(V) + sum_k I_pack_k(V) - P_load / V = 0

I_pack_k(V) is decreasing in V, but the constant-power term makes the balance non-monotone, and
pack current limits can create a second, low root. The stable operating point is the highest root:
the solver scans down from 40 V in 0.5 V steps to bracket it, then bisects. No root above 6 V means
the bus has collapsed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .battery import Pack
from .sources import source_current

V_LO, V_HI = 6.0, 40.0


@dataclass
class BusResult:
    v: float
    i_src: float
    i_packs: list[float]
    i_load: float
    collapsed: bool


def solve_bus(packs: list[Pack], p_load_w: float, v_set: float | None = None, i_src_max: float = 0.0,
              tol: float = 1e-4) -> BusResult:
    # pack parameters do not depend on the bus voltage: evaluate them once per solve
    pk = []
    for p in packs:
        i_ch = p.np_ * p.p.i_charge_max_a if p.charge_allowed() else 0.0
        pk.append((p.emf(), p.r_int(), i_ch, p.np_ * p.p.i_max_a))

    def balance(v: float) -> float:
        s = source_current(v, v_set, i_src_max) if v_set is not None else 0.0
        for e, r, i_ch, i_mx in pk:
            s += min(max((e - v) / r, -i_ch), i_mx)
        return s - p_load_w / v

    # the stable operating point is the highest root: scan down from V_HI for the first sign change
    hi = V_HI
    lo = None
    v = V_HI - 0.5
    while v >= V_LO:
        if balance(v) > 0:
            lo = v
            break
        hi = v
        v -= 0.5
    if lo is None:
        # no voltage in range where the sources carry the load: the bus has collapsed
        return BusResult(V_LO, 0.0, [p.current_at(V_LO) for p in packs], p_load_w / V_LO, True)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if balance(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    v = 0.5 * (lo + hi)
    i_src = source_current(v, v_set, i_src_max) if v_set is not None else 0.0
    return BusResult(v, i_src, [p.current_at(v) for p in packs], p_load_w / v, False)

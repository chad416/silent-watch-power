"""Reliability, maintainability and availability (DOSSIER section 12, Wasson ch. 34).

Formulas (Wasson 2016, ch. 34; the printed Eq. 34.37 has a typo, the sum form is used here):
    R(t) = exp(-lambda t)                         constant failure rate (useful-life period only)
    R_series = prod R_i = exp(-sum lambda_i t)    Eq. 34.36 / 34.37
    R_k_of_n (identical, independent) = sum_{j=k..n} C(n,j) R^j (1-R)^(n-j)   (Eq. 34.39 is the 1-of-n case)
    A_i = MTBF / (MTBF + MCMT)                     Eq. 34.53
    A_o = MTBM / (MTBM + MDT),  MDT = MAMT + LDT + ADT    Eq. 34.51, 34.42
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def r_exp(fpmh: float, hours: float) -> float:
    """Reliability over `hours` for a failure rate in failures per million hours."""
    return math.exp(-fpmh * 1e-6 * hours)


def k_of_n(r: float, k: int, n: int) -> float:
    return sum(math.comb(n, j) * r ** j * (1 - r) ** (n - j) for j in range(k, n + 1))


@dataclass
class Block:
    """One RBD block: n identical items of which k must work."""
    id: str
    name: str
    fpmh: float                  # per item, failures per 10^6 h
    n: int = 1
    k: int = 1
    mcmt_h: float = 1.0          # mean corrective maintenance time per item failure
    basis: str = "assumed"
    src: list[str] = field(default_factory=list)

    def reliability(self, hours: float) -> float:
        return k_of_n(r_exp(self.fpmh, hours), self.k, self.n)

    @property
    def lam_total(self) -> float:
        """Failure rate of the block's items (all items, maintenance demand), per hour."""
        return self.fpmh * 1e-6 * self.n


def series_reliability(blocks: list[Block], hours: float) -> float:
    r = 1.0
    for b in blocks:
        r *= b.reliability(hours)
    return r


def availability(blocks: list[Block], ldt_h: float, adt_h: float) -> dict:
    """Inherent and operational availability of the power system as a maintained series system.

    Every item failure is treated as a corrective maintenance action (redundant items still have
    to be repaired); preventive maintenance is not modelled.
    """
    lam = sum(b.lam_total for b in blocks)
    mtbf = 1.0 / lam
    mcmt = sum(b.lam_total * b.mcmt_h for b in blocks) / lam       # Eq. 34.48
    a_i = mtbf / (mtbf + mcmt)
    mdt = mcmt + ldt_h + adt_h
    a_o = mtbf / (mtbf + mdt)
    return {"mtbf_h": mtbf, "mcmt_h": mcmt, "A_i": a_i, "mdt_h": mdt, "A_o": a_o}


def importance(blocks: list[Block], hours: float) -> list[tuple[str, float]]:
    """Share of mission unreliability per block (1 - R_block over sum), largest first."""
    q = [(b.id, 1 - b.reliability(hours)) for b in blocks]
    tot = sum(x for _, x in q) or 1.0
    return sorted(((i, x / tot) for i, x in q), key=lambda t: -t[1])


def block_from_dict(d: dict) -> Block:
    return Block(**{k: v for k, v in d.items() if k in Block.__dataclass_fields__})

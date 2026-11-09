"""Analysis of alternatives scoring (DOSSIER section 11, Wasson ch. 32).

Method (Wasson 12-step method, Fig. 32.7, with the auto-normalised weighting of section 32.5.12):

1. screen: an alternative that misses a must-meet threshold (KPP threshold) is recorded and
   eliminated before scoring (Fig. 32.4 trade space);
2. utility: each criterion value is mapped to 0..1 by its utility function (linear increasing,
   linear decreasing, or S-curve between a threshold and an objective; Fig. 32.9);
3. figure of merit: FOM = sum_c w_c * u_c with weights from 1..10 importance ratings,
   w_c = rating_c / sum ratings;
4. sensitivity: each weight in turn is cut by 10 % and the rest renormalised (section 32.7); the
   winner must not change. A Monte Carlo over the ratings (each rating x U(0.5, 1.5)) gives how
   often each alternative ranks first (the author's addition; Wasson asks only for the
   one-at-a-time check).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Criterion:
    id: str
    name: str
    unit: str
    rating: float              # importance 1..10
    shape: str                 # "inc" / "dec" / "s" (S-curve, increasing)
    worst: float               # utility 0
    best: float                # utility 1
    threshold: float | None = None   # must-meet value (screening), same direction as best

    def utility(self, x: float) -> float:
        lo, hi = self.worst, self.best
        f = (x - lo) / (hi - lo)
        f = min(max(f, 0.0), 1.0)
        if self.shape == "s":
            # logistic on 0..1, scaled so that u(0) = 0 and u(1) = 1
            g = lambda z: 1.0 / (1.0 + math.exp(-10.0 * (z - 0.5)))  # noqa: E731
            return (g(f) - g(0.0)) / (g(1.0) - g(0.0))
        return f

    def meets(self, x: float) -> bool:
        if self.threshold is None:
            return True
        return x >= self.threshold if self.best > self.worst else x <= self.threshold


def weights(criteria: list[Criterion], ratings: list[float] | None = None) -> np.ndarray:
    r = np.array(ratings if ratings is not None else [c.rating for c in criteria], dtype=float)
    return r / r.sum()


def score(criteria: list[Criterion], table: dict[str, dict[str, float]], ratings=None) -> dict:
    """table: alternative id -> criterion id -> value. Returns screening, utilities and FOMs."""
    w = weights(criteria, ratings)
    out = {"screened_out": {}, "utility": {}, "fom": {}}
    for alt, vals in table.items():
        miss = [c.id for c in criteria if not c.meets(vals[c.id])]
        if miss:
            out["screened_out"][alt] = miss
            continue
        u = np.array([c.utility(vals[c.id]) for c in criteria])
        out["utility"][alt] = dict(zip([c.id for c in criteria], u.round(4).tolist()))
        out["fom"][alt] = float(w @ u)
    out["ranking"] = sorted(out["fom"], key=lambda a: -out["fom"][a])
    return out


def sensitivity_10pct(criteria: list[Criterion], table: dict) -> list[dict]:
    base = score(criteria, table)
    rows = []
    for i, c in enumerate(criteria):
        r = [x.rating for x in criteria]
        r[i] *= 0.9
        s = score(criteria, table, r)
        rows.append({"criterion": c.id, "winner": s["ranking"][0] if s["ranking"] else None,
                     "same_winner": bool(s["ranking"]) and s["ranking"][0] == base["ranking"][0],
                     "fom": s["fom"]})
    return rows


def monte_carlo_weights(criteria: list[Criterion], table: dict, n: int = 2000, seed: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    wins: dict[str, int] = {}
    base = [c.rating for c in criteria]
    for _ in range(n):
        r = [b * rng.uniform(0.5, 1.5) for b in base]
        s = score(criteria, table, r)
        if s["ranking"]:
            wins[s["ranking"][0]] = wins.get(s["ranking"][0], 0) + 1
    return {k: v / n for k, v in sorted(wins.items(), key=lambda t: -t[1])}


def criterion_from_dict(d: dict) -> Criterion:
    return Criterion(**{k: v for k, v in d.items() if k in Criterion.__dataclass_fields__})

"""Mission-compartment / electronics-bay thermal budget (DOSSIER section 8.6).

One air zone (the mission compartment that holds the equipment racks and the operators) behind an
armoured, lined shell. Heat paths:

* equipment: electrical input minus radiated RF of every in-bay load (loads.Load.heat_avg);
* crew: sensible metabolic heat per seated operator;
* roof: solar gain through the sol-air temperature  T_sa = T_amb + alpha * G / h_o;
* walls and floor: conduction from the ambient air;
* ventilation: the CBRN unit's filtered fresh air, UA_vent = rho * cp * flow (it runs in every
  mode to keep the overpressure, so it cools at night and heats on a hot day);
* cooling: the environmental control unit (ECU) removes Q_c with a coefficient of performance
  COP(T_amb) and draws P_e = Q_c / COP from the 28 V bus (plus its fans).

Steady state:   Q_equip + Q_crew + UA_roof (T_sa - T_z) + (UA_wall + UA_vent)(T_amb - T_z) - Q_c = 0
Transient:      C_z dT_z/dt = (same left-hand side)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Zone:
    ua_roof: float            # W/K
    ua_wall: float            # W/K
    ua_vent: float            # W/K, fresh-air ventilation
    alpha_roof: float         # solar absorptivity of the roof paint
    h_out: float              # W/m2K outside film coefficient
    c_zone_j_per_k: float     # air + racks + equipment effective heat capacity
    crew: int
    crew_sensible_w: float    # W per seated operator
    t_limit_c: float          # zone air limit for the equipment inlet
    ecu_q_rated_w: float      # ECU rated cooling at the rating point
    ecu_cop: list[tuple[float, float]]   # (T_amb degC, COP)
    ecu_fan_w: float          # fans, always on when the ECU runs

    def t_solair(self, t_amb: float, g_solar: float) -> float:
        return t_amb + self.alpha_roof * g_solar / self.h_out

    def gains_at(self, t_z: float, t_amb: float, g_solar: float, q_equip: float) -> float:
        return (q_equip + self.crew * self.crew_sensible_w
                + self.ua_roof * (self.t_solair(t_amb, g_solar) - t_z)
                + (self.ua_wall + self.ua_vent) * (t_amb - t_z))

    def t_uncooled(self, t_amb: float, g_solar: float, q_equip: float) -> float:
        """Steady-state zone temperature with no cooling."""
        ua = self.ua_total
        q_fixed = q_equip + self.crew * self.crew_sensible_w
        return (q_fixed + self.ua_roof * self.t_solair(t_amb, g_solar) + (self.ua_wall + self.ua_vent) * t_amb) / ua

    @property
    def ua_total(self) -> float:
        return self.ua_roof + self.ua_wall + self.ua_vent

    def cooling_needed(self, t_amb: float, g_solar: float, q_equip: float) -> float:
        """Q_c (W) to hold the zone at t_limit_c (0 if it stays below without cooling)."""
        return max(0.0, self.gains_at(self.t_limit_c, t_amb, g_solar, q_equip))

    def cop(self, t_amb: float) -> float:
        t, c = zip(*self.ecu_cop)
        return float(np.interp(t_amb, t, c))

    def ecu_electric_w(self, t_amb: float, g_solar: float, q_equip: float) -> tuple[float, float]:
        """(electrical draw W, cooling delivered W). Capacity-limited at ecu_q_rated_w."""
        q = self.cooling_needed(t_amb, g_solar, q_equip)
        if q <= 0:
            return 0.0, 0.0
        q_del = min(q, self.ecu_q_rated_w)
        return q_del / self.cop(t_amb) + self.ecu_fan_w, q_del

    def time_to_limit(self, t0: float, t_amb: float, g_solar: float, q_equip: float,
                      q_cool: float = 0.0) -> float:
        """Hours for the zone to rise from t0 to t_limit_c (inf if it never does)."""
        ua = self.ua_total
        t_inf = self.t_uncooled(t_amb, g_solar, q_equip) - q_cool / ua
        if t_inf <= self.t_limit_c or t0 >= self.t_limit_c:
            return math.inf if t_inf <= self.t_limit_c else 0.0
        tau = self.c_zone_j_per_k / ua
        return tau * math.log((t_inf - t0) / (t_inf - self.t_limit_c)) / 3600.0

    def step(self, t_z: float, dt: float, t_amb: float, g_solar: float, q_equip: float, q_cool: float) -> float:
        ua = self.ua_total
        t_inf = self.t_uncooled(t_amb, g_solar, q_equip) - q_cool / ua
        return t_inf + (t_z - t_inf) * math.exp(-ua * dt / self.c_zone_j_per_k)


def zone_from_dict(d: dict) -> Zone:
    kw = {k: v for k, v in d.items() if k in Zone.__dataclass_fields__}
    kw["ecu_cop"] = [tuple(x) for x in d["ecu_cop"]]
    return Zone(**kw)

"""Battery equivalent-circuit model (DOSSIER section 8.2, M&S plan section 10).

One *block* is the battery as bought (a 12 V lead-acid 6T block or a 24 V lithium-ion 6T block).
A *pack* is Ns blocks in series times Np strings in parallel; all blocks in a pack are assumed equal.

Model per block (first-order Thevenin):

    V_term = OCV(SOC) - I * R0(T, SOC) - V_rc          I > 0 = discharge
    dV_rc/dt = I/C1 - V_rc/(R1*C1)                     tau1 = R1*C1

* OCV(SOC): piecewise-linear table from the data file.
* R0(T): table of (T, R0) points, interpolated as ln R0 linear in 1/T (Arrhenius form between
  points, extrapolated with the end segments). The cold points are *derived* from the specification
  cranking ratings: a block that must deliver I_crank for 30 s at T with V >= V_end has a total
  resistance of at most (OCV - V_end) / I_crank. A model built on these bounds is a battery that
  just meets its specification: the conservative design case.
* Capacity: rated capacity at the reference rate, scaled by a temperature factor table and, for
  discharge, by Peukert's law through an effective current  I_eff = I * (I / I_ref)^(k - 1).
  k is derived from two published capacity ratings (C20 and the 25 A reserve capacity).
* Charging: coulombic efficiency eta_c; a charge-current limit; lithium-ion blocks accept no charge
  below their minimum charge temperature (their heater must warm them first).
* Thermal: one lumped mass per block, m*cp*dT/dt = I^2*R0 + P_heater - UA*(T - T_amb).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

T0_K = 273.15


@dataclass
class BlockParams:
    name: str
    chem: str                       # "pb_agm" or "li_ion"
    v_nom: float                    # V
    c_rated_ah: float               # Ah at i_ref_a, 25 degC
    i_ref_a: float                  # A, reference rate of c_rated_ah
    peukert_k: float
    r0_table: list[tuple[float, float]]   # (degC, ohm per block), ascending temperature
    r1: float                       # ohm
    tau1_s: float
    ocv_soc: list[tuple[float, float]]          # (SOC 0..1, V per block)
    cap_temp: list[tuple[float, float]]         # (degC, capacity factor)
    v_cutoff: float                 # V per block under load: end of discharge
    soc_min: float                  # SOC floor used by the battery owner (BMS cut-off or 0)
    dod_floor: float                # lowest SOC the controller lets the mission bank reach (life / spec)
    mass_kg: float
    volume_l: float
    eta_charge: float
    i_charge_max_a: float           # A per block
    t_charge_min_c: float           # degC; below this no charge accepted
    heater_w: float = 0.0           # W per block, drawn from the block itself
    heater_on_c: float = -100.0     # heater runs while block temperature < this
    heater_off_c: float = -100.0
    cp_j_per_kgk: float = 1000.0
    ua_w_per_k: float = 1.5
    r_soc_factor: float = 0.5       # R0 multiplier at SOC 0 is (1 + r_soc_factor)
    i_max_a: float = 1e9            # max discharge current per block (BMS limit)
    cost_eur: float = 0.0
    src: list[str] = field(default_factory=list)

    def _r0_cache(self):
        key = tuple(self.r0_table)
        c = self.__dict__.get("_r0c")
        if c is None or c[0] != key:
            pts = sorted(self.r0_table)
            c = (key, [1.0 / (t + T0_K) for t, _ in pts], [math.log(r) for _, r in pts])
            self.__dict__["_r0c"] = c
        return c[1], c[2]

    def r0_base(self, t_c: float) -> float:
        xs, ys = self._r0_cache()                         # xs descending as T ascends
        x = 1.0 / (t_c + T0_K)
        if len(xs) == 1:
            return math.exp(ys[0])
        # find the segment (xs is decreasing)
        j = 0
        while j < len(xs) - 2 and x < xs[j + 1]:
            j += 1
        x0, x1, y0, y1 = xs[j], xs[j + 1], ys[j], ys[j + 1]
        return math.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))

    def r0(self, t_c: float, soc: float) -> float:
        return self.r0_base(t_c) * (1.0 + self.r_soc_factor * (1.0 - min(max(soc, 0.0), 1.0)) ** 2)

    def ocv(self, soc: float) -> float:
        s, v = zip(*self.ocv_soc)
        return float(np.interp(min(max(soc, 0.0), 1.0), s, v))

    def cap_factor(self, t_c: float) -> float:
        t, f = zip(*self.cap_temp)
        return float(np.interp(t_c, t, f))

    def capacity_ah(self, t_c: float) -> float:
        return self.c_rated_ah * self.cap_factor(t_c)

    def peukert_factor(self, i_block: float) -> float:
        """I_eff / I for a discharge current per block (>= 1 above the reference rate)."""
        if i_block <= self.i_ref_a:
            return 1.0
        return (i_block / self.i_ref_a) ** (self.peukert_k - 1.0)

    def deliverable_ah(self, i_block: float, t_c: float = 25.0) -> float:
        """Capacity delivered at a constant discharge current (Peukert, temperature)."""
        return self.capacity_ah(t_c) / self.peukert_factor(i_block)


@dataclass
class Pack:
    p: BlockParams
    ns: int
    np_: int
    soc: float = 1.0
    t_c: float = 25.0
    v_rc: float = 0.0               # per block
    heater_on: bool = False
    ah_out: float = 0.0             # Ah delivered by the pack (bookkeeping)
    wh_out: float = 0.0

    # --- pack-level electrical quantities -------------------------------------------------
    @property
    def blocks(self) -> int:
        return self.ns * self.np_

    @property
    def mass_kg(self) -> float:
        return self.blocks * self.p.mass_kg

    @property
    def volume_l(self) -> float:
        return self.blocks * self.p.volume_l

    @property
    def nominal_wh(self) -> float:
        return self.blocks * self.p.v_nom * self.p.c_rated_ah

    def emf(self) -> float:
        """Open-circuit voltage minus the RC polarisation, pack level."""
        return self.ns * (self.p.ocv(self.soc) - self.v_rc)

    def r_int(self) -> float:
        return self.ns * self.p.r0(self.t_c, self.soc) / self.np_

    def charge_allowed(self) -> bool:
        return self.t_c >= self.p.t_charge_min_c and self.soc < 0.9999

    def current_at(self, v_bus: float) -> float:
        """Pack current (A, + = discharge) if its terminals sit at v_bus."""
        i = (self.emf() - v_bus) / self.r_int()
        i_ch_max = self.np_ * self.p.i_charge_max_a if self.charge_allowed() else 0.0
        return min(max(i, -i_ch_max), self.np_ * self.p.i_max_a)

    def terminal_v(self, i: float) -> float:
        return self.emf() - i * self.r_int()

    def heater_power(self) -> float:
        """Heater draw of the whole pack (W) for the present state (hysteresis updated in step)."""
        return self.blocks * self.p.heater_w if self.heater_on else 0.0

    def at_cutoff(self, i: float) -> bool:
        v_block = self.terminal_v(i) / self.ns
        return self.soc <= self.p.soc_min + 1e-9 or (i > 0 and v_block <= self.p.v_cutoff)

    # --- time step ---------------------------------------------------------------------------
    def step(self, i: float, dt: float, t_amb: float) -> None:
        """Advance by dt seconds with pack current i (A, + = discharge, heater draw included)."""
        pb = self.p
        i_blk = i / self.np_
        cap_as = pb.capacity_ah(self.t_c) * 3600.0
        if i_blk > 0:
            i_eff = i_blk * pb.peukert_factor(i_blk)
        else:
            i_eff = i_blk * pb.eta_charge
        self.soc = min(1.0, max(0.0, self.soc - i_eff * dt / cap_as))
        a = math.exp(-dt / pb.tau1_s)
        self.v_rc = self.v_rc * a + i_blk * pb.r1 * (1.0 - a)
        # thermal
        q = i_blk ** 2 * pb.r0(self.t_c, self.soc) + (pb.heater_w if self.heater_on else 0.0)
        m_cp = pb.mass_kg * pb.cp_j_per_kgk
        t_inf = t_amb + q / pb.ua_w_per_k
        self.t_c = t_inf + (self.t_c - t_inf) * math.exp(-pb.ua_w_per_k * dt / m_cp)
        if pb.heater_w > 0:
            if self.t_c < pb.heater_on_c:
                self.heater_on = True
            elif self.t_c > pb.heater_off_c:
                self.heater_on = False
        self.ah_out += i * dt / 3600.0
        self.wh_out += i * self.terminal_v(i) * dt / 3600.0


def solve_constant_power(emf: float, r: float, p: float) -> float | None:
    """Current drawn by a constant-power load p from a source emf behind r. None = voltage collapse."""
    if p <= 0:
        return 0.0
    disc = emf * emf - 4.0 * r * p
    if disc < 0:
        return None
    return (emf - math.sqrt(disc)) / (2.0 * r)


def block_from_dict(d: dict) -> BlockParams:
    keys = BlockParams.__dataclass_fields__.keys()
    kw = {k: v for k, v in d.items() if k in keys}
    kw["ocv_soc"] = [tuple(x) for x in d["ocv_soc"]]
    kw["cap_temp"] = [tuple(x) for x in d["cap_temp"]]
    kw["r0_table"] = sorted(tuple(x) for x in d["r0_table"])
    return BlockParams(**kw)


def derive_peukert_k(c_a: float, i_a: float, c_b: float, i_b: float) -> float:
    """Peukert exponent from two capacity ratings (Ah at current A)."""
    return 1.0 + math.log(c_a / c_b) / math.log(i_b / i_a)


def derive_r_from_cca(ocv_block: float, cca_a: float, v_end: float = 7.2) -> float:
    """Upper bound on total block resistance from the CCA rating (SAE J537 style end voltage)."""
    return (ocv_block - v_end) / cca_a


def crank_test(pack: Pack, i_crank: float, t_crank_s: float, attempts: int, rest_s: float,
               t_amb: float, v_min: float, dt: float = 0.1) -> tuple[bool, float]:
    """Run crank attempts on a copy-like pack (mutates it). Returns (passed, lowest terminal V)."""
    vmin_seen = 1e9
    for k in range(attempts):
        t = 0.0
        while t < t_crank_s:
            v = pack.terminal_v(i_crank)
            if pack.soc <= pack.p.soc_min + 1e-6:
                v = 0.0                    # empty: the bank cannot deliver the crank current
            vmin_seen = min(vmin_seen, v)
            pack.step(i_crank, dt, t_amb)
            t += dt
        if k < attempts - 1:
            t = 0.0
            while t < rest_s:
                pack.step(0.0, 1.0, t_amb)
                t += 1.0
    return vmin_seen >= v_min, vmin_seen

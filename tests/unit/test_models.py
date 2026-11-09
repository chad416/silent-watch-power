"""Unit tests of the physics models and the data set."""
import math

import numpy as np
import pytest

from swpower import aoa, data, rma
from swpower.analysis import crank_reserve_soc, validate_against_product, validate_battery_models
from swpower.battery import Pack, derive_peukert_k, solve_constant_power
from swpower.bus import solve_bus
from swpower.loads import validate
from swpower.sspc import SSPCRating, Wire, coordination
from swpower.transient import FrontEnd, holdup_capacitance, simulate, waveform_from_dict


# --- data -------------------------------------------------------------------------------------------
@pytest.mark.req("MS-03")
def test_every_source_key_resolves():
    assert data.check_sources() == []


@pytest.mark.req("MS-03")
def test_load_inventory_is_consistent():
    assert validate(data.budget().loads) == []


@pytest.mark.req("SR-02")
def test_budget_layers_add_up():
    b = data.budget()
    m = b.by_mode("silent_watch")
    assert m["cbe_w"] < m["with_growth_w"] < m["allocated_w"]
    assert m["allocated_w"] == pytest.approx(m["with_growth_w"] * (1 + b.system_margin))
    assert m["coincident_peak_w"] > m["cbe_w"]


# --- battery ------------------------------------------------------------------------------------------
@pytest.mark.req("MS-01")
def test_peukert_exponent_from_the_two_spec_ratings():
    k = derive_peukert_k(120.0, 6.0, 25.0 * 230 / 60, 25.0)
    assert k == pytest.approx(data.blocks()["agm_6t"].peukert_k, abs=1e-3)


@pytest.mark.req("MS-01")
def test_battery_models_meet_their_specification_ratings():
    for r in validate_battery_models():
        assert r["pass"], r


@pytest.mark.req("MS-02")
def test_spec_minimum_model_tracks_a_qualified_product_in_the_watch_range():
    for r in validate_against_product():
        if 30 <= r["product_min"] <= 1200:      # silent-watch discharge rates
            assert 0.93 <= r["ratio"] <= 1.08, r


@pytest.mark.req("MS-01")
def test_cold_reduces_capacity_and_raises_resistance():
    b = data.blocks()["agm_6t"]
    assert b.capacity_ah(-32) < b.capacity_ah(0) < b.capacity_ah(25)
    assert b.r0(-40, 1) > b.r0(-18, 1) > b.r0(25, 1)


def test_constant_power_solution_and_collapse():
    i = solve_constant_power(26.0, 0.01, 1000.0)
    assert (26.0 - 0.01 * i) * i == pytest.approx(1000.0)
    assert solve_constant_power(26.0, 0.01, 26.0 ** 2 / 0.04 + 1) is None


def test_li_ion_accepts_no_charge_below_zero():
    b = data.blocks()["li_6t"]
    p = Pack(b, 1, 1, soc=0.5, t_c=-5.0)
    assert p.current_at(29.0) == 0.0
    p.t_c = 5.0
    assert p.current_at(29.0) < 0


def test_li_ion_heater_holds_the_band():
    b = data.blocks()["li_6t"]
    p = Pack(b, 1, 1, soc=1.0, t_c=-32.0)
    for _ in range(3000):
        p.step(10.0, 2.0, -32.0)
    assert b.heater_on_c - 2 <= p.t_c <= b.heater_off_c + 2


# --- bus ------------------------------------------------------------------------------------------------
def test_bus_with_source_holds_set_point_and_charges():
    p = Pack(data.blocks()["agm_6t"], 2, 2, soc=0.6, t_c=25.0)
    r = solve_bus([p], 1500.0, v_set=28.0, i_src_max=300.0)
    assert r.v == pytest.approx(28.0, abs=0.05) and r.i_packs[0] < 0


def test_bus_source_current_limited_battery_supplies_the_rest():
    p = Pack(data.blocks()["agm_6t"], 2, 2, soc=0.9, t_c=25.0)
    r = solve_bus([p], 5000.0, v_set=28.0, i_src_max=50.0)
    assert r.i_src == pytest.approx(50.0) and r.i_packs[0] > 0 and r.v < 28.0


def test_bus_finds_the_high_root_with_pack_current_limits():
    p = Pack(data.blocks()["li_6t"], 1, 2, soc=0.9, t_c=20.0)
    r = solve_bus([p], 4000.0)
    assert not r.collapsed and r.v > 24.0


# --- engine-start reserve --------------------------------------------------------------------------------
@pytest.mark.req("SR-20")
def test_two_block_start_bank_cannot_crank_at_minus_32():
    assert math.isnan(crank_reserve_soc("agm_6t", 2, 1, -32.0))


@pytest.mark.req("SR-20", "SR-21")
def test_four_block_start_bank_cranks_at_minus_32_with_a_reserve():
    r = crank_reserve_soc("agm_6t", 2, 2, -32.0)
    assert 0.0 < r < 0.3


@pytest.mark.req("SR-22")
def test_no_practical_agm_bank_cranks_at_minus_46_from_partial_charge():
    assert math.isnan(crank_reserve_soc("agm_6t", 2, 2, -46.0))


# --- thermal ----------------------------------------------------------------------------------------------
@pytest.mark.req("SR-40")
def test_cooling_unit_covers_the_hot_design_point():
    z = data.zone()
    from swpower.analysis import equipment_heat
    q = equipment_heat(data.budget(), "silent_watch")
    c = data.climates()["A1_hot_dry"]
    assert z.cooling_needed(c["t_amb"], c["g_solar"], q) < z.ecu_q_rated_w
    assert z.t_uncooled(c["t_amb"], c["g_solar"], q) > z.t_limit_c


def test_zone_steady_state_balances():
    z = data.zone()
    t = z.t_uncooled(30.0, 500.0, 1000.0)
    assert z.gains_at(t, 30.0, 500.0, 1000.0) == pytest.approx(0.0, abs=1e-6)


# --- SSPC / wire --------------------------------------------------------------------------------------------
@pytest.mark.req("SR-30")
def test_sspc_curve_shape():
    r = SSPCRating(10.0, 1.15, 2.0, 3.0, 6.0, 2.5e-4)
    assert math.isinf(r.trip_time(11.0))
    assert r.trip_time(20.0) == pytest.approx(3.0)
    assert r.trip_time(15.0) > r.trip_time(20.0) > r.trip_time(50.0)
    assert r.trip_time(80.0) == pytest.approx(2.5e-4)


@pytest.mark.req("SR-30")
def test_coordination_detects_an_undersized_wire():
    r = SSPCRating(10.0, 1.15, 2.0, 3.0, 6.0, 2.5e-4)
    thin = Wire(0.75, 9.6, 143.0)
    ok = Wire(1.5, 15.6, 143.0)
    assert not coordination(r, thin, np.logspace(0, 1.5, 40))["coordinated"]
    assert coordination(r, ok, np.logspace(0, 1.5, 40))["coordinated"]


# --- transients (short cases; the long ones run in the integration tests) -------------------------------------
@pytest.mark.req("SR-12")
def test_spike_is_clamped_below_the_fet_rating():
    import yaml
    d = yaml.safe_load(open(data.DATA / "transients.yaml", encoding="utf-8"))
    r = simulate(waveform_from_dict("spike_pos", d["cases"]["spike_pos"]), FrontEnd(v_br=122.0, r_dyn=0.05))
    assert r.va.max() < 0.8 * 200.0 and r.e_tvs < 15.0 and r.dropouts == 0


@pytest.mark.req("SR-12")
def test_1275f_spike_is_energy_limited():
    import yaml
    d = yaml.safe_load(open(data.DATA / "transients.yaml", encoding="utf-8"))
    r = simulate(waveform_from_dict("f_spike_pos", d["cases"]["f_spike_pos"]), FrontEnd(v_br=122.0, r_dyn=0.05))
    assert r.e_tvs < 0.5 * 8.2e-6 * 250.0 ** 2      # cannot exceed the energy stored in the 8.2 uF capacitor
    assert r.dropouts == 0


def test_holdup_formula():
    c = holdup_capacitance(100.0, 0.01, 28.0, 9.0)
    assert 0.5 * c * (28.0 ** 2 - 9.0 ** 2) == pytest.approx(100.0 * 0.01)


# --- trade study and RMA maths -----------------------------------------------------------------------------
def test_utility_shapes():
    s = aoa.Criterion("x", "x", "h", 5, "s", 0, 12, 4)
    assert s.utility(0) == pytest.approx(0) and s.utility(12) == pytest.approx(1) and s.utility(6) == pytest.approx(0.5)
    d = aoa.Criterion("m", "m", "kg", 5, "dec", 1000, 0)
    assert d.utility(250) == pytest.approx(0.75)
    assert s.meets(4.0) and not s.meets(3.9)


def test_screening_and_ranking():
    crit = [aoa.Criterion("a", "a", "", 5, "inc", 0, 10, 2), aoa.Criterion("b", "b", "", 5, "dec", 10, 0)]
    sc = aoa.score(crit, {"X": {"a": 8, "b": 5}, "Y": {"a": 1, "b": 0}, "Z": {"a": 5, "b": 1}})
    assert "Y" in sc["screened_out"] and sc["ranking"][0] == "Z"


def test_reliability_formulas():
    assert rma.k_of_n(0.9, 1, 2) == pytest.approx(0.99)
    assert rma.k_of_n(0.9, 2, 2) == pytest.approx(0.81)
    b = [rma.Block("a", "a", 100.0), rma.Block("b", "b", 50.0)]
    assert rma.series_reliability(b, 1000.0) == pytest.approx(math.exp(-0.15))
    av = rma.availability(b, 10.0, 2.0)
    assert av["A_i"] > av["A_o"] and av["mtbf_h"] == pytest.approx(1e6 / 150.0)

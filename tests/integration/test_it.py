"""Integration tests IT01-IT06 (DOSSIER 13.2): the full model against the requirements."""
import pytest
import yaml

from swpower import data
from swpower.analysis import endurance
from swpower.studies import FRONT_ENDS, cfg, rma_study, sspc_study, transient_study


@pytest.mark.it("IT01")
@pytest.mark.req("SR-01")
def test_recommended_bank_meets_the_threshold_in_both_design_corners():
    c = cfg()
    spec = c["alternatives"]["A2"]
    for climate in c["kpp"]["design_climates"]:
        r = endurance(spec["arch"], climate, spec["np"], "", hours=12, dt=30)
        assert r["t_essential_h"] >= c["kpp"]["threshold_h"], (climate, r["t_essential_h"])
        assert r["crank_ok"], climate


@pytest.mark.it("IT01")
@pytest.mark.req("SR-01")
def test_baseline_bank_misses_the_threshold():
    r = endurance("A0_common_agm", "A2_basic_hot", None, "", hours=6, dt=30)
    assert r["t_essential_h"] < 1.0


@pytest.mark.it("IT02")
@pytest.mark.req("SR-01", "SR-37")
def test_apu_variant_holds_the_essential_suite_for_twelve_hours_when_allowed():
    spec = cfg()["alternatives"]["A5"]
    r = endurance(spec["arch"], "A2_basic_hot", spec["np"], spec["apu"], hours=12, dt=30, silent_req=False)
    assert r["t_essential_h"] > 12 and r["apu_h"] > 0 and r["fuel_l"] > 0


@pytest.mark.it("IT02")
@pytest.mark.req("SR-37")
def test_silent_demand_keeps_the_apu_off():
    spec = cfg()["alternatives"]["A5"]
    r = endurance(spec["arch"], "A2_basic_hot", spec["np"], spec["apu"], hours=8, dt=30, silent_req=True)
    assert r["apu_h"] == 0


@pytest.mark.it("IT03")
@pytest.mark.req("SR-21", "SR-33")
def test_common_bank_keeps_its_crank_reserve_at_minus_18():
    r = endurance("A0_common_agm", "C1_basic_cold", None, "", hours=3, dt=30,
                  climate_override={"t_amb": -18.0})
    assert r["crank_ok"]


@pytest.mark.it("IT04")
@pytest.mark.req("SR-10", "SR-11", "SR-12", "SR-13", "SR-14", "SR-15")
def test_final_front_end_passes_every_1275e_and_1275f_case_and_earlier_designs_do_not():
    res = transient_study(fronts=["v1_tvs36", "v3_tvs122_2fet", "v4_final"])
    assert all(x["pass"] for x in res["v4_final"].values()), {k: v["checks"] for k, v in res["v4_final"].items()}
    assert not res["v1_tvs36"]["surge_100v"]["checks"]["tvs_energy_ok"]
    assert not res["v3_tvs122_2fet"]["f_surge"]["pass"]          # the 1275F surge tail outlasts its fault timer


@pytest.mark.it("IT05")
@pytest.mark.req("SR-30")
def test_every_sspc_channel_is_coordinated_with_its_wire():
    s = sspc_study()
    assert s["worst_damage_to_trip"] >= 1.0
    covered = [f for f in s["fault_cases"] if f["covered"]]
    assert len(covered) >= 6


@pytest.mark.it("IT06")
@pytest.mark.req("SR-60")
def test_reliability_targets_of_the_recommended_growth_path():
    r = rma_study("A5")
    assert r["rbd"]["k = n - 1 (one spare block)"]["R_12h"] >= 0.98
    assert r["availability"]["A_o"] >= 0.95


def test_front_end_table_is_complete():
    d = yaml.safe_load(open(data.DATA / "transients.yaml", encoding="utf-8"))
    assert set(d["cases"]) >= {"surge_100v", "starting", "spike_pos", "spike_neg", "reverse_polarity", "ripple"}
    assert set(d["cases"]) >= {"f_surge", "f_spike_pos", "f_ripple", "f_starting", "li_start_14v4"}
    assert "v4_final" in FRONT_ENDS


@pytest.mark.it("IT07")
@pytest.mark.req("SR-41", "SR-42", "SR-43")
def test_arc_detector_on_held_out_cases():
    from swpower.studies import arc_study
    r = arc_study(n_each=8, seed=2000)["classes"]
    assert r["series"]["tripped"] == r["series"]["n"] and r["series"]["max_ms"] <= 100
    assert r["parallel"]["tripped"] == r["parallel"]["n"] and r["parallel"]["max_ms"] <= 500
    assert all(v["tripped"] == 0 for k, v in r.items() if k.startswith("nuisance"))


@pytest.mark.it("IT08")
@pytest.mark.req("SR-23")
def test_cranking_conflict_and_its_resolution():
    from swpower.studies import conflict_study
    c = conflict_study()
    rows = {r["bank"]: r for r in c["rows"]}
    cold = [k for k in c["cases"] if "degC" in k and not k.startswith("Fig")]
    assert all(rows["Li 1s1p"][k] < 16.0 for k in cold)          # a single spec-minimum Li 6T: below 16 V
    assert all(rows["AGM 2s2p"][k] >= 16.0 for k in c["cases"])   # the chosen start bank holds 16 V
    assert all(rows["Li 1s2p"][k] >= 16.0 for k in c["cases"])

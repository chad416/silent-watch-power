"""FB_PowerMgmt (plc/FB_PowerMgmt.st) run through the ST translator: load-shed state machine."""
import numpy as np
import pytest

from swpower.sim import controller

PRIO = [1, 1, 1, 2, 2, 3, 3, 4, 4, 2, 0, 0, 0, 0, 0, 0]


def make(soc=0.9, v=26.0, charge=False, **kw):
    fb = controller()
    fb.Prio = PRIO
    fb.ChReq = [p > 0 for p in PRIO]
    fb.CycleMs = 1000
    fb.SocMB = soc
    fb.VBus = v
    fb.ChargeOK = charge
    fb.TBay = 30.0
    fb.PLoadW = 1500.0
    fb.SocReserve = 0.10
    fb.UsableWh = 6000.0
    for k, val in kw.items():
        setattr(fb, k, val)
    return fb


def scan(fb, seconds, **kw):
    for k, v in kw.items():
        setattr(fb, k, v)
    for _ in range(int(seconds)):
        fb.scan(1000)


def on_prios(fb):
    return sorted({PRIO[i] for i in range(16) if fb.ChOn[i]})


@pytest.mark.req("SR-31")
def test_shed_order_follows_priority():
    fb = make(soc=0.9)
    scan(fb, 2)
    assert on_prios(fb) == [1, 2, 3, 4]
    scan(fb, 12, SocMB=0.45)
    assert fb.ShedLevel == 1 and on_prios(fb) == [1, 2, 3]
    scan(fb, 12, SocMB=0.30)
    assert fb.ShedLevel == 2 and on_prios(fb) == [1, 2]
    scan(fb, 12, SocMB=0.15)
    assert fb.ShedLevel == 3 and on_prios(fb) == [1]


@pytest.mark.req("SR-31")
def test_shed_needs_10s_debounce_and_steps_one_level_at_a_time():
    fb = make(soc=0.9)
    scan(fb, 2)
    scan(fb, 9, SocMB=0.15)
    assert fb.ShedLevel == 0
    scan(fb, 2)
    assert fb.ShedLevel == 1
    scan(fb, 11)
    assert fb.ShedLevel == 2


@pytest.mark.req("SR-32")
def test_restore_needs_a_source_and_hysteresis():
    fb = make(soc=0.30)
    scan(fb, 30)
    assert fb.ShedLevel == 2
    scan(fb, 120, SocMB=0.40)             # above L2 threshold but within hysteresis, no source
    assert fb.ShedLevel == 2
    scan(fb, 120, SocMB=0.40, ChargeOK=True)  # source on, but 0.40 < 0.35 + 0.10
    assert fb.ShedLevel == 2
    scan(fb, 70, SocMB=0.52, ChargeOK=True)  # usable SOC (0.52-0.10)/0.90 = 0.47 > 0.35 + 0.10
    assert fb.ShedLevel == 1


@pytest.mark.req("SR-33")
def test_reserve_disconnects_everything_and_needs_charge_to_restore():
    fb = make(soc=0.15)
    scan(fb, 40)
    assert fb.ShedLevel == 3 and on_prios(fb) == [1]
    scan(fb, 2, SocMB=0.09)
    assert fb.LvdActive and fb.ReserveAlarm and not any(fb.ChOn)
    scan(fb, 5, SocMB=0.12)               # SOC estimate recovers a little at rest: stays off
    assert fb.LvdActive
    scan(fb, 5, SocMB=0.16, ChargeOK=True)
    assert not fb.LvdActive


@pytest.mark.req("SR-34")
def test_fast_undervoltage_shed_without_waiting_for_soc():
    fb = make(soc=0.8)
    scan(fb, 3)
    scan(fb, 6, VBus=22.5)
    assert fb.ShedLevel == 1
    scan(fb, 10)                          # one level per sag event only
    assert fb.ShedLevel == 1
    scan(fb, 2, VBus=24.0)
    scan(fb, 6, VBus=22.5)
    assert fb.ShedLevel == 2


@pytest.mark.req("SR-33")
def test_low_voltage_disconnect_after_10s_below_21v():
    fb = make(soc=0.6)
    scan(fb, 3)
    scan(fb, 9, VBus=20.5)
    assert not fb.LvdActive
    scan(fb, 3)
    assert fb.LvdActive and not any(fb.ChOn)


@pytest.mark.req("SR-35")
def test_cbrn_channel_held_while_alarm():
    fb = make(soc=0.15, CbrnCh=6)         # channel 6 is priority 3
    scan(fb, 40)
    assert not fb.ChOn[5]
    scan(fb, 1, CbrnAlarm=True)
    assert fb.ChOn[5]
    scan(fb, 1, CbrnAlarm=False)
    assert not fb.ChOn[5]


@pytest.mark.req("SR-36")
def test_vital_channel_gets_one_retry_then_latches():
    fb = make(soc=0.9)
    scan(fb, 2)
    trip = [False] * 16
    trip[0] = True
    scan(fb, 4, ChTrip=trip)
    assert not any(fb.ChResetCmd)
    fb.scan(1000)
    assert fb.ChResetCmd[0] and not fb.TripLatched[0]
    scan(fb, 1, ChTrip=[False] * 16)
    scan(fb, 1, ChTrip=trip)              # trips again: latched
    assert fb.TripLatched[0] and not fb.ChOn[0]
    scan(fb, 1, ChTrip=[False] * 16, Reset=True)
    scan(fb, 1, Reset=False)
    assert not fb.TripLatched[0] and fb.ChOn[0]


@pytest.mark.req("SR-36")
def test_non_vital_trip_latches_at_once():
    fb = make(soc=0.9)
    scan(fb, 2)
    trip = [False] * 16
    trip[5] = True
    scan(fb, 1, ChTrip=trip)
    assert fb.TripLatched[5] and not fb.ChOn[5]


@pytest.mark.req("SR-31")
def test_thresholds_follow_the_usable_band_above_the_reserve():
    fb = make(soc=0.70, SocReserve=0.62)    # common bank: usable SOC (0.70-0.62)/0.38 = 0.21
    scan(fb, 40)
    assert fb.ShedLevel == 3 and not fb.LvdActive


@pytest.mark.req("SR-37")
def test_apu_request_respects_silent_demand():
    fb = make(soc=0.30, AutoStartEn=True, SilentReq=True)
    scan(fb, 30)
    assert not fb.ApuStartReq
    scan(fb, 1, SilentReq=False)
    assert fb.ApuStartReq
    scan(fb, 1, SocMB=0.96, ChargeOK=True)
    assert not fb.ApuStartReq


@pytest.mark.req("SR-38")
def test_bay_overtemperature_sheds_deferrable_loads():
    fb = make(soc=0.9)
    scan(fb, 2)
    scan(fb, 1, TBay=56.0)
    assert fb.BayHot and fb.ShedLevel >= 2 and on_prios(fb) == [1, 2]


@pytest.mark.req("SR-39")
def test_combiner_closes_only_with_a_source_and_charging_voltage():
    fb = make(soc=0.9, VBus=27.8, ChargeOK=False)
    scan(fb, 10)
    assert not fb.CombinerCmd
    scan(fb, 6, ChargeOK=True)
    assert fb.CombinerCmd
    scan(fb, 3, ChargeOK=False, VBus=25.0)
    assert not fb.CombinerCmd


@pytest.mark.req("SR-31", "SR-33", "SR-36")
def test_random_input_fuzz_invariants():
    """IT03-style fuzz: invariants hold on every scan for random inputs."""
    rng = np.random.default_rng(7)
    fb = make()
    fb.CbrnCh = 6
    for _ in range(5000):
        fb.SocMB = float(rng.uniform(0, 1))
        fb.VBus = float(rng.uniform(18, 30))
        fb.ChargeOK = bool(rng.random() < 0.3)
        fb.CbrnAlarm = bool(rng.random() < 0.1)
        fb.TBay = float(rng.uniform(20, 60))
        fb.ChTrip = [bool(rng.random() < 0.02) for _ in range(16)]
        fb.Reset = bool(rng.random() < 0.01)
        fb.scan(1000)
        assert 0 <= fb.ShedLevel <= 3
        for i in range(16):
            if fb.ChOn[i]:
                assert not fb.LvdActive and not fb.TripLatched[i] and PRIO[i] > 0
                assert PRIO[i] <= 4 - fb.ShedLevel or (fb.CbrnAlarm and i == 5)
            if PRIO[i] == 0:
                assert not fb.ChOn[i]

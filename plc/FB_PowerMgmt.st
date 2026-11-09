(* FB_PowerMgmt: load management for a 28 V vehicle mission bus with a 16-channel SSPC panel.
   DOSSIER section 6 (modes and states), section 9 (load shedding), SR-30..SR-39.

   Shed levels (state within the SILENT WATCH / HALT modes):
     0  all requested channels on
     1  priority 4 (comfort) shed
     2  priority 3 (deferrable mission) shed
     3  priority 2 (essential) shed; only priority 1 (vital) left
   Mission-bank reserve reached (SocMB <= SocReserve) or low-voltage disconnect: all channels off.
   The CBRN channel is held like priority 1 while CbrnAlarm is TRUE.

   Inputs are engineering values from the battery monitor and the SSPC panel. Prio[], ChReq[] and
   CbrnCh / SocReserve / UsableWh are configuration set once at power-up. *)
FUNCTION_BLOCK FB_PowerMgmt
VAR_INPUT
    CycleMs : DINT := 1000;          (* scan time handed in by the caller, ms *)
    VBus : REAL;                     (* mission bus voltage, V *)
    SocMB : REAL;                    (* mission bank state of charge, 0..1 *)
    TBay : REAL;                     (* electronics bay air temperature, degC *)
    PLoadW : REAL;                   (* measured mission-bus load, W *)
    ChargeOK : BOOL;                 (* engine or APU running and its source on line *)
    CbrnAlarm : BOOL;
    AutoStartEn : BOOL;              (* crew allows the APU to start by itself *)
    SilentReq : BOOL;                (* crew demands silence: no automatic start *)
    Reset : BOOL;                    (* crew reset: clears latched trips and the LVD *)
    SocReserve : REAL := 0.10;       (* mission bank floor (architecture dependent) *)
    UsableWh : REAL := 2000.0;       (* mission bank energy between 100 % and 0 % SOC, Wh *)
    CbrnCh : INT := 0;               (* SSPC channel of the CBRN filtration unit, 0 = none *)
    Prio : ARRAY[1..16] OF INT;      (* 1 vital .. 4 comfort, 0 = channel unused *)
    ChReq : ARRAY[1..16] OF BOOL;    (* crew request per channel *)
    ChTrip : ARRAY[1..16] OF BOOL;   (* SSPC tripped *)
END_VAR
VAR_OUTPUT
    ChOn : ARRAY[1..16] OF BOOL;     (* SSPC on command *)
    ChResetCmd : ARRAY[1..16] OF BOOL;  (* one-scan SSPC trip reset *)
    ShedLevel : INT;
    LvdActive : BOOL;                (* mission bus disconnected to protect the bank *)
    CombinerCmd : BOOL;              (* close the start/mission bank combiner *)
    ApuStartReq : BOOL;
    LowSocWarn : BOOL;
    ReserveAlarm : BOOL;
    BayHot : BOOL;
    HoursLeft : REAL;                (* to the reserve at the present load, h *)
    TripLatched : ARRAY[1..16] OF BOOL;
END_VAR
VAR
    i : INT;
    lvl : INT;
    socU : REAL;                     (* usable SOC: 0 at the reserve, 1 when full *)
    target : INT;
    allowed : BOOL;
    tripAge : ARRAY[1..16] OF DINT;  (* ms since the trip was seen *)
    retryUsed : ARRAY[1..16] OF BOOL;
    tUV : TON;                       (* fast under-voltage shed *)
    tShedUp : TON;                   (* SOC-driven shed debounce *)
    tShedDn : TON;                   (* restore debounce *)
    tLvd : TON;
    tCombClose : TON;
    tCombOpen : TON;
    rReset : R_TRIG;
    uvFired : BOOL;
END_VAR
VAR CONSTANT
    SOC_L1 : REAL := 0.50;           (* shed comfort below 50 % of the usable band *)
    SOC_L2 : REAL := 0.35;
    SOC_L3 : REAL := 0.22;
    SOC_HYST : REAL := 0.10;         (* restore only 10 % above the shed threshold *)
    V_UV : REAL := 23.0;             (* fast shed when the bus sags below this ... *)
    V_LVD : REAL := 21.0;            (* ... and disconnect below this *)
    V_COMB_CLOSE : REAL := 26.4;     (* voltage-sensing combiner *)
    V_COMB_OPEN : REAL := 25.6;
    T_BAY_HOT : REAL := 55.0;        (* bay air limit: shed deferrable bay loads *)
    T_BAY_OK : REAL := 50.0;
    SOC_APU_STOP : REAL := 0.95;
    RETRY_MS : DINT := 5000;
END_VAR

rReset(CLK := Reset);

(* --- combiner (voltage-sensing relay behaviour, only while a source is on line) --- *)
tCombClose(IN := ChargeOK AND VBus >= V_COMB_CLOSE, PT := T#5S);
tCombOpen(IN := (NOT ChargeOK) OR VBus < V_COMB_OPEN, PT := T#2S);
IF tCombClose.Q THEN
    CombinerCmd := TRUE;
ELSIF tCombOpen.Q THEN
    CombinerCmd := FALSE;
END_IF;

(* --- SOC-driven target level, on the usable band above the reserve --- *)
socU := (SocMB - SocReserve) / MAX(1.0 - SocReserve, 0.05);
IF socU < SOC_L3 THEN
    target := 3;
ELSIF socU < SOC_L2 THEN
    target := 2;
ELSIF socU < SOC_L1 THEN
    target := 1;
ELSE
    target := 0;
END_IF;

(* raise the level after 10 s below a threshold *)
tShedUp(IN := target > ShedLevel, PT := T#10S);
IF tShedUp.Q THEN
    ShedLevel := ShedLevel + 1;
    tShedUp(IN := FALSE, PT := T#10S);   (* restart: one level per 10 s *)
END_IF;

(* restore one level only with a source on line and SOC above threshold + hysteresis *)
lvl := 0;
IF socU < SOC_L3 + SOC_HYST THEN
    lvl := 3;
ELSIF socU < SOC_L2 + SOC_HYST THEN
    lvl := 2;
ELSIF socU < SOC_L1 + SOC_HYST THEN
    lvl := 1;
END_IF;
tShedDn(IN := ChargeOK AND lvl < ShedLevel AND NOT BayHot, PT := T#60S);
IF tShedDn.Q THEN
    ShedLevel := ShedLevel - 1;
    tShedDn(IN := FALSE, PT := T#60S);
END_IF;

(* fast under-voltage shed: one level per 5 s of sag, without waiting for the SOC estimate *)
tUV(IN := VBus < V_UV AND NOT ChargeOK AND NOT uvFired, PT := T#5S);
IF tUV.Q THEN
    ShedLevel := MIN(ShedLevel + 1, 3);
    uvFired := TRUE;
END_IF;
IF VBus >= V_UV + 0.5 OR ChargeOK THEN
    uvFired := FALSE;
END_IF;

(* electronics bay over-temperature: drop deferrable loads to cut heat *)
IF TBay > T_BAY_HOT THEN
    BayHot := TRUE;
ELSIF TBay < T_BAY_OK THEN
    BayHot := FALSE;
END_IF;
IF BayHot AND ShedLevel < 2 THEN
    ShedLevel := 2;
END_IF;

IF ShedLevel > 3 THEN
    ShedLevel := 3;
ELSIF ShedLevel < 0 THEN
    ShedLevel := 0;
END_IF;

(* --- reserve and low-voltage disconnect --- *)
ReserveAlarm := SocMB <= SocReserve;
tLvd(IN := VBus < V_LVD AND NOT ChargeOK, PT := T#10S);
IF (ReserveAlarm OR tLvd.Q) AND NOT ChargeOK THEN
    LvdActive := TRUE;
END_IF;
IF LvdActive AND ChargeOK AND SocMB > SocReserve + 0.05 THEN
    LvdActive := FALSE;
END_IF;
IF rReset.Q AND NOT ReserveAlarm AND VBus >= V_LVD THEN
    LvdActive := FALSE;
END_IF;
LowSocWarn := socU < SOC_L2;

(* --- auxiliary power unit request (hysteresis on SOC) --- *)
IF AutoStartEn AND NOT SilentReq AND (ShedLevel >= 2 OR socU < SOC_L2) THEN
    ApuStartReq := TRUE;
END_IF;
IF SilentReq OR NOT AutoStartEn OR SocMB >= SOC_APU_STOP THEN
    ApuStartReq := FALSE;
END_IF;

(* --- time to the reserve at the present load --- *)
IF PLoadW > 1.0 THEN
    HoursLeft := MAX(SocMB - SocReserve, 0.0) * UsableWh / PLoadW;
ELSE
    HoursLeft := 99.0;
END_IF;

(* --- per channel: trip handling and the on command --- *)
FOR i := 1 TO 16 DO
    ChResetCmd[i] := FALSE;
    IF rReset.Q THEN
        TripLatched[i] := FALSE;
        retryUsed[i] := FALSE;
    END_IF;
    IF ChTrip[i] AND NOT TripLatched[i] THEN
        tripAge[i] := tripAge[i] + CycleMs;
        IF Prio[i] = 1 AND NOT retryUsed[i] THEN
            IF tripAge[i] >= RETRY_MS THEN
                ChResetCmd[i] := TRUE;       (* one automatic retry for vital channels *)
                retryUsed[i] := TRUE;
                tripAge[i] := 0;
            END_IF;
        ELSE
            TripLatched[i] := TRUE;          (* second trip, or a non-vital channel: latch *)
        END_IF;
    ELSIF NOT ChTrip[i] THEN
        tripAge[i] := 0;
    END_IF;

    allowed := Prio[i] >= 1 AND Prio[i] <= 4 - ShedLevel;
    IF CbrnAlarm AND i = CbrnCh AND Prio[i] >= 1 THEN
        allowed := TRUE;
    END_IF;
    ChOn[i] := ChReq[i] AND allowed AND NOT LvdActive AND NOT TripLatched[i];
END_FOR;
END_FUNCTION_BLOCK

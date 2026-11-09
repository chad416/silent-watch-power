(* FB_ArcDetect: DC arc-fault decision logic for one SSPC channel, scanned once per 1 ms feature frame.
   DOSSIER 9.3, SR-41..SR-43. The SSPC signal processor supplies, per frame:
     HfNorm  - mean 5-20 kHz band envelope divided by the frame's mean current (NASA-style detector band)
     Dips    - number of current dips below 30 % of the frame mean lasting >= 20 us (arc extinctions)
     BurstMs - time in the frame above 2 x the 100 ms running mean and above 30 A (parallel-arc bursts)
     IMean   - frame mean current, A
   Series arc: evidence = band energy AND extinction dips together, integrated with a leak; a steady
   in-band ripple (motor, radio, bus ripple) has energy but no dips, a load step has neither.
   Parallel arc: three separate bursts inside 500 ms; a single switch-on inrush is one burst. *)
FUNCTION_BLOCK FB_ArcDetect
VAR_INPUT
    HfNorm : REAL;
    Dips : INT;
    BurstMs : REAL;
    IMean : REAL;
    Reset : BOOL;
END_VAR
VAR_OUTPUT
    Trip : BOOL;
    Kind : INT;                      (* 1 series arc, 2 parallel arc *)
    SerScore : REAL;
    BurstCount : INT;
END_VAR
VAR
    evidence : BOOL;
    inBurst : BOOL;
    winMs : DINT;
END_VAR
VAR CONSTANT
    I_MIN : REAL := 0.5;             (* below ~0.43 A a copper arc is not sustained *)
    TH_HF : REAL := 0.03;            (* tuned on training seeds 0-99 *)
    SER_DECAY : REAL := 0.97;
    SER_TRIP : REAL := 10.0;
    BURST_MIN_MS : REAL := 0.3;
    PAR_COUNT : INT := 3;
    PAR_WIN_MS : DINT := 500;
END_VAR

IF Reset THEN
    Trip := FALSE;
    Kind := 0;
    SerScore := 0.0;
    BurstCount := 0;
    winMs := 0;
END_IF;

(* series arc *)
evidence := IMean > I_MIN AND HfNorm > TH_HF AND Dips >= 1;
IF evidence THEN
    SerScore := SerScore * SER_DECAY + 1.0;
ELSE
    SerScore := SerScore * SER_DECAY;
END_IF;
IF SerScore >= SER_TRIP AND NOT Trip THEN
    Trip := TRUE;
    Kind := 1;
END_IF;

(* parallel arc *)
IF BurstCount > 0 THEN
    winMs := winMs + 1;
    IF winMs > PAR_WIN_MS THEN
        BurstCount := 0;
        winMs := 0;
    END_IF;
END_IF;
IF BurstMs >= BURST_MIN_MS THEN
    IF NOT inBurst THEN
        BurstCount := BurstCount + 1;
        IF BurstCount = 1 THEN
            winMs := 0;
        END_IF;
    END_IF;
    inBurst := TRUE;
ELSE
    inBurst := FALSE;
END_IF;
IF BurstCount >= PAR_COUNT AND NOT Trip THEN
    Trip := TRUE;
    Kind := 2;
END_IF;
END_FUNCTION_BLOCK

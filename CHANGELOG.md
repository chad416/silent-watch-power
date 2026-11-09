# Changelog

## 0.2.0 - 2026-11-10 (dossier Rev B)

- Read the revisions in force, MIL-STD-1275F (2022) and MIL-PRF-32565C w/Amendment 2 (2022). Added the 1275F cases: 500 ms two-stage surge, 8.2 uF spike circuit, 4 V p-p ripple, the six-event starting test, and the 14.4 V cranking case.
- Front end v4. The 1275F surge tail outlasted v3's 0.2 s fault timer, so v4 has a 0.8 s timer and 250 V FETs. v4 passes all 14 E and F cases.
- Thermal model sourced and checked:
  - compartment limit 32.2 degC from a published crewed-vehicle requirement (was an assumed 40 degC);
  - hot-dry case 49 degC with 730 W/m2;
  - cooling COP anchored at 1.5 at 52 degC;
  - plausibility check against two measured vehicle interiors.
- Recommendation moves from 8 to 10 Li-ion 6T blocks: 4.58 h at +43 degC, 6.28 h at -32 degC.
- New studies:
  - 4 h / 12 h crew-posture study;
  - cranking-voltage conflict study and interface issue report IIR-01;
  - arc-fault detection (`swpower/arcfault.py`, `plc/FB_ArcDetect.st`) scored on held-out cases.
- New requirements SR-23 and SR-41..43; IT07 and IT08 added.
- Vendor document addresses removed from the repository; vendor documents are cited by generic title.

## 0.1.0 - 2026-11-09 (dossier Rev A)

- First issue:
  - dossier on the Wasson spine;
  - battery, bus, source, thermal, MIL-STD-1275E front-end, SSPC, trade-study and RMA models;
  - `plc/FB_PowerMgmt.st` with SCL export;
  - 48 tests and CI.

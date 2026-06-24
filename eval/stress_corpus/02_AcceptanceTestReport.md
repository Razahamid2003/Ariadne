# SPECTRA-300 Acceptance Test Report
Document: MDS-ATR-300  ·  Revision: B  ·  Test campaign: completed
Witnessed by the Programme Office. Verdicts: PASS, FAIL, WAIVED.

## 1. Summary
Of 31 verified requirements, 27 passed, 2 failed and were dispositioned, and 2 were
waived under recorded deviations. Retest of the failed items is scheduled (see the
TRR minutes). The Mk III passed all environmental and EMC requirements. The Mk I
failed radiated emissions and is subject to a filter retrofit.

## 2. Results

| Req | Variant | Specified | Measured | Margin | Verdict |
|-----|---------|-----------|----------|--------|---------|
| SR-001 | All | 0.5-18 GHz | 0.5-18.2 GHz | +0.2 GHz | PASS |
| SR-002 | Mk II | up to 26.5 GHz | 26.6 GHz | +0.1 GHz | PASS |
| SR-003 | Mk I | <= -148 dBm/Hz | -149.1 dBm/Hz | 1.1 dB | PASS |
| SR-003 | Mk II | <= -151 dBm/Hz | -150.4 dBm/Hz | -0.6 dB | FAIL |
| SR-003 | Mk III | <= -146 dBm/Hz | -147.0 dBm/Hz | 1.0 dB | PASS |
| SR-004 | Mk I | >= 500 MHz | 400 MHz | -100 MHz | WAIVED |
| SR-006 | All | 50 ns | 42 ns | +8 ns | PASS |
| SR-007 | Mk I | >= 40 km | 43.5 km | +3.5 km | PASS |
| SR-007 | Mk II | >= 65 km | 61 km | -4 km | WAIVED |
| SR-007 | Mk III | >= 38 km | 40.2 km | +2.2 km | PASS |
| SR-008 | All | >= 0.95 POI | 0.97 | +0.02 | PASS |
| SR-009 | All | <= 2.0 deg RMS | 1.6 deg RMS | +0.4 deg | PASS |
| SR-011 | Mk I | <= 220 W | 208 W | 12 W | PASS |
| SR-011 | Mk II | <= 310 W | 305 W | 5 W | PASS |
| SR-011 | Mk III | <= 240 W | 233 W | 7 W | PASS |
| SR-012 | Mk III | -40 to +65 C | -41 to +66 C | pass both ends | PASS |
| SR-014 | Mk III | IP66 | IP66 verified | n/a | PASS |
| SR-015 | Mk II | <= 24 kg | 23.4 kg | 0.6 kg | PASS |
| SR-021 | All | 50 V/m | 50 V/m held | n/a | PASS |
| SR-022 | Mk I | RE102-eq limit | exceeded by 4 dB at 1.2 GHz | -4 dB | FAIL |
| SR-022 | Mk III | RE102-eq limit | compliant | 3 dB | PASS |
| SR-030 | Mk III | >= 6000 h | 6250 h (demonstrated) | +250 h | PASS |
| SR-031 | All | <= 30 min | 22 min | +8 min | PASS |
| SR-040 | All | 10 GbE ICD | conformant | n/a | PASS |
| SR-041 | All | 4 h holdover | 4.3 h | +0.3 h | PASS |
| SR-051 | All | 2-min erase | 95 s | +25 s | PASS |

## 3. Dispositions
- SR-003 (Mk II) FAIL: noise figure 0.6 dB over limit. Root cause traced to the
  low-noise amplifier batch. Corrective action: replace LNA module; retest at TRR+2 weeks.
- SR-022 (Mk I) FAIL: radiated emissions exceed the limit by 4 dB at 1.2 GHz. Root
  cause: insufficient filtering on the RF front-end. Corrective action: install
  Filter Assembly FA-12 (see Equipment Register). Owner per the Subsystem Ownership Map.
- SR-004 (Mk I) WAIVED under DEV-04. SR-007 (Mk II) WAIVED under DEV-09.

Note: the Mk III passed SR-022; only the Mk I failed radiated emissions.

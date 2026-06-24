# SPECTRA-300 Electronic Support Measures Sensor — System Requirements Specification
Document: MDS-SRS-300  ·  Revision: D  ·  Classification: Company Confidential (fictional)
Prepared by Meridian Defence Systems for the Programme Office.

This specification covers the SPECTRA-300 family of wideband signal-monitoring
receivers in three variants: Mk I (baseline ground), Mk II (extended-range ground),
and Mk III (ruggedised naval). Unless a requirement names a variant, it applies to all.

## 1. Performance requirements
- SR-001  The receiver shall cover the instantaneous frequency range 0.5 GHz to 18 GHz.
- SR-002  The Mk II shall extend upper coverage to 26.5 GHz; Mk I and Mk III remain at 18 GHz.
- SR-003  Displayed Average Noise Level (DANL) shall not exceed -148 dBm/Hz (Mk I), -151 dBm/Hz (Mk II), -146 dBm/Hz (Mk III).
- SR-004  Instantaneous bandwidth shall be at least 500 MHz (Mk I), 1 GHz (Mk II), 500 MHz (Mk III).
- SR-005  Frequency measurement accuracy shall be within 1 ppm referenced to the internal OCXO.
- SR-006  Pulse-on-pulse handling shall resolve two pulses separated by 50 ns or more.
- SR-007  The detection range against the reference emitter (Annex A, Table A-1) shall be at least 40 km (Mk I), 65 km (Mk II), 38 km (Mk III).
- SR-008  Probability of intercept shall be at least 0.95 across the SR-001 band within one 2-second scan.
- SR-009  The system shall geolocate an emitter to within 2.0 degrees RMS bearing accuracy.

## 2. Electrical and environmental
- SR-010  Prime power shall be 28 VDC nominal (MIL-STD-1275 equivalent, fictional).
- SR-011  Power consumption shall not exceed 220 W (Mk I), 310 W (Mk II), 240 W (Mk III).
- SR-012  Operating temperature shall be -32 C to +55 C (Mk I/II) and -40 C to +65 C (Mk III).
- SR-013  The Mk III shall meet salt-fog endurance of 96 hours per the naval annex (see Annex B, clause B-4).
- SR-014  Ingress protection shall be IP54 (Mk I/II) and IP66 (Mk III).
- SR-015  Mass shall not exceed 18 kg (Mk I), 24 kg (Mk II), 31 kg (Mk III) excluding antenna.

## 3. Electromagnetic compatibility (EMC)
- SR-020  Conducted emissions shall comply with the programme EMC plan, limit class CE102-equivalent (fictional).
- SR-021  Radiated susceptibility shall be 50 V/m, 30 MHz to 18 GHz.
- SR-022  Radiated emissions shall not exceed the RE102-equivalent limit (fictional) across 10 kHz to 18 GHz.
- SR-023  The Mk III shall additionally meet shipboard EMI per Annex B, clause B-7.

## 4. Reliability, availability, maintainability
- SR-030  Mean Time Between Failures (MTBF) shall be at least 4,500 hours (Mk I), 4,000 hours (Mk II), 6,000 hours (Mk III).
- SR-031  Mean Time To Repair (MTTR) shall not exceed 30 minutes at line-replaceable-unit level.
- SR-032  Operational availability (Ao) shall be at least 0.985.
- SR-033  No single point of failure shall disable both receiver channels simultaneously.

## 5. Interface and data
- SR-040  The system shall output detections over a 10 Gigabit Ethernet interface using the ICD defined in Annex C.
- SR-041  Time synchronisation shall use PTP (IEEE 1588) with a holdover of at least 4 hours within 1 microsecond.
- SR-042  The recording subsystem shall retain at least 8 hours of wideband capture per mission.

## 6. Security
- SR-050  The system shall operate fully disconnected from any external network (air-gapped).
- SR-051  All non-volatile storage shall support a 2-minute emergency erase.
- SR-052  Audit logs shall be retained for 180 days on the system and exported on demand.

## 7. Waivers and deviations
Two deviations are recorded against this revision. Deviation DEV-04 waives SR-004
instantaneous bandwidth on the Mk I from 500 MHz to 400 MHz pending a receiver
firmware update; see the Acceptance Test Report. Deviation DEV-09 concerns SR-007
Mk II detection range and is tracked separately (customer confirmation pending).

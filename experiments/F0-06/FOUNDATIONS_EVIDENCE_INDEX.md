# Foundations kanıt indeksi

Belge r1 · G0 PASS / ACCEPTED · Windows PASS · Linux NOT_RUN

Makine kaydı: [FOUNDATIONS_EVIDENCE_INDEX.json](FOUNDATIONS_EVIDENCE_INDEX.json).
Tüm görevler F0-00–F0-06 COMPLETE; T-F00–T-F09 PASS. REQ-F00 tanımlı değildir; F0-00 → REQ-F01.

| Grup | Geçen | JUnit |
|---|---:|---|
| f00 | 6 | canonical/regression/junit/f00.xml |
| f01 | 16 | canonical/regression/junit/f01.xml |
| f02 | 102 | canonical/regression/junit/f02.xml |
| f03 | 159 | canonical/regression/junit/f03.xml |
| f04 | 39 | canonical/regression/junit/f04.xml |
| f05-unit | 127 | canonical/regression/junit/f05-unit.xml |
| tf08 | 16 | canonical/regression/junit/tf08.xml |
| f05-mutations | 32 | canonical/regression/junit/f05-mutations.xml |
| f06 | 26 | canonical/regression/junit/f06.xml |

212 eski SHA256SUMS kaydı, 18 Stage 1 kaydı, robot manifesti ve 14 yerel Git dışı dosya denetlendi. Eski paylaşılan dosya revizyonları kendi ham commit bloblarıyla doğrulandı; güncel hashle eşitlik iddiası yok.

Config/schema/robot devir hashleri [handoff-inputs.json](handoff-inputs.json), veri/query/result hashleri [reproduction-summary.json](reproduction-summary.json) içindedir. Komut/JUnit/log yolları ve tüm dosya hashleri JSON indeksindedir. G0 etkisi: kritik açık hata yok.

- Linux not executed
- Physical model accuracy/calibration not verified
- Collision and physical robot safety not verified
- Timing is host-specific
- Wide-start DLS performance remains the measured baseline; no success-rate gate
- Coverage is empirical, not complete reachability coverage
- External solvers are Core work, not started
- No trained neural model
- Foundations closure is not production or robot safety approval
- Manufacturer PDF raw bytes/hash remain unavailable
- Historical F0-00 JSON checkout line endings differ from Git blobs

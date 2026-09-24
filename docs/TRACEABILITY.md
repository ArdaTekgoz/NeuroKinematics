# Gereksinim görev test ve kanıt matrisi

Belge r8 · 24 Eylül 2026. F0-00–F0-06 COMPLETE; T-F00–T-F09 PASS; G0 PASS / ACCEPTED. Core READY / NOT_STARTED; C1-01/02/03 başlatılmadı. Sonraki fazların testleri plan hedefidir, çalıştırılmış kanıt değildir. REQ-F00 tanımlı değildir; F0-00 → REQ-F01.

| Görev | Gereksinim | Test | Kanıt | Durum |
|---|---|---|---|---|
| [F0-00](tasks/F0-00.md) | REQ-F01 | T-F00 | [`RUN-20260918-001`](../experiments/F0-00/RUN_REPORT.md) | PASS · TAMAMLANDI |
| [F0-01](tasks/F0-01.md) | REQ-F01 | T-F01 | [`RUN-20260918-002`](../experiments/F0-01/RUN_REPORT.md) | PASS · TAMAMLANDI |
| [F0-02](tasks/F0-02.md) | REQ-F02 | T-F02 | [`RUN-20260918-003`](../experiments/F0-02/RUN_REPORT.md), JSON/JUnit/SHA256SUMS | PASS · TAMAMLANDI; 102/102, 10000 q |
| [F0-03](tasks/F0-03.md) | REQ-F03 | T-F03, T-F04 | [RUN-20260919-001](../experiments/F0-03/RUN_REPORT.md), `jacobian-validation-summary.json`, `metric-validation-summary.json`, JUnit/SHA256SUMS | PASS · TAMAMLANDI; 159/159 |
| [F0-04](tasks/F0-04.md) | REQ-F04 | T-F05, T-F06, T-F07 | [RUN-20260921-001](../experiments/F0-04/RUN_REPORT.md), JSON/JUnit/SHA256SUMS | PASS · TAMAMLANDI |
| [F0-05](tasks/F0-05.md) | REQ-F05 | T-F08 | [RUN_REPORT](../experiments/F0-05/RUN_REPORT.md) | COMPLETE / PASS |
| [F0-06](tasks/F0-06.md) | REQ-F06 | T-F09 | [RUN_REPORT](../experiments/F0-06/RUN_REPORT.md) | COMPLETE / PASS |
| [C1-01](tasks/C1-01.md) | REQ-C01 | T-C00 | `experiments/C1-01/` | NOT_STARTED; G0 sonrası uygun |
| [C1-02](tasks/C1-02.md) | REQ-C03 | T-C07 | `experiments/C1-02/` | NOT_STARTED; G0 sonrası uygun |
| [C1-03](tasks/C1-03.md) | REQ-C02 | T-C01, T-C02 | `experiments/C1-03/` | NOT_STARTED; G0 sonrası uygun |
| [C1-04](tasks/C1-04.md) | REQ-C03 | T-C03 | `experiments/C1-04/` | PLANLANDI |
| [C1-05](tasks/C1-05.md) | REQ-C03, REQ-C04 | T-C04 | `experiments/C1-05/` | PLANLANDI |
| [C1-06](tasks/C1-06.md) | REQ-C04, REQ-C05 | T-C05 | `experiments/C1-06/` | PLANLANDI |
| [C1-07](tasks/C1-07.md) | REQ-C06 | T-C06 | `experiments/C1-07/` | PLANLANDI |
| [H2-01](tasks/H2-01.md) | REQ-H01 | T-H01 | `experiments/H2-01/` | PLANLANDI |
| [H2-02](tasks/H2-02.md) | REQ-H02 | T-H02 | `experiments/H2-02/` | PLANLANDI |
| [H2-03](tasks/H2-03.md) | REQ-H03 | T-H03 | `experiments/H2-03/` | PLANLANDI |
| [H2-04](tasks/H2-04.md) | REQ-H04 | T-H04, T-H05 | `experiments/H2-04/` | PLANLANDI |
| [H2-05](tasks/H2-05.md) | REQ-H05 | T-H06 | `experiments/H2-05/` | PLANLANDI |
| [H2-06](tasks/H2-06.md) | REQ-H06 | T-H07 | `experiments/H2-06/` | PLANLANDI |
| [S3-01](tasks/S3-01.md) | REQ-S01 | T-S01 | `experiments/S3-01/` | PLANLANDI |
| [S3-02](tasks/S3-02.md) | REQ-S02 | T-S02 | `experiments/S3-02/` | PLANLANDI |
| [S3-03](tasks/S3-03.md) | REQ-S03, REQ-S04 | T-S03, T-S05 | `experiments/S3-03/` | PLANLANDI |
| [S3-04](tasks/S3-04.md) | REQ-S05, REQ-S06 | T-S04, T-S06 | `experiments/S3-04/` | PLANLANDI |
| [S3-05](tasks/S3-05.md) | REQ-S06 | T-S07 | `experiments/S3-05/` | PLANLANDI |

## REQ-F02 uygulama ve kanıt bağı

Uygulama commit'i: `d92dd213bb96f8932bd0019541dd13dd7365afaf`.

| Gereklilik | Değişiklik | Test | Ham kanıt |
|---|---|---|---|
| Immutable kimlik, sıra ve limit | `kinematics/model.py` | `test_robot_reference.py` | `chain-inspection.json`, JUnit |
| Bağımsız XML zinciri ve float64 FK | `chain.py`, `transforms.py`, `custom_fk.py` | `test_math_chain.py`, Pinocchio'suz süreç | `unit-junit.xml` |
| İsimden referans joint/frame ve relatif base | `pinocchio_fk.py` | `test_robot_reference.py` | `chain-inspection.json`, `handpicked-results.json` |
| Sabit PCG64/seed, örnek hash'i ve hata yolları | `validation.py` | `test_evidence.py` | `config.json`, `sample-hash.json`, `diagnostics.json` |
| 10000 q / iki maksimum ≤1e-9 | `validation.py`, Pixi görevleri | `test_acceptance.py` | `fk-validation-summary.json`, `pytest-junit.xml` |
| F0-00/F0-01 korunması | `scripts/run_f02_acceptance.py` | F0-00 6/6, F0-01 16/16 | `commands.json`, `f00-junit.xml`, `f01-junit.xml` |

Yukarıdaki kanıt dosyalarının kökü `experiments/F0-02/`, modüllerin kökü
`src/neurokinematics/kinematics/`, testlerin kökü `tests/f0_02/`.
Maksimumlar: konum `4.75098925995612e-16 m`, rotation Frobenius
`9.159602786276758e-16`; aşım ve nonfinite sayıları sıfır. F0-02 kapanışı
F0-03'e geçişi hazırladı; F0-03'ün güncel sonucu aşağıdadır.


## REQ-F03 uygulama ve kanıt bağı

Uygulama commit'i: 981f6143ce38574021edac7373586976cf97bdf4.

| Gereklilik | Değişiklik | Test | Kanıt (experiments/F0-03) |
|---|---|---|---|
| Bağımsız geometrik Jacobian, TCP/base | jacobian.py | test_analytic.py | unit-junit.xml, chain-and-frame-contract.json |
| Pinocchio frame/satır/sütun | pinocchio_jacobian.py | nonidentity base ve reference-frame testi | unit-junit.xml |
| SO(3) merkezi fark, h/limit | finite_difference.py | küçük/pi açı, limit ve ikinci derece stencil | unit-junit.xml |
| Metrikler ve SVD | metrics.py | test_metrics.py 48/48 | metric-validation-summary.json, metrics-junit.xml |
| 256 deterministik +21 elle seçilmiş q | jacobian_validation.py | test_acceptance.py, üç h | jacobian-validation-summary.json, sample-hash.json |
| 12 hatayı yakalama | test_mutations.py | 12/12 detected | mutation-results.json |
| Hataları/singularity alt grubunu koruma | jacobian_validation.py | test_evidence.py | diagnostics.json, handpicked-jacobians.json |
| F0-00/01/02 regresyon ve hash | run_f03_acceptance.py | 6/16/102 PASS, checksum corruption testi | commands.json, JUnit, SHA256SUMS |

İlk hash durdurması tarihsel preflight.json'da; kullanıcı düzeltmesi ve dört
gerçek immutable hash kontrolü authorized-preflight.json'da korunur.
F0-04 tamamlandı. Güncel G0 kararı PASS / ACCEPTED; sonraki fazlar başlatılmadı.

## REQ-F04 uygulama ve kanıt bağı

Uygulama commit'i: `16010d518c24400f6c6d43a2459456dd822f34a8`.

| Gereklilik | Değişiklik | Test | Kanıt (experiments/F0-04) |
|---|---|---|---|
| LHS, canonical typed-array hash ve deterministik shard | `data/factory.py` | T-F05 | dataset manifest, determinism summary |
| Group-first split, duplicate ve train-only normalizasyon | `data/factory.py` | T-F06 | split/duplicate/normalization JSON |
| Pinocchio FK etiketi ve bağımsız FK yeniden denetimi | F0-02 servisleri + factory | T-F07 | fk-validation-summary.json |
| Normalize Jacobian/SVD ve hard subsetler | F0-03 servisleri + factory | T-F07 | hard-subsets-summary.json |
| Ampirik coverage ve üç çözünürlük | frozen config + `coverage()` | T-F07 | coverage summary/sensitivity |
| 17 hata sınıfını yakalama | doğrulayıcılar | mutation suite 17/17 | mutation-results.json |
| Regresyon ve bütünlük | `run_f04_acceptance.py` | 6/16/102/159 PASS | commands, JUnit, SHA256SUMS |

F0-05 COMPLETE; güncel kapanış ve G0 kanıtı aşağıdaki REQ-F05/REQ-F06 kayıtlarındadır.

## REQ-F05 uygulama ve kanıt bağı · 24 Eylül 2026 · belge r2

| Gereklilik | Değişiklik | Test | Kanıt (experiments/F0-05) |
|---|---|---|---|
| Sabit damping DLS, base/TCP ve bağımsız Pinocchio FK | `solvers/dls.py`, `benchmark/validation.py` | unit 127/127; T-F08 16/16 | config, solver-config, JUnit |
| Query bağımsızlığı ve determinism | `benchmark/queries.py` | iki üretim, 12.000 query, F0-04 duplicate/group=0 | query-manifest, query-hashes |
| 10/50 ms, beş pass, tam JSONL ve ayrık timing | `benchmark/runner.py`, CLI | 120.000 satır doğrulama | solver-summary, result-verification, result-manifest |
| Profil/subset hata ve latency dağılımları | aggregator | tüm ve başarılı denemeler ayrı | benchmark-summary, subgroup-summary, deadline-summary |
| Bozuk girdi/analitik dış sınır/kanıtsız çözülmeme | proof + validator | T-F08 | failure-cases, tf08-junit |
| Production hatalarını kabulde reddetme | mutasyon testleri | 32/32 algılama | mutation-results, mutation-junit |
| Eski görevler, lock ve kanıt bütünlüğü | `run_f05_acceptance.py`, Pixi | F0-00–04 6/16/102/159/39; SHA verify PASS | commands, JUnitler, SHA256SUMS |

**Karar:** F0-05 PASS / TAMAMLANDI; [RUN-20260924-F005](../../experiments/F0-05/RUN_REPORT.md).
Uygulama commit'i `3e55954`.
Bu satırlar önceki, tarihsel F0-04 devir cümlesini güncel sonuç olarak
yorumlamaz. F0-06 COMPLETE; G0 PASS / ACCEPTED.

## REQ-F06 uygulama ve kanıt bağı

| Gereksinim | Değişiklik | Test | Kanıt (experiments/F0-06) |
|---|---|---|---|
| Önceki görev bütünlüğü | audit_f06_history.py; baseline/overlay ayrımı | 212 + 18 tarihsel checksum, 29 robot manifest dosyası, 14 yerel büyük dosya | history-audit.json, stage1-audit.json |
| Kilitli temiz kurulum | run_f06_clean.py; yeni detached worktree | install --locked; lock --check; resmi regresyon 497/497 | clean-environment.json, commands.json, canonical/regression/junit |
| Gerçek küçük uçtan uca tekrar | Açık config/manifest public API parametreleri; reproduce_f06.py | İki bağımsız 384 veri/query ve 768 benchmark; FK/schema/split/train-only/limit PASS | canonical/reproduction, reproduction-summary.json |
| Hataları reddeden kapanış | foundations_gate.py | 26/26; 25 negatif kontrol | mutation-results.json, f06.xml |
| G0 ve Core devri | Hashli paket, indeks ve karar belgeleri | JSON yapısal ve checksum kontrolü | FOUNDATIONS_EVIDENCE_INDEX.json, SHA256SUMS, G0_DECISION.md, CORE_HANDOFF.md |

F0-06 uygulama commit'i `8e53698a414d34e96039e64a48c153e7decfb7b1`.
G0 PASS / ACCEPTED; Foundations COMPLETE; Core READY / NOT_STARTED.
Linux NOT_RUN; fiziksel robot ve collision/safety doğrulanmadı.

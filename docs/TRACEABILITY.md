# Gereksinim görev test ve kanıt matrisi

Belge r18 · 3 Ekim 2026. F0-00–F0-06 COMPLETE; T-F00–T-F09 PASS; G0 PASS / ACCEPTED. Core aktif; C1-01 COMPLETE / T-C00 PASS, C1-02 COMPLETE / T-C07 PASS, C1-03 COMPLETE / T-C01 ve T-C02 PASS / ACCEPTED, C1-04 COMPLETE / T-C03 PASS / doğrudan IK NO-GO. Diğer Core testleri çalıştırılmış kanıt değildir. REQ-F00 tanımlı değildir; F0-00 → REQ-F01.

| Görev | Gereksinim | Test | Kanıt | Durum |
|---|---|---|---|---|
| [F0-00](tasks/F0-00.md) | REQ-F01 | T-F00 | [`RUN-20260918-001`](../experiments/F0-00/RUN_REPORT.md) | PASS · TAMAMLANDI |
| [F0-01](tasks/F0-01.md) | REQ-F01 | T-F01 | [`RUN-20260918-002`](../experiments/F0-01/RUN_REPORT.md) | PASS · TAMAMLANDI |
| [F0-02](tasks/F0-02.md) | REQ-F02 | T-F02 | [`RUN-20260918-003`](../experiments/F0-02/RUN_REPORT.md), JSON/JUnit/SHA256SUMS | PASS · TAMAMLANDI; 102/102, 10000 q |
| [F0-03](tasks/F0-03.md) | REQ-F03 | T-F03, T-F04 | [RUN-20260919-001](../experiments/F0-03/RUN_REPORT.md), `jacobian-validation-summary.json`, `metric-validation-summary.json`, JUnit/SHA256SUMS | PASS · TAMAMLANDI; 159/159 |
| [F0-04](tasks/F0-04.md) | REQ-F04 | T-F05, T-F06, T-F07 | [RUN-20260921-001](../experiments/F0-04/RUN_REPORT.md), JSON/JUnit/SHA256SUMS | PASS · TAMAMLANDI |
| [F0-05](tasks/F0-05.md) | REQ-F05 | T-F08 | [RUN_REPORT](../experiments/F0-05/RUN_REPORT.md) | COMPLETE / PASS |
| [F0-06](tasks/F0-06.md) | REQ-F06 | T-F09 | [RUN_REPORT](../experiments/F0-06/RUN_REPORT.md) | COMPLETE / PASS |
| [C1-01](tasks/C1-01.md) | REQ-C01 | T-C00 | [Kabul raporu](../experiments/C1-01/RUN-20260928-T-C00-acceptance.md), [verify gate](../experiments/C1-01/udp-v2/verify/gate.json) | COMPLETE / PASS / ACCEPTED; 600.000 kayıt |
| [C1-02](tasks/C1-02.md) | REQ-C03 | T-C07 | [Aşama 2 kabul raporu](../experiments/C1-02/RUN-20260929-T-C07-acceptance.md), [acceptance](../experiments/C1-02/acceptance.json), [tekrar üretim](../experiments/C1-02/determinism-summary.json), [SHA](../experiments/C1-02/SHA256SUMS) | COMPLETE / PASS / ACCEPTED; 24.000 çift, 34/34 eş shard; 2.281 eksik wide etiketi korundu |
| [C1-03](tasks/C1-03.md) | REQ-C02 | T-C01, T-C02 | [Kabul raporu](../experiments/C1-03/stage2/RUN_REPORT.md), [acceptance](../experiments/C1-03/stage2/acceptance.json), [SHA](../experiments/C1-03/stage2/SHA256SUMS) | COMPLETE / PASS / ACCEPTED; 1086 q/dtype, 32 gradient q, iki koşu eş |
| [C1-04](tasks/C1-04.md) | REQ-C03 | T-C03 | [Aşama 2 çalışma kaydı](../experiments/C1-04/stage2/RUN_REPORT.md), [E-C01 özeti](../experiments/C1-04/stage2/E-C01-summary.json), [kapanış kararı](../experiments/C1-04/stage2/acceptance.json), [SHA](../experiments/C1-04/stage2/SHA256SUMS) | COMPLETE / T-C03 PASS / E-C01 üç seed; doğrudan IK NO-GO |
| [C1-05](tasks/C1-05.md) | REQ-C03, REQ-C04 | T-C04 | [Çalışma kaydı](../experiments/C1-05/stage2/RUN_REPORT.md), [sonuçlar](../experiments/C1-05/stage2/RESULTS.md), [karar](../experiments/C1-05/stage2/acceptance.json), [hashli C1-06 devri](../experiments/C1-05/stage2/C1-06-handoff.json). | COMPLETE / T-C04 PASS; doğrudan IK NO-GO |
| [C1-06](tasks/C1-06.md) | REQ-C04, REQ-C05 | T-C05 | [Çalışma kaydı](../experiments/C1-06/stage2/RUN_REPORT.md), [sonuçlar](../experiments/C1-06/stage2/RESULTS.md), [kabul](../experiments/C1-06/stage2/final-001/acceptance.json) | COMPLETE / T-C05 PASS; H2 REDDEDİLDİ; doğrudan IK NO-GO |
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

**Karar:** F0-05 PASS / TAMAMLANDI; [RUN-20260924-F005](../experiments/F0-05/RUN_REPORT.md).
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

## REQ-C01 Aşama 2 devam bağı · 25 Eylül 2026

| Gereklilik | Değişiklik | Test | Kanıt / durum |
|---|---|---|---|
| Dondurulmuş sorgu, robot/TCP ve beş solver kimliği | `core/contract.py`, C1-01 şeması | `tests/c1_01/test_adapter.py` | 8/8 yerel PASS; `q_target` worker girdisinde yok |
| Ortak IPC, hata ve bağımsız aday kontrolü | `core/worker.py`, `results.py`, DLS/MoveIt worker | Adapter negatif testleri, Pinocchio FK tahrifi | Yerel PASS; C++ derleme NOT_RUN |
| Beş solver smoke kapısı ve tam 12.000 × 2 × 5 deneme | `core/runner.py`, `cli.py` | Kolay/sınır/tekillik smoke; T-C00 | Docker ortamı bekleniyor; smoke ve tam benchmark NOT_RUN |

Durum `IN_PROGRESS / STAGE_2_IMPLEMENTING`. Tam REQ-C01 → T-C00 → ham sonuç → bağımsız doğrulama → karar zinciri henüz kapanmadı; [devam raporu](../experiments/C1-01/RUN_REPORT.md).

26 Eylül 2026 ek kanıt: kullanıcı Docker/WSL/Ubuntu x86_64 başlangıç kontrolü PASS; Dockerfile + apt/source/build/lock scriptleri hazır, gerçek build ve solver smoke NOT_RUN. [COMMANDS.md](../experiments/C1-01/COMMANDS.md) hazır komutu ve kalıcı ekran görüntülerini ayırır.

2026-09-26 REQ-C01 → pinned Docker build/lock → Linux adapter8PASS, beş smoke/offline PASS, ADR-008 scoped regression209PASS → linux-portable-critical-regression.xml ve linux-smoke/*.jsonl. T-C00 ve 10/50ms pilot NOT_RUN; scripts/run_c101_pilot.py hazır. Stage2 commit/push kapanış sonrası.

2026-09-26 REQ-C01/T-C00 fullINCOMPLETE → linux-full/benchmark-manifest.json, globalstderr+summary; 4tam120000yöntem henüzofflineNOT_RUN. Workerqueueisolationfix →11PASSstage2-queue-isolation-tests.xml; yeniLinux doğrulamaNOT_RUN.

2026-09-26 REQ-C01 dört yöntemin480000kayıtintegrityPASS → linux-full/completed-methods-verification.json ve *-verified-summary.json. GlobalINCOMPLETEdolayısıylaT-C00kabulüyok. IPCqueuefixyerel11PASS; hedefli500kayıtrestartstresshazırNOT_RUN.

2026-09-26 REQ-C01/DDSlifecyclefix → ADR-009UDPv4 →500restartstressPASS → linux-global-restart-stress-udp/stress-gate.json(raw/lockhashPASS). TamT-C00kabulüyok;ANA testöncesikullanıcıonaysınırı,MAIN_TEST_HANDOFF.md.

## 27 Eylül 2026 · REQ-C01 ana koşu hazırlığı

| Gereklilik | Değişiklik | Doğrulama / kanıt |
|---|---|---|
| Aynı runtime ve tekrar üretim | UDPv4, salt okunur kaynak/test kopyaları, runtime SHA ve aşamalar arası kanıt bağı | `tests/c1_01/test_session_gates.py`; yeni Linux prepare NOT_RUN |
| Kalan bütçe ve ortak toplam süre | Mutlak monoton bitiş zamanı, worker tarafında kalan süre, toplamda giriş hazırlığı | Süre protokolü regresyonları; yeni native build/probe NOT_RUN |
| Hata kanıtı ve bitiş bütünlüğü | Bozuk JSON baytları, VALIDATION_ERROR ayrımı, gereksiz son restart kaldırıldı | `tests/c1_01/test_main_runner_regressions.py` |
| Kilitli bağımlılıklarla build | Mevcut exact image üzerinden yalnız yerel worker build ve closure denetimi | `scripts/build_c101_runtime.ps1`; Linux NOT_RUN |

Sonuçlar ve komutlar: [RUN-20260927-main-preparation](../experiments/C1-01/RUN-20260927-main-preparation.md). Tam T-C00 ve görev kabulü bekleniyor.

27 Eylül ek bağ: REQ-C01 kalan süre protokolü → yeni native worker image `00a76905...` → gerçek worker build ve dependency/input audit PASS (`runtime-build-v1/docker-build.log`). Runtime girişinde Pixi reinstall engeli → `PIXI_NO_INSTALL=true`, `-ResumeLock` → yerel 23 session testi + lock yayınlama/failure kontrolleri PASS (`runtime-noinstall-session-tests.xml`, `runtime-lock-resume-check.json`). Final Linux lock ve T-C00 NOT_RUN.

27 Eylül güncel bağ: `-ResumeLock` Linux kullanıcı koşusu PASS → yeni environment-lock SHA `9369f45b...`, bağımlılık/kaynak/input audit PASS → `runtime-build-v1/lock-binding-verification.json` salt okunur PASS. Yeni Linux prepare/protokol, smoke/pilot ve T-C00 ölçümü henüz NOT_RUN.

27 Eylül REQ-C01 aynı runtime/kalan bütçe → `udp-v1` Linux prepare kullanıcı koşusu 258 PASS / 1 ADR-008 deselected + beş gerçek worker expired-request probe PASS → `udp-v1/prepare/gate.json`, `regression.xml`, `protocol/expired-request-probe.json`. Gate dosyaları, test kimlikleri ve snapshot/runtime bağı salt okunur PASS → `udp-v1-prepare-evidence-check.json`. Yeni smoke/pilot/T-C00 NOT_RUN; tam kabul bekliyor.

27 Eylül REQ-C01 aynı runtime kapısı → smoke ölçüm öncesi drift FAIL (kullanıcı logu ve session failure JSON) → salt okunur alan farkı tanısı `scripts/diagnose_c101_runtime.py` / `.ps1`. Yerel syntax/fark tespiti/snapshot hash kontrolü PASS; gerçek Docker tanısı ve yeni smoke ölçümleri NOT_RUN. Kilit ve eşikler değiştirilmedi.

27 Eylül aynı runtime kapısı → kullanıcı tanısı MATCH / sıfır fark (`udp-v1-runtime-diagnostic-cd126d7c4b774b23b204a085887944d9.log`); önceki drift nedeni belirlenmedi. Prepare PASS korunarak aynı session smoke tekrar denemesi sırada; ana ölçümler henüz NOT_RUN.

27 Eylül ikinci drift FAIL → `diagnose_c101_runtime.py --smoke-startup` aynı kayıtlı main başlangıcını ölçüm öncesinde durdurur → yerel dispatch engelleme/mutex cleanup/restore PASS. Gerçek Docker tanısı NOT_RUN; snapshot/lock/gate ve kabul eşikleri değişmedi.

27 Eylül REQ-C01 aynı kaynak tavanı → tanıda yalnız MemTotal 8 kB farkı → ADR-010 / `memory_policy` cgroup v2 8 GiB + swap0, host RAM ayrı gözlem → `memory-policy-session-tests.xml` 29 PASS + `session-launcher-flow-check.json` mock akış PASS. Frozen 52 PASS; yeni Linux `udp-v2` prepare ve ana ölçümler NOT_RUN. Solver toleransları/bütçeleri değişmedi.

27 Eylül REQ-C01 / ADR-010 → `udp-v2` Linux prepare 264 PASS / 1 deselected + beş worker probe PASS + gerçek 8 GiB/swap0 policy → `udp-v2/prepare/gate.json`, `regression.xml`, `runtime-lock.json`. Hash/test kimliği/snapshot bağları salt okunur PASS → `udp-v2-prepare-evidence-check.json`. Yeni ana ölçümler NOT_RUN.

27 Eylül REQ-C01 beş yöntem entegrasyonu → `udp-v2` smoke 40 SUCCESS / beş solver PASS, Linux bağımsız FK/sıra kontrolü → `udp-v2/smoke/gate.json` ve beş raw/summary/stderr. Hash/bayt/satır/status/warmup ve runtime/prepare bağları salt okunur PASS → `udp-v2-smoke-evidence-check.json`. Pilot/full/verify henüz NOT_RUN.

27 Eylül REQ-C01 pilot bütünlüğü → `udp-v2` pilot 120 kayıt / beş yöntem PASS, fatal0 → `udp-v2/pilot/gate.json` ve beş raw/summary/stderr. Runtime/önceki gate/hash/order/status/warmup bağları PASS → `udp-v2-pilot-evidence-check.json`. Global restart/warmup maliyeti kayıtlı; full T-C00 ve verify NOT_RUN.

28 Eylül REQ-C01 T-C00 full ölçüm → `udp-v2/full/gate.json`: 600.000 kayıt, beş yöntem MEASURED_UNVERIFIED. Ham SHA/bayt/satır ve runtime/önceki gate bağları salt okunur PASS → `udp-v2-full-evidence-check.json`. Bağımsız offline satır/FK doğrulaması henüz NOT_RUN; sonuç kabul edilmedi.

28 Eylül REQ-C01 → aynı frozen girdiler + beş solver/C++ worker + UDPv4/8 GiB runtime → prepare 264 PASS/1 deselected, smoke 40 SUCCESS, pilot 120 PASS, full 600.000 ölçüm → `udp-v2/verify/gate.json` beş yöntem PASS / fatal0; summary/gate hash bağı `udp-v2-verify-evidence-check.json` PASS → [T-C00 kabul raporu](../experiments/C1-01/RUN-20260928-T-C00-acceptance.md). Karar **REQ-C01 / T-C00 PASS / ACCEPTED, C1-01 COMPLETE**. Raw full LOCAL_ONLY; uzak arşiv NOT_CONFIRMED. C1-02 ve diğer Core işleri NOT_STARTED.

## 29 Eylül 2026 · REQ-C03 / T-C07 kabul bağı

F0-04 kökleri ve frozen C1-02 config/schema → `pairs.py` local/wide üretimi ve dört adaylı DLS teacher → 90 satırlık kaynak kontrollü pilot ve iki ayrı 24.000 satırlık üretim → `pair_validation.py` kaynak, soy, split, benchmark, normalizasyon ve input izolasyon denetimi → 7 T-C07 + 12 mutation + 42 seçili arayüz regresyonu PASS → [kabul raporu](../experiments/C1-02/RUN-20260929-T-C07-acceptance.md). Ham 50/50 dağılım, 2.281 eksik wide etiketiyle birlikte korundu; test satırı filtrelenmedi. İkinci üretimde 34/34 shard ve canonical SHA eş; karar **REQ-C03 / T-C07 PASS / ACCEPTED, C1-02 COMPLETE**. C1-03 NOT_STARTED, C1-04 bağımlı, G1 açık.

## 29 Eylül 2026 · REQ-C02 Aşama 1 bağı

| Gereklilik | Dondurulan değişiklik | Kontrol / kanıt | Durum |
|---|---|---|---|
| Robot/TCP ve referans kimliği | 85 girdili manifest, 15 G0 ve 29 robot dosyası | input-hashes.json, stage1-check.json | Statik hash/yapı doğrulandı |
| p/R ve autograd, dtype/batch/frame | Küçük Torch kernel tasarımı; ADR-011 | STAGE1_REVIEW.md, TEST_MATRIX.md | Uygulama NOT_RUN |
| T-C01 / T-C02 kabulü | 1086 q, 32 gradient q, eşikler/scalars/stencil | config.json, samples.jsonl, SAMPLING_CONTRACT.md | T-C01 NOT_RUN / T-C02 NOT_RUN |
| Hata ve tekrar üretim | 24 negatif/mutation sınıfı, ikinci temiz ortam kapısı | NEGATIVE_MUTATION_MATRIX.md, COMMANDS.md | NOT_RUN |

Kanıt kökü [experiments/C1-03](../experiments/C1-03/RUN_REPORT.md). Genel durum **IN_PROGRESS / STAGE_1_COMPLETE**; gereksinim kabul bağı henüz kapanmadı. Kullanıcının açık Aşama 2 onayı bekleniyor. C1-02 dosyaları korunur; C1-04 başlamaz.

## 29 Eylül 2026 · C1-03 Aşama 2 ara kaydı

Açık kullanıcı onayı ve Stage1 hash denetimi sonrası Torch FK uygulandı. Yerel T-C01/T-C02, 110 unit/negatif/arayüz testi ve 277 Foundations regresyonu geçti. ADR-012 ve protokol r2 ortam/harness düzeltmelerini kaydeder; eşikler ve örnekler değişmedi. **IN_PROGRESS / CLEAN_REPRODUCTION_PENDING**; ikinci temiz ortam ve nihai kabul audit bekleniyor. [Çalışma kaydı](../experiments/C1-03/stage2/RUN_REPORT.md). C1-04 başlamadı.

## 29 Eylül 2026 · C1-03 nihai kabul

**REQ-C02 / T-C01 / T-C02 PASS / ACCEPTED; C1-03 COMPLETE.**
Standart Torch fixed/revolute kernel, dondurulmuş KUKA robot/TCP/frame ve
autograd sözleşmesini iki gerçek koşuda geçti. Her ortamda 1086 q/dtype,
32 iç konfigürasyon/2880 türev, 32 gradcheck/Jacobian, sensitivity/batch/edge,
110 test (21 C1-02 arayüz dahil), 24 öldürülen gerçek source mutant ve
277 Foundations regresyonu PASS; skip0. Yeni checkout/ortamda 15/15 komut
PASS; 2317 satır/2695396 bayt raw sonuçlar iki koşuda bayt düzeyinde aynı.
Eşikler, örnekler, Foundations ve C1-02 girdileri değiştirilmedi.

[Nihai çalışma kaydı](../experiments/C1-03/stage2/RUN_REPORT.md),
[kabul kararı](../experiments/C1-03/stage2/acceptance.json),
[komutlar](../experiments/C1-03/stage2/COMMANDS.md) ve
[kanıt manifesti](../experiments/C1-03/stage2/evidence-manifest.json).
Uygulama commitleri `7d9e282` ve `4022e2359306a780422c94f25252bd2eaa90ed8f`.
Kapanış commit kimliği Git geçmişinden okunur. Önceki ara durum kayıtları
tarihseldir; güncel karar bu kabul kaydıdır. Linux/CUDA/fiziksel robot NOT_RUN;
performans/etkin emek NOT_MEASURED. C1-04 girdileri hazır, **NOT_STARTED**;
neural eğitim, G1 kararı ve v1.0.0 etiketi bu kapsamda oluşturulmadı.

| Gereksinim | Değişiklik | Test | Ham kanıt ve karar |
|---|---|---|---|
| REQ-C02 robot/frame/poz eşliği | torch_fk.py; fixed/revolute zincir, dtype/batch doğrulaması | T-C01: 1086 q/dtype +19 analitik | stage2/full-exact-a/results.jsonl, clean-b/full/results.jsonl; PASS |
| REQ-C02 kesintisiz doğru gradyan | Torch Rodrigues/matmul; bağımsız Pinocchio FD validator | T-C02: 32 q/2880 türev, 32 gradcheck/Jacobian, sensitivity/edge | Aynı JSONL; summary/junit; PASS |
| REQ-C02 hata reddi ve korunmuş arayüz | 24 gerçek source mutation; 21 C1-02 arayüz testi | 110 unit +277 F0 regresyonu/ortam, skip0 | unit-exact-a.xml, mutations-exact-a, clean-b eşleri; PASS |
| REQ-C02 tekrarlanabilir kabul | Hashli exact overlay, ADR-012, 15 komut fresh driver | İki source/config/sample/artifact/raw audit | acceptance.json, clean-b/complete.json, evidence-manifest.json; ACCEPTED |

## 2 Ekim 2026 · REQ-C03 / T-C03 Aşama 1 bağı

C1-02 kabulündeki 24.000 çift/34 shard ve 2.281 etiketsiz wide; C1-03 Windows CPU FK kabulü → [C1-04 dondurulmuş config](../experiments/C1-04/config.json), [girdi SHA manifesti](../experiments/C1-04/input-hashes.json), [T-C03/E-C01 test matrisi](../experiments/C1-04/TEST_MATRIX.md) ve [negatif kontroller](../experiments/C1-04/NEGATIVE_CONTROL_MATRIX.md) → Stage1 hash/erişim audit'i ve [RUN_REPORT](../experiments/C1-04/RUN_REPORT.md). Neural eğitim ve T-C03 **NOT_RUN**; REQ-C03 kabul bağı açık, C1-04 **IN_PROGRESS / STAGE_1_COMPLETE**. C1-05 ve G1 açılmadı.

## 3 Ekim 2026 · REQ-C03 / T-C03 Aşama 2 bağı

Dondurulmuş Stage1 SHA/split/model sözleşmesi → `src/neurokinematics/neural/c104.py` loader ve iki MLP → yanlış eşleme 64/64 reddi ve 64/32 küçük öğrenme T-C03 PASS → aynı etiketli train/validation ve bütçede iki model × üç seed E-C01 → 21.600 per-row bağımsız FK ile Profile A/B, ham limit ihlali ve gerçek poz ölçümü → altı checkpoint SHA ve 60 sabit çıkarım/FK temiz ortam tekrarının PASS sonucu → [RUN_REPORT](../experiments/C1-04/stage2/RUN_REPORT.md), [acceptance](../experiments/C1-04/stage2/acceptance.json), [manifest](../experiments/C1-04/stage2/evidence-manifest.json). Her koşuda Profil A 0/3.600; **T-C03 görev testi PASS, doğrudan IK NO-GO**. Düşük başarı C1-05 E-C03 kontrollü FK kaybı denemesine [devredildi](../experiments/C1-04/stage2/NEXT_MODEL_DECISION.md); C1-05/T-C04 ve C1-06 nihai test NOT_STARTED/NOT_RUN. G1 açık.

## 8 Ekim 2026 · C1-05 Aşama 1 · Belge r20

REQ-C03/REQ-C04 → ADR-013 + dondurulmuş config/matris → girdi erişim/audit
scriptleri → 34 shard/6 checkpoint/21600 q doğrulaması + 129 regresyon ve tam
T-C01/T-C02 PASS → [C1-05 RUN_REPORT](../experiments/C1-05/RUN_REPORT.md).
Bu zincir Aşama 1 kanıtıdır; yeni training-FK/pilot/varyant deneyleri **NOT_RUN**,
T-C04 açık. C1-05 **STAGE_1_COMPLETE**; Aşama 2 açık onay bekler.


## 8 Ekim 2026 · REQ-C03 / REQ-C04 / T-C04 Aşama 2 bağı · Belge r21

**COMPLETE / T-C04 PASS; doğrudan IK NO-GO.** E-C03, E-C04 ve E-C05
ayrı etkilerle üçer eşli seed üzerinde tamamlandı: 18 model koşusu, 33.210
optimizer adımı, 64.800 validation satırı. Her koşuda Profil A/B 0/3.600;
FK yönelim hatasını azalttı, limit cezasının etkisi karma, tanh limit ihlali sıfır.
E-C06/07/08 ve Res-MLP/curriculum ön kayıtlı SKIP; dört özgün config kullanıldı.
129 regresyon, 36 yeni test, 19 yeni kaynak mutantı ve iç/dış alan FK/gradyan
kontrolleri PASS. Temiz checkout/yeni ortamda 18 checkpointten 180 çıkarım/FK
birebir tekrarlandı. Taze ortamda eğitim NOT_RUN; ağırlıklar LOCAL_ONLY,
uzak arşiv NOT_CONFIRMED. Test/10.000 benchmark mühürlü; G1 açık.

[Çalışma kaydı](../experiments/C1-05/stage2/RUN_REPORT.md), [sonuçlar](../experiments/C1-05/stage2/RESULTS.md), [karar](../experiments/C1-05/stage2/acceptance.json), [hashli C1-06 devri](../experiments/C1-05/stage2/C1-06-handoff.json).
Sonraki görev C1-06 için FK_TANH ailesinin üç seed'i araştırma adayı olarak
devredilir; C1-06 bu çalışmada başlatılmadı. Önceki tarihli kayıtlar tarihseldir.

Gereksinim → ADR-013 ve frozen config → `training_fk.py`, `physics.py`,
`c105.py` → `tests/c1_05/` ve kaynak mutantları → `stage2/domain/`,
`stage2/E-C03/`, `stage2/E-C04/`, `stage2/E-C05/` → `results-audit.json`,
`clean/witness-result.json` → T-C04 PASS, operasyonel NO-GO.


## 8 Ekim 2026 · C1-06 ön kayıt kanıtı

C1-06 **IN_PROGRESS / STAGE_1_COMPLETE; T-C05 NOT_RUN**. 322 girdi dosyası,
21 checkpoint ve 600.000 baseline satırının byte/SHA erişimi doğrulandı.
31 sentetik/negatif test ve 10 satırlık analitik smoke PASS. Üç seedli H2,
root kümeli bootstrap, tam payda ve süre sınırları ön kayıtlı. Nihai test
SEALED_NOT_RUN; Aşama 2 açık kullanıcı onayı bekler. C1-07/G1 başlamadı.
[Çalışma kaydı](../experiments/C1-06/RUN_REPORT.md). Önceki plan/tarihli kayıtlar korunmuştur.


## 9 Ekim 2026 · C1-06 nihai araştırma kapanışı

**COMPLETE / REQ-C04, REQ-C05 / T-C05 PASS; H2 REDDEDİLDİ.**
8 Ekim onaylı final kampanyası 21 checkpoint × 12.000 sorgu × beş geçişte
1.260.000 neural ölçüm ve 600.000 eşli historical baseline raw kaydını
bağımsız denetledi. Her neural modelde Profil A/B 0/12.000. Üç seedli
FK_TANH−Q farkı main, boundary, singularity ve eşit ağırlıklı zor kümede
0 yüzde puanı; %95 empirik paired bootstrap CI [0,0]. Ön kayıtlı +2 yp
zor-küme hedefi sağlanmadı; teknik araştırma kabulü pozitif H2 değildir.
59 sentetik test, 210 sabit tanık, tam-payda/sızıntı ve raw/SHA audit PASS.
Ağırlıklar ve 2.331.485.455 bayt yeni raw LOCAL_ONLY; uzak arşiv
NOT_CONFIRMED. Farklı platform süreleriyle üstünlük iddiası yoktur.
Doğrudan IK NO-GO; collision NOT_CHECKED. G1 ve v1.0.0 etiketi verilmedi.

[Çalışma kaydı](../experiments/C1-06/stage2/RUN_REPORT.md), [sonuçlar](../experiments/C1-06/stage2/RESULTS.md),
[kabul](../experiments/C1-06/stage2/final-001/acceptance.json),
[C1-07 devri](../experiments/C1-06/stage2/final-001/C1-07-handoff.json).
C1-07 girdileri hazır; görev NOT_STARTED. Tarihli önceki plan/ara kayıtlar korunur.


## 9 Ekim 2026 · C1-06R R0/R1 ve eğitim teslimi

REQ-C02 → ayrı exact CUDA overlay + mevcut FK/TCP → CPU/GPU 1086 q/dtype
ve 32 gradient q/cihaz PASS → `C1-06R/r0r1/attempt-001/runtime.json`.
REQ-C03 → 20.400 train/validation, mevcut teacher etiketlerinin tamamı
Profil B, provenance/normalizasyon PASS → `data-audit.json`. Küçük local64
61/64 ve LBFGS-r2 61/64 FAIL korunur; ayrı ölçek tanısı r3 64/64; mixed64
64/64. Bu küçük tanılar genel başarı değildir.
REQ-C04 → Q/FK × linear/tanh, aynı veri/başlangıç/bütçe, üç seed → 471
test PASS, 480-update smoke ve dört CUDA resume eşliği →
[training-freeze](../experiments/C1-06R/training-freeze.json).
REQ-C05 → eski final erişimi yok; yeni final NOT_CREATED; ana eğitim
NOT_RUN. REQ-C06/C1-07 ve G1 açık.
[Çalışma kaydı](../experiments/C1-06R/RUN_REPORT_R0R1.md),
[eğitim komutu](../experiments/C1-06R/USER_TRAINING.md).


## 9 Ekim 2026 · C1-06R round1 R4 denetimi

REQ-C03/04 → değişmeyen veri/config ile kullanıcıda 12 koşu ve 1.440.000
update → 85 SHA/envanter, epoch/seed/permutation, best/last yeniden
çıkarım PASS → [audit.json](../experiments/C1-06R/round1-analysis/audit.json).
REQ-C05 → tüm 960 validation kaydında A/B sıfır; model başına main
0/3000; ≥%95 kapısı NOT_MET → [sonuç](../experiments/C1-06R/round1-analysis/RESULTS.md).
13 yeni bozuk-kayıt/metadata testi PASS; 24 train gradyan probu ek tanıdır.
H2-R final NOT_EVALUATED, yeni final NOT_CREATED, eski H2 REJECTED
korunur. REQ-C06/C1-07/G1 açık; önce kontrollü tanı.
[Çalışma kaydı](../experiments/C1-06R/round1-analysis/RUN_REPORT.md).


## 10 Ekim 2026 · C1-06R tanı 2 / decoder düzeltmesi

REQ-C03/04 → aynı kök/kapasite/bütçe, 8 kısa hücre ve 4 optimizer takip
→ [tanı sonuçları](../experiments/C1-06R/diagnostic2/results.json): bütün
validation A0/3600. REQ-C02/05 → 20.400 provenance, 18.453 teacher ve
bağımsız FK, 474 regresyon PASS → [özet](../experiments/C1-06R/diagnostic2/summary.json).
Perfect-teacher oracle → 100/19 float32 limit taşması → ADR-016 yeni
decoder → 15.204/15.204 train ve3249/3249 validation A/B; gerçek model
başarısı değişmedi → [etki](../experiments/C1-06R/diagnostic2/project-review/precision-fix.json).
Eski C1-06/H2 ve round1 korunur. Göreli pose temsil tanısı NOT_RUN;
REQ-C06/C1-07/G1 açık, yeni final NOT_CREATED.


## 10 Ekim 2026 · C1-06R göreli pose eşli tanısı

REQ-C03/04 → ADR-017, aynı13-boyut/kapasite ve80.000 update,16 hücre
→ dört dönüşüm/normalizasyon testi ve4 önceki raw kontrol tensor eşliği
PASS → [audit](../experiments/C1-06R/diagnostic3/audit.json).
REQ-C05 → tam3600 validation paydası,16 checkpoint replay → her modelde
A/B0; göreli local medyan hata iyileşmesi ürün başarısı değildir →
[sonuç](../experiments/C1-06R/diagnostic3/RESULTS.md).
Eski sonuç/frozen girdiler ve H2 REJECTED korunur; yeni final NOT_CREATED.
Geniş train hassasiyeti çözülmedi; kapasite/optimizasyon ayrımı NOT_RUN,
REQ-C06/C1-07/G1 açık.


## 10 Ekim 2026 · C1-06R optimizer/ölçek/kapasite ayrımı

REQ-C03/04 → ADR-018, aynı2048 local göreli/residual örnekte dört koşul
→ 7 test, tanı3 referans tensor/metrik eşliği ve4 checkpoint tam replay
PASS → [audit](../experiments/C1-06R/diagnostic4/audit.json).
REQ-C05 → değişmeyen A/B, tam3600 payda → dört modelde validation A/B0,
geniş model train A3/2048 → [sonuç](../experiments/C1-06R/diagnostic4/RESULTS.md).
Kalan başarısızlık → sonuç sonrası pose/limit/eklem hata ayrımı → geniş
modelde1886 train satırı iki pose eşiğini aşıyor,19 limit dışı →
[ayrım](../experiments/C1-06R/diagnostic4/error-decomposition.json).
Frozen122/önceki kanıtlar ve H2 REJECTED korunur. Train geometrik duyarlılık
tanısı NOT_RUN; yeni final NOT_CREATED; REQ-C06/C1-07/G1 açık.


## 10 Ekim 2026 · C1-06R geometri/amaç ve veri kapsamı

REQ-C02 → ADR-019,4096 FK/Jacobian,32FD,2048 teacher → PASS;
REQ-C03/04 → sabit model gradyan/Taylor analizi ve ADR-020 eşli amaç
takibi10000 update → Q train A57/2048, POSE_A33/2048;
REQ-C05 → değişmeyen A/B ve3600 tam validation paydası → iki modelde0;
488 test ve tam reload/hash → [audit](../experiments/C1-06R/diagnostic5/audit.json).
F0-04/C1-02 veri desteği →7000 train local provenance PASS, tek local/root,
medyan nearest-root/düzeltme6,49 → [inceleme](../experiments/C1-06R/diagnostic5/sampling-review.json).
Kinematik kusur kanıtı bulunmadı; yerel öğrenme ile global genelleme ayrımı
→ [yeniden plan](../experiments/C1-06R/diagnostic5/REPLAN.md), NOT_RUN.
Önceki H2/frozen122 korunur; yeni final NOT_CREATED; REQ-C06/C1-07/G1 açık.


## 10 Ekim 2026 · C1-02R/v1 yön kapsamı

REQ-C02 → ADR-021,8192 sürümlü train-kök satırı, bağımsız FK/Jacobian,
deterministik üretim ve root/group ayrımı → PASS;
REQ-C03/04 → aynı model/amaç/bütçe ile tekrar vs yön,20000 update →
64 tekrar orijinal A64/64 fakat yeni yön0/512; yön train A105/512,3/4096;
REQ-C05 → tam3600 validation ve değişmeyen A/B → tüm modeller0;
57 test,33984 prediction audit, frozen122 → [audit](../experiments/C1-06R/diagnostic6/audit.json).
512-kök yeni yön medyanı40,10mm/8,19°→13,79mm/4,34°: sürekli iyileşme var,
ürün kabulü yok. Sonraki train-only yerel temsil ölçeği kontrolü NOT_RUN.
[Sonuç](../experiments/C1-06R/diagnostic6/RESULTS.md),
[kayıt](../experiments/C1-06R/diagnostic6/RUN_REPORT.md).
Önceki H2/frozen girdiler korunur; yeni final NOT_CREATED; REQ-C06/C1-07/G1 açık.


## 10 Ekim 2026 · Yerel ölçekleme ve ölçüt doğruluğu

REQ-C02/05 → bağımsız FK/atan2, eşik/AND/invalid testleri ve validation
root oracle3600/3600 A/B → [audit](../experiments/C1-06R/diagnostic7/audit.json),PASS;
REQ-C03/04 → ADR-022, aynı directions verisi/bütçe ile RAW/LOCAL_Z,20000update
→ train A105→259/512,3→60/4096; tüm probe/validation A/B0; NOT_MET.
64 test,33984 prediction, iki RAW tensor/metrik replay ve frozen122 PASS.
Core §6/TEST_PROTOCOL/PLAN → ürün%95 ve araştırma kapanışı ayrımı teyit;
20mm/10° açıklayıcı local duyarlılık bile RAW512%62,83; kabul eşiği değişmedi.
[Kıstas/yöntem değerlendirmesi](../experiments/C1-06R/diagnostic7/CRITERION_AND_DIRECTION_REVIEW.md),
[çalışma kaydı](../experiments/C1-06R/diagnostic7/RUN_REPORT.md).
Yerel türev/sıfır-düzeltme tanısı NOT_RUN; final NOT_CREATED; C1-07/G1 açık.

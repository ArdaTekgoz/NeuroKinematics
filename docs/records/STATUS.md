# NeuroKinematics mevcut durum

29 Eylül 2026 · Belge r16

| Bileşen | Durum | Kanıt |
|---|---|---|
| Kaynak ana rapor | Korundu | archive altındaki aynı baytlı kopya ve hash |
| Revize tasarım ve dört faz raporu | Hazır | raporlar ve docs/raporlar |
| Roadmap ve görev planları | Hazır | docs/roadmaps ve 25 görev kaydı |
| Foundations yazılımı | COMPLETE | F0-00–F0-06 PASS; G0 PASS / ACCEPTED; [RUN-20260924-F006](../../experiments/F0-06/RUN_REPORT.md) |
| F0-00 kapsam ve ortam | TAMAMLANDI | [RUN-20260918-001](../../experiments/F0-00/RUN_REPORT.md), T-F00 6/6 PASS |
| F0-01 robot modeli ve manifest | TAMAMLANDI | [RUN-20260918-002](../../experiments/F0-01/RUN_REPORT.md), T-F01 16/16 PASS |
| F0-02 bağımsız ileri kinematik | TAMAMLANDI | [RUN-20260918-003](../../experiments/F0-02/RUN_REPORT.md), T-F02 102/102 PASS; 10000 q |
| F0-03 Jacobian ve metrik | TAMAMLANDI | [RUN-20260919-001](../../experiments/F0-03/RUN_REPORT.md); 159/159 PASS; T-F03 256+21 q × 3 h; T-F04 48/48 |
| F0-04 deterministik veri fabrikası | TAMAMLANDI | [RUN-20260921-001](../../experiments/F0-04/RUN_REPORT.md); T-F05/06/07 PASS; 10000+1000+1000 kayıt |
| F0-05 sayısal baseline ve ölçüm | TAMAMLANDI | [RUN-20260924-F005](../../experiments/F0-05/RUN_REPORT.md); T-F08 16/16, 120000 ölçüm |
| F0-06 kapanış ve faz devri | COMPLETE | T-F09 PASS; temiz ortam, determinism ve kapanış testleri |
| Core; C1-01 | AKTİF; C1-01 COMPLETE / T-C00 PASS | [C1-01 kabul raporu](../../experiments/C1-01/RUN-20260928-T-C00-acceptance.md); 600.000 kayıt, verify PASS |
| Core; C1-02 | COMPLETE / T-C07 PASS / ACCEPTED | [Aşama 2 kabul raporu](../../experiments/C1-02/RUN-20260929-T-C07-acceptance.md); 24.000 çift, 34 shard, leakage ve tekrar üretim PASS |
| Core; C1-03 | IN_PROGRESS / STAGE_1_COMPLETE; T-C01/T-C02 NOT_RUN | [Aşama 1 kayıtları](../../experiments/C1-03/RUN_REPORT.md); frozen matematik/örnekleme/bağımlılık sözleşmesi, Aşama 2 açık onayı bekler |
| Hybrid ve ONNX | PLANLANDI | Ölçüm yok |
| Studio ve ikinci robot | PLANLANDI | Ölçüm yok |
| Gerçek robot ve ileri araştırma | ERTELENDİ | Ayrı kapsam gerekiyor |

Tamamlanan görevler: [F0-00](../tasks/F0-00.md), [F0-01](../tasks/F0-01.md), [F0-02](../tasks/F0-02.md), [F0-03](../tasks/F0-03.md), [F0-04](../tasks/F0-04.md), [F0-05](../tasks/F0-05.md) ve [F0-06](../tasks/F0-06.md). Exact KR 6 R900 sixx varlıkları korunarak veri fabrikası ve sayısal baseline doğrulandı. F0-06 tamamlandı; G0 PASS / ACCEPTED.

Depo yerleşimi, plan mutabakatı, açık varsayımlar ve F0-00 başlangıç sırası [FOUNDATIONS_KICKOFF](FOUNDATIONS_KICKOFF.md) kaydında açıklanır. Bu hazırlık kaydı bir Foundations görevinin kapandığı anlamına gelmez.

T-F00 kullanıcı bilgisayarında native Windows 11 x64 üzerinde çalıştırılmıştır. Bu sonuç Linux'un çalıştırıldığı, robot modelinin T-F01'i geçtiği, FK/Jacobian'ın doğrulandığı veya fiziksel robot doğruluğunun kanıtlandığı anlamına gelmez. F0-00 koşusunun başlangıç HEAD'i `9b51aefcc6c1e87a7be36c8c0b055c91144ee86d`, kapanış ve push commit'i `ae9054c514c81b5ce237f89c86a9319b202c8745`'tir; kapanış commit'i `origin/main` üzerine pushlanmıştır. Koşu sırasındaki commitlenmemiş çalışma ağacı kaydı tarihsel olarak korunur; kullanıcıya ait geçici Word lock dosyası çalışma kapsamına alınmamıştır.

T-F01 aynı Windows hostunda çalıştırılmış, 16/16 PASS vermiştir. Uygulama commit'i `4048c428afceaab4418d6107897dcd36c2d48f33`'tür. Linux yürütmesi, fiziksel doğruluk, collision/safety ve FK/Jacobian doğrulaması bu sonuçtan çıkarılamaz.

T-F02 uygulama commit'i `d92dd213bb96f8932bd0019541dd13dd7365afaf`'tır. Maksimum konum farkı `4.75098925995612e-16 m`, rotation Frobenius farkı `9.159602786276758e-16`; iki `1e-9` eşiği de geçti. Aşım/nonfinite/geçersiz sonuç sıfır. PCG64 seed `20260918`, sample SHA-256 `8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101`. F0-02 kapanış koşusunda F0-00 6/6, F0-01 16/16, F0-02 102/102 PASS. O görev kapsamında Linux, Jacobian, veri fabrikası, IK/ML ve fiziksel güvenlik doğrulaması yapılmadı. Kaynak raporlar ve kullanıcıya ait geçici dosya korundu.


F0-03 uygulama commit'i: 981f6143ce38574021edac7373586976cf97bdf4. Seed 20260919; sample
`678eb4286863026880792ef0cc3c0a9d4f92e16f85b1aa009705cbf0b59b26e7`.
Ana h=1e-6 maksimum normalize fark 1.7676058530094515e-10 ≤1e-5;
üç h ve üç yöntem çifti geçti. 12 mutasyon yakalandı, T-F04 48/48 PASS.
F0-00 6/6, F0-01 16/16, F0-02 102/102 yeniden geçti. İlk görev metnindeki
62 karakterlik TCP hash yazım hatası kullanıcı yetkisiyle düzeltildi; varlık
değişikliği veya F0-02 regresyonu yok. Linux ve fiziksel güvenlik doğrulanmadı.

F0-04 uygulama commit'i `16010d518c24400f6c6d43a2459456dd822f34a8`.
PCG64 seedler 20260920–20260924; config/schema hashleri RUN_REPORT'tadır.
10000 main, 1000 boundary, 1000 singularity kaydı üretildi. Dataset content
SHA-256 `5cb4e64580ecaf99afd11b3c8b98e06ed00c712e83bf2d9ee4d8c3acd58173fe`;
iki temiz üretimde 12 file/content hash birebir eşti. Main split 7000/1500/1500;
üç grup kesişimi ve çapraz-split q tekrarı sıfır. 17/17 mutasyon ve önceki faz
regresyonları geçti. Linux, fiziksel doğruluk, collision ve safety doğrulanmadı.

## Tarihsel kayıt: 24 Eylül 2026 · Belge r8 · F0-05 durumu

F0-05 **PASS / TAMAMLANDI**: [RUN-20260924-F005](../../experiments/F0-05/RUN_REPORT.md).
Uygulama commit'i `3e55954`.
Aşama 2 açık kullanıcı onayı `stage2-approval.json` dosyasında. 12.000 bağımsız
query ve 120.000 ölçüm satırı doğrulandı; iki query üretimi aynı hash'i verdi;
F0-04 ile exact q tekrarı ve grup kesişimi sıfır. F0-05 unit 127/127,
T-F08 16/16, mutasyon 32/32; F0-00–F0-04 regresyonları 6/16/102/159/39 PASS.
Profile B 50 ms deadline başarısı %68,627; wide başlangıçlar daha zordur.
Kanıt checksum denetimi ve büyük JSONL hashleri PASS. Foundations devam ediyor;
F0-06 sıradaki iştir, **başlatılmadı**. G0 **açık**. Önceki görevlerin tarihsel
devir cümleleri kendi tarihlerine ait kayıtlar olarak korunur.

## 24 Eylül 2026 · F0-06 kapanış kaydı

F0-06 COMPLETE; Foundations COMPLETE; G0 PASS / ACCEPTED; Core READY / NOT_STARTED.
Kanıt: [RUN-20260924-F006](../../experiments/F0-06/RUN_REPORT.md); [G0 kararı](../../experiments/F0-06/G0_DECISION.md),
[Core devri](../../experiments/F0-06/CORE_HANDOFF.md).
Temiz native Windows worktree'de locked install ve 497 regresyon testi geçti;
F0-06 26/26 test, iki gerçek 384 veri/384 query/768 benchmark satırlı koşu PASS.
T-F09 PASS. Önceden dondurulmuş eşikler, seedler ve üretim configleri korunmuştur.
Linux NOT_RUN; fiziksel robot, kalibrasyon, collision/safety doğrulanmamıştır.
C1-01, C1-02, C1-03: NOT_STARTED; G0 sonrası uygun. Bu görevde Core başlatılmadı.

- [x] Gereksinim → değişiklik → test → kanıt bağı kaydedildi.
- [x] T-F00–T-F09 ve temiz ortam kanıtları doğrulandı.
- [x] G0 kabulü ve açık sınırlamalar kaydedildi.
- [x] Hashli Core girdileri ve tekrar komutları devredildi.

Önceki plan ve tarihli görev kayıtları tarihsel bağlamıyla korunur.

## 24 Eylül 2026 · C1-01 Aşama 1 kaydı

F0-06 COMPLETE ve G0 PASS / ACCEPTED kapısından sonra Core aktifleştirildi. C1-01 Stage 1 inceleme, kaynak pinleri, Ubuntu 24.04/Jazzy ortak platform kararı, config ve SHA manifesti hazırlandı. C1-01 genel durumu `IN_PROGRESS / STAGE_1_COMPLETE`; T-C00 `NOT_RUN`. C1-02/03 ve diğer Core işleri `NOT_STARTED`. Linux ve harici solver kurulumu `NOT_RUN`; performans `NOT_MEASURED`. [Çalışma kaydı](../../experiments/C1-01/RUN_REPORT.md) ve [platform ADR](../adr/ADR-007-core-harici-baseline-platformu.md) ayrıntıları verir.

## 25 Eylül 2026 · C1-01 Aşama 2 devam durumu · Belge r11

C1-01 `IN_PROGRESS / STAGE_2_IMPLEMENTING`; kullanıcı [açık onay](../../experiments/C1-01/stage2-approval.json) verdi. Ortak sözleşme, DLS IPC, sonuç doğrulama ve MoveIt C++ worker kaynağı çalışma ağacında; yerel adapter testleri 8/8 PASS. Docker Desktop/WSL 2 kullanıcı ortamı henüz doğrulanmadı; ROS/C++ build, harici solver smoke ve T-C00 `NOT_RUN`. Beş kolay solver smoke PASS olmadan tam benchmark çalıştırılmayacak. Bu tarihli kayıt yukarıdaki tarihsel Stage 1 tablosunu ve Foundations kararını değiştirmez.

## 26 Eylül 2026 · Docker/WSL başlangıç kontrolü

Kullanıcının Docker Linux amd64 server, WSL 2, hello-world ve Ubuntu 24.04 x86_64 çıktıları PASS. C1-01 build/lock scriptleri hazırlandı; gerçek source build/lock audit, harici smoke ve T-C00 NOT_RUN. Kullanıcının build çıktısı bekleniyor; [komut ve PNG kanıtları](../../experiments/C1-01/COMMANDS.md).

2026-09-26 C1-01 devam: kullanıcı Docker source build (10 paket) ve Linux ortam lock PASS; image/Pixi lock/adet read-only kontrolü PASS. Linux adapter ve beş solver smoke NOT_RUN; T-C00 NOT_RUN, STAGE_2_IMPLEMENTING sürüyor. Kanıt: experiments/C1-01/docker-build.log, environment-lock.json, docker-build-evidence/.

2026-09-26 C1-01 Linux adapter: 8/8 PASS, JUnit failures/errors/skipped=0 (linux-adapter-tests.xml). Beş solver entegrasyon smoke sıradaki adım / NOT_RUN; T-C00 NOT_RUN, STAGE_2_IMPLEMENTING.

2026-09-26 C1-01 beş solver Linux smoke PASS (40 kayıt), ham hashler PASS. Ek Windows FK yeniden doğrulamasında orientation_error_deg uyuşmazlığı; aynı image Linux offline kontrol sırada. T-C00 NOT_RUN / STAGE_2_IMPLEMENTING.

2026-09-26 C1-01: kullanıcı aynı Linux image offline verify-results beş yöntem PASS (8'er kayıt); Windows F0-05/F0-06 kritik regresyon 201 PASS. Linux kritik regresyon ve küçük 10/50ms uçtan uca smoke sırada; T-C00 NOT_RUN.

2026-09-26 Linux kritik regresyon 208 PASS / 1 FAIL: F0-06 eski Windows FK orientation_error_deg yeniden hesap uyuşmazlığı. Benchmark gate açık değil; read-only Linux sayısal tanı sırada. F0 ve eşikler değişmedi.

2026-09-26 ADR-008: tarihsel Windows residual testi Linux FAIL olarak korunur; native C1-01 ek mutation testi Windows PASS, Linux kapsamlı koşu sırada / NOT_RUN. Sayısal kabul eşikleri değişmedi.

## 26 Eylül 2026 · Pilot öncesi durum
Linux scoped kritik regresyon 209 PASS / 1 deselected (ADR-008); build/lock ve beş solver smoke/offline PASS. Küçük120-kayıt 10/50ms pilot scripti hazır / NOT_RUN; T-C00600000 deneme NOT_RUN. Sonraki sıra ve Stage2 commit/push kapanış planı experiments/C1-01/COMMANDS.md içinde. Görev hâlâ IN_PROGRESS / STAGE_2_IMPLEMENTING; Core fazı tamamlanmadı.

2026-09-26 pilot120kayıt PASS, raw hashler PASS; full öncesi runner cold-restart kusuru bulundu/düzeltildi. Yerel adapter/restart10PASS. Image rebuild + yenilenen Linux210test/smoke/pilot NOT_RUN; T-C00 NOT_RUN.

2026-09-26 warm restart düzeltmeli image build/lock PASS (8393248c1e48...); eski kanıt attempts/20260926-135117-390/. Yeni image Linux kritik210test/smoke/pilot NOT_RUN; T-C00 NOT_RUN.

2026-09-26 yeni image kritik210PASS/1deselected; eski smoke/pilot before-warmup-fix adlarıyla korunur. Yeni image smoke/pilot sıradaNOT_RUN; T-C00NOT_RUN.

2026-09-26 yeni image beş smokePASS,40kayıt/hashPASS; her yöntem20warmup/1launch/0restart. Yeni120kayıt pilot sıradaNOT_RUN; fullT-C00NOT_RUN.

2026-09-26 yeni pilot120PASS; global14launch/280warmup(12restart) doğrulandı. Full600000ölçüm launch script hazır/NOT_RUN. C1-01STAGE_2_IMPLEMENTING, T-C00henüz kabul edilmedi.

2026-09-26 T-C00INCOMPLETE:4yöntem120000er MEASURED_UNVERIFIED/global115INCOMPLETE; tümrawhashPASS. IPCrestart readerqueue/communicate kusuru düzeltildi, yerel11PASS. Aynıimage4raw doğrulama sıradaNOT_RUN; globalfixLinuxNOT_RUN. C1-01IN_PROGRESS.

2026-09-26 aynıimage4yöntem480000raw offlinePASS; PARTIAL_VERIFIED/full_acceptancefalse. Global115INCOMPLETE. IPCfixrebuild/211test/restartstresssıradaNOT_RUN; görevkapanmadı.

2026-09-26 IPCfiximage8fa7616b... build/lockPASS; önceki480000verifiedve115incompletekanıt155030archiveiçinde. Yeni211testNOT_RUN. Kullanıcı isteği: yeniANA smoke/benchmarkkodundanönceDUR/onay/ajan değişimi; mevcutregresyoniledevam.

2026-09-26 IPCfiximageLinux211PASS/1deselected(JUnit0hata). Mevcutglobal500kayıtrestartstresssıradaNOT_RUN; sonrakiANA testkodundanönce kullanıcıonayı/ajandeğişimi beklenir. T-C00kapanmadı.

2026-09-26 IPCfixglobalrestartstress121/500INCOMPLETE;aynıINVALID_OUTPUTreadyhatası. Öncekiqueuefixgerçekduruşugidermedi. Aynıimagebozuksatırbyte/stdouttanısı sıradaNOT_RUN; yeniANA smoke/benchmarkbekletilir.

2026-09-26 gerçekbozukreadyFastDDSRTPS_TRANSPORT_SHMsegmenthatastdoutolarakyakalandı(251SHMgiriş). ADR-009kontrollüUDPv4deneyiöneri; aynıimage500restartstressNOT_RUN. ANA testkodunageçilmiyor.

2026-09-26 UDPv4restartstress500PASS/256restart/5160warmup;fastrtpsSHM0,raw/lockbağPASS. KullanıcıisteğiyleANA testkodundanönceDUR:ajandeğişimi+açıkonaybeklenir. Devir:experiments/C1-01/MAIN_TEST_HANDOFF.md. C1-01IN_PROGRESS/T-C00INCOMPLETE.

## 27 Eylül 2026 · C1-01 ana koşu hazırlığı

Kullanıcı ana test hazırlığını onayladı. Durum `IN_PROGRESS / STAGE_2_IMPLEMENTING`; önceki T-C00 `INCOMPLETE / PARTIAL_VERIFIED`. UDPv4 ortak runtime ve hash bağlı prepare/smoke/pilot/full/verify akışı hazırlandı. Kalan süre aktarımı kusuru nedeniyle yalnız native worker yeniden derlenecek. Yeni Linux build ve ana ölçümler `NOT_RUN`; eski 480.000 kayıt ayrı korunur. Kapanış, Aşama 2 commit/push ve sonraki Core görevi henüz yapılmadı. [Güncel çalışma kaydı](../../experiments/C1-01/RUN-20260927-main-preparation.md).

27 Eylül devam: yeni native worker build ve build içi dependency/input audit PASS; image `00a76905...`. Final ortam lock çağrısı Pixi editable reinstall/ağ hatasıyla durdu. `PIXI_NO_INSTALL=true` ve mevcut image'da `-ResumeLock` düzeltmesi hazır, yerel 23 session testi ve PowerShell kilit yayınlama negatif kontrolleri PASS. Linux lock kurtarma ve yeni ana testler NOT_RUN; görev IN_PROGRESS.

27 Eylül runtime lock kurtarma PASS: image `00a76905...`, 691 paket, CPU 0/1; ortam lock SHA `9369f45ba52acd6962244bf56ced425c0d895e00771bb5c672895196c6e66990`. Kaydedilmiş lock/audit/image/input bağları salt okunur PASS. Yeni Linux prepare/protokol kontrolü sıradaki adım, NOT_RUN; C1-01 IN_PROGRESS.

27 Eylül Linux prepare PASS: kullanıcı koşusunda 258 test geçti, ADR-008 kapsamında 1 deselected; beş gerçek worker expired-request probe PASS. `udp-v1` runtime SHA `4ececbf3...`; gate'in 10 dosyası, test kimlikleri ve snapshot bağları salt okunur PASS (`udp-v1-prepare-evidence-check.json`). Sırada aynı session smoke → pilot → full → verify. Yeni ölçümler NOT_RUN; C1-01 IN_PROGRESS, kabul ve Aşama 2 commit/push bekliyor.

27 Eylül smoke çağrısı ölçüm öncesi runtime drift kontrolünde FAIL. Snapshot kaynak/test/script hashleri tekrar PASS; değişen diğer alan henüz bilinmiyor. Salt okunur `diagnose_c101_runtime.ps1` hazır; gerçek Docker tanısı NOT_RUN. Prepare PASS korunuyor, smoke ölçümleri başlamadı; C1-01 IN_PROGRESS.

27 Eylül kullanıcı Docker runtime tanısı MATCH / sıfır fark; kayıtlı SHA `4ececbf3...`. Önceki drift nedeni bilinmiyor. Prepare PASS, smoke klasörü/aktif kilit yok; aynı session smoke tekrar denemesi sırada. Build/prepare veya kabul kontrolü değişmedi; C1-01 IN_PROGRESS.

27 Eylül ikinci smoke başlangıcı da runtime drift FAIL; ölçüm başlamadı. Capture-only MATCH kök neden kanıtı değil. Tanı gerçek kayıtlı main başlangıcında runtime yakalayıp solver dispatchinden önce duracak şekilde geliştirildi; yerel interception/cleanup kontrolü PASS, Docker tanısı NOT_RUN. Prepare PASS korunuyor, yeni ölçümler bekliyor.

27 Eylül başlangıç tanısı DRIFT: yalnız host MemTotal 8 kB farkı. ADR-010 uygulanarak kaynak tavanı 8 GiB / sıfır swap olarak cgroup v2 ile kilitlendi; host RAM gözlemleri gate'e taşındı. Yerel 29 test, dört mock launcher senaryosu ve 52 frozen-contract kontrolü PASS. Yeni build yok; `udp-v2` prepare sırada NOT_RUN (264 PASS / 1 deselected bekleniyor). `udp-v1` kanıtları korunuyor, C1-01 IN_PROGRESS.

27 Eylül `udp-v2` Linux prepare PASS: 264 test / 1 ADR-008 deselected, beş worker protokolü, 8 GiB cgroup tavanı / swap0 doğrulandı. Runtime SHA `ad1bd5b2...`; 10 kanıt dosyası, test kimlikleri ve snapshot bağları salt okunur PASS (`udp-v2-prepare-evidence-check.json`). Sırada aynı session smoke; yeni smoke/pilot/full/verify NOT_RUN, C1-01 IN_PROGRESS.

27 Eylül `udp-v2` smoke PASS: beş yöntem × sekiz = 40 SUCCESS kaydı; yöntem başına 20 warmup / bir launch / sıfır restart. Linux bağımsız doğrulama gate'i ve yerel hash/runtime/prepare bağı PASS (`udp-v2-smoke-evidence-check.json`). Sırada 120 kayıt pilot; pilot/full/verify NOT_RUN, görev IN_PROGRESS.

27 Eylül `udp-v2` pilot PASS: 120 kayıt, beş yöntem 24'er; fatal altyapı hatası yok. Runtime/gate/ham kanıt/sıra bağları PASS. Global 13 launch / 260 warmup / 11 restart; yalnız global kaba doğrusal süre yaklaşık 19 saat, güvenilir ETA değil. Sırada 600.000 ölçüm full; full/verify NOT_RUN, C1-01 IN_PROGRESS, commit/push kabul sonrası.

28 Eylül `udp-v2` full ölçüm tamamlandı: 600.000 kayıt, beş yöntem × 120.000, durum MEASURED_UNVERIFIED. Ham dosyaların SHA/bayt/satır, özet ve gate bağları salt okunur PASS (`udp-v2-full-evidence-check.json`); bağımsız satır/FK doğrulaması NOT_RUN. Global 60.225 restart/1.204.540 warmup; full 18,47 saat. Sırada verify, ardından kabul ve Aşama 2 commit/push; C1-01 IN_PROGRESS.

## 28 Eylül 2026 · C1-01 T-C00 kabulü

Kullanıcının `udp-v2` offline verify koşusu beş yöntemde PASS: 600.000/600.000 satır, her yöntemde 12.000 farklı sorgu ve 120.000 kayıt, fatal altyapı hatası 0. Full→verify gate ve beş summary hash bağı yerelde salt okunur PASS (`udp-v2-verify-evidence-check.json`). REQ-C01 / T-C00 **PASS / ACCEPTED**; C1-01 **COMPLETE**. [Çalışma/kabul kaydı](../../experiments/C1-01/RUN-20260928-T-C00-acceptance.md). Full raw yaklaşık 1,1 GB LOCAL_ONLY, ayrı uzak arşiv NOT_CONFIRMED; bu sınır raporda açık. Etkin insan emeği NOT_MEASURED. Aşama 2 uygulama ve kabul commit'i `207bf734de6536e2b590e922930df1547fdc29f1` `origin/main`'e push edildi; sonraki C1-02 NOT_STARTED, G1/Core faz kabulü yapılmadı.

## 29 Eylül 2026 · C1-02 kabulü · Belge r15

Kullanıcı Aşama 2'yi açıkça onayladı. Dondurulmuş 43 girdi hash'i PASS; 90 wide pilotunda kaynak sınırı ve teacher raporu PASS. İki temiz üretim 24.000'er çift ve 34'er NPZ shard verdi; canonical veri SHA-256 `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`, 34/34 shard eş. T-C07 7/7, mutasyon 12/12, seçili arayüz regresyonu 42/42 PASS; leakage/normalizasyon/benchmark soy denetimi PASS. 2.281 wide teacher etiketi eksik, test envanterinden çıkarılmadı; etiketli train modu %55,25 local / %44,75 wide. [Kabul raporu](../../experiments/C1-02/RUN-20260929-T-C07-acceptance.md) ve [acceptance.json](../../experiments/C1-02/acceptance.json). REQ-C03 / T-C07 **PASS / ACCEPTED**, C1-02 **COMPLETE**. Büyük raw veri LOCAL_ONLY; uzak arşiv NOT_CONFIRMED, insan emeği NOT_MEASURED. Sıradaki C1-03 NOT_STARTED; G1/Core kabulü yapılmadı.

## 29 Eylül 2026 · C1-03 Aşama 1 · Belge r16

C1-03 **IN_PROGRESS / STAGE_1_COMPLETE**; T-C01/T-C02 **NOT_RUN**. Mevcut parser üzerine küçük Torch FK yaklaşımı, torch 2.10.0+cpu hashli Windows overlay, 1086 q ve 32 bağımsız gradient q sözleşmesi donduruldu. [RUN_REPORT](../../experiments/C1-03/RUN_REPORT.md), [config](../../experiments/C1-03/config.json), [ADR-011](../adr/ADR-011-c103-torch-fk.md). Yalnız statik hash/yapı/örnekleme denetimi yapıldı; Torch kurulmadı, FK/gradyan uygulanmadı veya ölçülmedi. Açık Aşama 2 onayı bekleniyor; C1-04 ve G1 başlamadı.

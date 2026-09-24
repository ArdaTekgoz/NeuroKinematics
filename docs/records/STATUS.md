# NeuroKinematics mevcut durum

24 Eylül 2026 · Belge r10

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
| Core; C1-01 | AKTİF; IN_PROGRESS / STAGE_1_COMPLETE | [C1-01 Stage 1](../../experiments/C1-01/STAGE1_REVIEW.md); harici solver ve T-C00 NOT_RUN |
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

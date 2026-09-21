# NeuroKinematics mevcut durum

21 Eylül 2026 · Belge r7

| Bileşen | Durum | Kanıt |
|---|---|---|
| Kaynak ana rapor | Korundu | archive altındaki aynı baytlı kopya ve hash |
| Revize tasarım ve dört faz raporu | Hazır | raporlar ve docs/raporlar |
| Roadmap ve görev planları | Hazır | docs/roadmaps ve 25 görev kaydı |
| Foundations yazılımı | DEVAM EDİYOR | F0-00–F0-04 PASS; F0-05'e geçiş hazır; G0 kapanmadı |
| F0-00 kapsam ve ortam | TAMAMLANDI | [RUN-20260918-001](../../experiments/F0-00/RUN_REPORT.md), T-F00 6/6 PASS |
| F0-01 robot modeli ve manifest | TAMAMLANDI | [RUN-20260918-002](../../experiments/F0-01/RUN_REPORT.md), T-F01 16/16 PASS |
| F0-02 bağımsız ileri kinematik | TAMAMLANDI | [RUN-20260918-003](../../experiments/F0-02/RUN_REPORT.md), T-F02 102/102 PASS; 10000 q |
| F0-03 Jacobian ve metrik | TAMAMLANDI | [RUN-20260919-001](../../experiments/F0-03/RUN_REPORT.md); 159/159 PASS; T-F03 256+21 q × 3 h; T-F04 48/48 |
| F0-04 deterministik veri fabrikası | TAMAMLANDI | [RUN-20260921-001](../../experiments/F0-04/RUN_REPORT.md); T-F05/06/07 PASS; 10000+1000+1000 kayıt |
| Core eğitimi ve benchmark | PLANLANDI | Ölçüm yok |
| Hybrid ve ONNX | PLANLANDI | Ölçüm yok |
| Studio ve ikinci robot | PLANLANDI | Ölçüm yok |
| Gerçek robot ve ileri araştırma | ERTELENDİ | Ayrı kapsam gerekiyor |

Tamamlanan görevler: [F0-00](../tasks/F0-00.md), [F0-01](../tasks/F0-01.md), [F0-02](../tasks/F0-02.md), [F0-03](../tasks/F0-03.md) ve [F0-04](../tasks/F0-04.md). Exact KR 6 R900 sixx varlıkları korunarak deterministik veri fabrikası doğrulandı. F0-05'e geçiş hazır; F0-05 başlatılmadı.

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

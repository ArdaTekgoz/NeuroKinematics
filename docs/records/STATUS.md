# NeuroKinematics mevcut durum

18 Eylül 2026 · Belge r4

| Bileşen | Durum | Kanıt |
|---|---|---|
| Kaynak ana rapor | Korundu | archive altındaki aynı baytlı kopya ve hash |
| Revize tasarım ve dört faz raporu | Hazır | raporlar ve docs/raporlar |
| Roadmap ve görev planları | Hazır | docs/roadmaps ve 25 görev kaydı |
| Foundations yazılımı | DEVAM EDİYOR | F0-00–F0-02 PASS; F0-03'e geçiş hazır; G0 kapanmadı |
| F0-00 kapsam ve ortam | TAMAMLANDI | [RUN-20260918-001](../../experiments/F0-00/RUN_REPORT.md), T-F00 6/6 PASS |
| F0-01 robot modeli ve manifest | TAMAMLANDI | [RUN-20260918-002](../../experiments/F0-01/RUN_REPORT.md), T-F01 16/16 PASS |
| F0-02 bağımsız ileri kinematik | TAMAMLANDI | [RUN-20260918-003](../../experiments/F0-02/RUN_REPORT.md), T-F02 102/102 PASS; 10000 q |
| F0-03 Jacobian ve metrik | PLANLANDI | Geçiş hazır; başlatılmadı |
| Core eğitimi ve benchmark | PLANLANDI | Ölçüm yok |
| Hybrid ve ONNX | PLANLANDI | Ölçüm yok |
| Studio ve ikinci robot | PLANLANDI | Ölçüm yok |
| Gerçek robot ve ileri araştırma | ERTELENDİ | Ayrı kapsam gerekiyor |

Tamamlanan görevler: [F0-00](../tasks/F0-00.md), [F0-01](../tasks/F0-01.md) ve [F0-02](../tasks/F0-02.md). Exact KR 6 R900 sixx varlıkları korunarak bağımsız XML/NumPy FK ile Pinocchio referansı 10000 float64 q üzerinde doğrulandı. Sıradaki görev F0-03'tür; bu çalışmada başlatılmadı.

Depo yerleşimi, plan mutabakatı, açık varsayımlar ve F0-00 başlangıç sırası [FOUNDATIONS_KICKOFF](FOUNDATIONS_KICKOFF.md) kaydında açıklanır. Bu hazırlık kaydı bir Foundations görevinin kapandığı anlamına gelmez.

T-F00 kullanıcı bilgisayarında native Windows 11 x64 üzerinde çalıştırılmıştır. Bu sonuç Linux'un çalıştırıldığı, robot modelinin T-F01'i geçtiği, FK/Jacobian'ın doğrulandığı veya fiziksel robot doğruluğunun kanıtlandığı anlamına gelmez. F0-00 koşusunun başlangıç HEAD'i `9b51aefcc6c1e87a7be36c8c0b055c91144ee86d`, kapanış ve push commit'i `ae9054c514c81b5ce237f89c86a9319b202c8745`'tir; kapanış commit'i `origin/main` üzerine pushlanmıştır. Koşu sırasındaki commitlenmemiş çalışma ağacı kaydı tarihsel olarak korunur; kullanıcıya ait geçici Word lock dosyası çalışma kapsamına alınmamıştır.

T-F01 aynı Windows hostunda çalıştırılmış, 16/16 PASS vermiştir. Uygulama commit'i `4048c428afceaab4418d6107897dcd36c2d48f33`'tür. Linux yürütmesi, fiziksel doğruluk, collision/safety ve FK/Jacobian doğrulaması bu sonuçtan çıkarılamaz.

T-F02 uygulama commit'i `d92dd213bb96f8932bd0019541dd13dd7365afaf`'tır. Maksimum konum farkı `4.75098925995612e-16 m`, rotation Frobenius farkı `9.159602786276758e-16`; iki `1e-9` eşiği de geçti. Aşım/nonfinite/geçersiz sonuç sıfır. PCG64 seed `20260918`, sample SHA-256 `8fb7e88758aa841310ae4d665d76d00a4488a5b79217ca4d5c80a825715c7101`. Son koşuda F0-00 6/6, F0-01 16/16, F0-02 102/102 PASS. Linux, Jacobian, veri fabrikası, IK/ML ve fiziksel güvenlik doğrulaması yapılmadı. Kaynak raporlar ve kullanıcıya ait geçici dosya korundu.

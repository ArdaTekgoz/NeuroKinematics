# NeuroKinematics mevcut durum

18 Eylül 2026 · Belge r2

| Bileşen | Durum | Kanıt |
|---|---|---|
| Kaynak ana rapor | Korundu | archive altındaki aynı baytlı kopya ve hash |
| Revize tasarım ve dört faz raporu | Hazır | raporlar ve docs/raporlar |
| Roadmap ve görev planları | Hazır | docs/roadmaps ve 25 görev kaydı |
| Foundations yazılımı | DEVAM EDİYOR | F0-00 PASS; F0-01 sırada |
| F0-00 kapsam ve ortam | TAMAMLANDI | [RUN-20260918-001](../../experiments/F0-00/RUN_REPORT.md), T-F00 6/6 PASS |
| Core eğitimi ve benchmark | PLANLANDI | Ölçüm yok |
| Hybrid ve ONNX | PLANLANDI | Ölçüm yok |
| Studio ve ikinci robot | PLANLANDI | Ölçüm yok |
| Gerçek robot ve ileri araştırma | ERTELENDİ | Ayrı kapsam gerekiyor |

Tamamlanan ilk görev: [F0-00](../tasks/F0-00.md). Kapsam ve ortam sözleşmesi, native Windows/Pixi platform kararı, iki platformlu dependency lock'u ve KR 6 R900 sixx girdi kilidi oluşturuldu. Sıradaki görev [F0-01 robot modeli ve manifesttir](../tasks/F0-01.md); F0-02 ancak onun kabulünden sonra başlar.

Depo yerleşimi, plan mutabakatı, açık varsayımlar ve F0-00 başlangıç sırası [FOUNDATIONS_KICKOFF](FOUNDATIONS_KICKOFF.md) kaydında açıklanır. Bu hazırlık kaydı bir Foundations görevinin kapandığı anlamına gelmez.

T-F00 kullanıcı bilgisayarında native Windows 11 x64 üzerinde çalıştırılmıştır. Bu sonuç Linux'un çalıştırıldığı, robot modelinin T-F01'i geçtiği, FK/Jacobian'ın doğrulandığı veya fiziksel robot doğruluğunun kanıtlandığı anlamına gelmez. Çalışma çıktıları HEAD `9b51aef` üzerinde henüz commitlenmemiştir; kullanıcıya ait geçici Word lock dosyası çalışma kapsamına alınmamıştır.

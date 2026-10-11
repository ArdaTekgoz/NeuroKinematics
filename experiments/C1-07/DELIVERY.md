# Core/G1 teslim paketi

11 Ekim 2026. Nihai karar [G1_DECISION](G1_DECISION.md);
makinece okunabilir durum [acceptance.json](acceptance.json).

| Amaç | Dosya |
|---|---|
| Akademik yöntem ve negatif sonuç | [Kaynak rapor](../../docs/research/C1-06_NEGATIVE_RESULTS.md), [PDF](../../docs/publication/core/Core_Negatif_Sonuc_Raporu.pdf) |
| Modeli doğru yorumlama | [MODEL_CARD](MODEL_CARD.md), [HYBRID_HANDOFF](HYBRID_HANDOFF.json) |
| Temiz tekrar kanıtı | [complete](closure/clean/complete.json), [witness](closure/clean/witness-result.json), [loglar](closure/clean/commands/) |
| Gereksinim, değişiklik, test, kanıt | [RUN_REPORT](closure/RUN_REPORT.md), [integrity](closure/integrity.json) |
| Ara verme ve geri dönüş | [CORE_RESUME](../../docs/records/CORE_RESUME.md) |
| Paylaşılabilir analiz | [LinkedIn taslağı](../../docs/publication/core/LINKEDIN.md), [grafik verisi](../../docs/publication/core/figure-data.csv), [kaynak SHA](../../docs/publication/core/evidence-index.json) |
| Büyük dosyaların saklanması | [arşiv makbuzu](closure/archive-receipt.json), [607 dosya envanteri](closure/archive-inventory.json), [restore makbuzu](closure/restore-receipt.json) |
| Teslim bütünlüğü | DELIVERY_SHA256SUMS; yollar depo köküne göredir |

SHA256SUMS yalnız önceki hazırlığın 16 dosyalık dondurulmuş kaydıdır;
değiştirilmedi. DELIVERY_SHA256SUMS nihai paketi ayrıca kapsar; kendisini
kapsamaz. Git commit kaynak/tarihçeyi, yerel research-artifacts.zip büyük
baytları korur. Son commit/push ve repository.bundle doğrulaması arşiv
klasöründeki git-delivery.json dosyasındadır; kendi commit SHA'sını içeren
dosyayı aynı commit'e ekleme döngüsünden kaçınmak için Git dışındadır.

G1 araştırma PASS; ürün NOT_MET; H2 REJECTED; H1 NOT_MEASURED.
Araştırma başarısızlığının belgelenmesi robotik ürünün hazır olduğu anlamına
gelmez. Uzak büyük dosya arşivi ve bağımsız aygıt yedeği NOT_CONFIRMED.

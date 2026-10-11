# NeuroKinematics depo çalışma kuralları

Bu dosya depo kökünden başlayan bütün çalışma alanı için geçerlidir. Daha alt klasörde ek bir `AGENTS.md` bulunursa, o dosyanın daha dar kapsamlı ve çelişmeyen kuralları ayrıca uygulanır.

## Faz ve görev yönetimi

- Güncel çalışma girişi `START_HERE.md`, ayrıntılı devir `docs/records/PRE_HYBRID_HANDOFF.md`, kanonik plan `docs/roadmaps/MASTER_ROADMAP.md` r3'tür (11 Ekim 2026).
- G0 ve G1 araştırma kabulü PASS; C1-06S robot-özel geometrik IK PASS. Hybrid uygulaması henüz başlamadı; ilk görev H2-01'dir. Foundations çıktıları değişmez girdidir.
- C1-06S yeni neural model değildir. Doğrudan neural ürün hedefi NOT_MET, H1 NOT_MEASURED; yeni eğitim H2-N1/H2-N2'de planlıdır. G2 mühendislik kabulü ile G-N neural katkı kabulünü ayrı tut.
- C1 araştırmasındaki reddedilen H2 hipotezini Hybrid fazının H2 görevleriyle karıştırma. Eski neural launcher ile otomatik eğitim başlatma.
- Görülmüş C1/C1-06S final verisini yeni model seçimi veya yeni bağımsız final testi için kullanma. Arşivde sealed veri yalnız opak bayt olarak korunur.
- Tarihsel checksumlar değişmez. Revize edilen giriş belgelerinin tarihsel kopyaları `experiments/PRE-HYBRID-20261011/historical-snapshots.json` ile açıkça eşlenir; hash eşikleri gevşetilmez.
- Görevler roadmap bağımlılık sırasıyla yürütülür.
- Her görevden önce görev tanımı, gereklilikler ve kabul ölçütleri okunur.
- Her görev için gereksinim -> değişiklik -> test -> kanıt ilişkisi korunur.
- Faz kabul ölçütleri karşılanmadan sonraki faz başlatılmaz.
- Kabul eşikleri başarısız sonucu başarılı göstermek için düşürülmez.
- Kapsam veya mimari değişikliği gerekirse gerekçesi ADR ile kaydedilir.
- Orijinal kaynak raporlar korunur.
- Çalıştırılmayan test, ölçülmeyen performans ve doğrulanmayan varsayım açıkça belirtilir.
- Yazılım sürümü ve doküman revizyonu ayrı takip edilir.

## Çalışma ve kanıt kayıtları

- Her anlamlı görev sonunda `docs/templates/RUN_REPORT.md` şablonu kullanılarak bir Markdown çalışma kaydı hazırlanır.
- Çalışma kaydı yapılan değişiklikleri, çalıştırılan komutları, test sonuçlarını, kanıt yollarını, açık sorunları ve sonraki adımı içerir.
- Görev durumu, `docs/records/STATUS.md` ve izlenebilirlik kayıtları gerektiğinde birlikte güncellenir.
- İlgisiz kullanıcı değişiklikleri commit kapsamına alınmaz.
- Gizli bilgiler, yerel sanal ortamlar, üretilmiş büyük veri kümeleri ve model ağırlıkları normal Git takibine eklenmez.
- Geri döndürülemez Git işlemleri ve force push yapılmaz.

## Teknik sınırlar

- İlk aşamada robot modeli, birimler, koordinat dönüşümleri, eklem sırası ve TCP sözleşmesi doğrulanır.
- Referans ve bağımsız ileri kinematik ile Jacobian doğrulanmadan öğrenme aşamasına geçilmez.
- Sayısal çözücünün başarısızlığı tek başına hedefin erişilemez olduğunu kanıtlamaz.
- Kinematik geçerlilik, çarpışmasızlık veya fiziksel robot güvenliğiyle eş tutulmaz.
- Ayrıntılı toleranslar ve test adetleri ilgili rapordan alınır; burada farklı eşikler türetilmez.

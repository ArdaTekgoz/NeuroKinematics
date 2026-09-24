# NeuroKinematics depo çalışma kuralları

Bu dosya depo kökünden başlayan bütün çalışma alanı için geçerlidir. Daha alt klasörde ek bir `AGENTS.md` bulunursa, o dosyanın daha dar kapsamlı ve çelişmeyen kuralları ayrıca uygulanır.

## Faz ve görev yönetimi

- Aktif geliştirme fazı Core'dur (G0 PASS / ACCEPTED, 24 Eylül 2026). Foundations çıktıları değişmez girdidir.
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

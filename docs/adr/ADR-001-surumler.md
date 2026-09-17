# ADR 001 Sürüm yapısı

Durum: Tasarım kararı · 17 Eylül 2026

Neuro şemsiyesi altında tek aktif proje NeuroKinematics olarak tutulur. Foundations, Core, Hybrid ve Studio dört ardışık ürün hedefidir. Kodda modüler paketler kullanılır; her faz için ayrı repository açılmaz. Localization gibi bağımsız alanlar araştırma birikiminde kalır.

Gerekçe: Kullanıcının fazlara ayrılmış rapor ve sistematik geliştirme hedefi; revize belgedeki kapsam kontrolü önerisi. Alternatif tek büyük release, erken ürün ailesi veya ayrı ayrı çekirdek kopyalarıydı. Seçilen yol doğrulanan bileşenleri tekrar kullanır.

Sonuç: Bir ana rapor, dört faz raporu, ortak test sözlüğü ve görev başına kayıt. Ürün hedef sürümü, belge r1 sürümü ve deney kimliği ayrıdır.

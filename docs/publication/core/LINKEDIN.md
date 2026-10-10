# LinkedIn paylaşım paketi

11 Ekim 2026. Taslak; otomatik yayımlanmadı. Sayısal kaynak:
[grafik verisi](figure-data.csv), [akademik rapor](../../research/C1-06_NEGATIVE_RESULTS.md).

## Paylaşım metni

Bir robotik yapay zekâ projesinde başarısız bir deneyden geriye ne kalır?

NeuroKinematics'te ters kinematik için geliştirdiğim modeller, hedeflediğim
2 mm konum ve 1° yönelim doğruluğunu güvenilir biçimde sağlayamadı.
Eklem limitlerine uyumu da birlikte arayan bu başarı tanımını korudum.

Son kontrollü deney öğreticiydi: üç eğitim seed'inde eğitim başarısı
3/0/0'dan 568/657/667'ye çıktı (her birinde 4096 örnek). Aynı köklerde
yeni yönlerde de ilerleme oldu. Ancak görülmemiş köklerde başarı
1/3600, 0/3600 ve 0/3600 kaldı; medyan hatalar kötüleşti.

Bu süreçte ileri kinematik ve gradyan doğrulamalarını, veri sözleşmelerini,
kayıp ve temsil seçimlerini, örnekleme kapsamını ve sıfır hareket davranışını
inceledim. Gerçek bir sayısal dönüşüm kusuru bulduk ve düzelttik; bu düzeltme
genel başarısızlığı açıklamaya yetmedi.

Benim için temel ders: eğitimde öğrenme, aynı noktada yeni yönlere uyum
ve yeni konfigürasyonlara genelleme ayrı ayrı ölçülmeli.

Core araştırma aşamasının yöntemini, olumsuz sonuçlarını ve yeniden üretim
kanıtlarını arşivliyorum. Projeye döndüğümde sıradaki soru, öğrenilmiş
başlangıcın aynı süre bütçesinde sayısal çözücüye fayda sağlayıp sağlamadığı
olacak. Hibrit üstünlük henüz ölçülmedi.

Kod, deney kayıtları ve rapor:
https://github.com/ArdaTekgoz/NeuroKinematics/tree/codex/c1-06r

#Robotics #InverseKinematics #MachineLearning #ReproducibleResearch

## Görseller ve açıklamaları

1. [generalization.png](generalization.png): ana gönderi görseli. Eğitim,
   aynı kökte yeni yön ve görülmemiş kök sonuçlarını aynı yüzde ekseninde
   gösterir. Sütun üstü sayılar başarılı örnek adetleridir.
   Alternatif metin: Üç seed'de merkezleme eğitim ve yeni yön başarısını
   artırırken validation başarısı yalnız bir seed'de 1/3600 olur; diğerleri sıfırdır.
2. [local-errors.png](local-errors.png): ikinci görsel. Local validation'da
   konum/yönelim medyanları ve limit ihlallerinin kötüleştiğini gösterir.
   Alternatif metin: RAW modelin 16 mm civarı konum ve 5 derece civarı
   yönelim hataları merkezlenmiş modelde yaklaşık 20 mm ve 6–7 dereceye yükselir.

## Okuyucunun doğru çıkarım yapması için

Grafikler D9'un bütün altı hücresini içerir. Eğitim ve yeni yön paydası
4096; validation paydası 3600; hata grafiğinde local payda 1800'dür.
Validation araştırma sırasında tekrar kullanılmıştır; bağımsız final
diye sunulamaz. Üç seed'i birleştirip 10800 bağımsız hedef iddiası kurma.

Core/G1 araştırma kapanışı doğrudan IK ürün başarısı anlamına gelmez.
"Robot %95 başarıya ulaştı", "neural IK imkânsız", "hybrid daha hızlı",
"fiziksel robot güvenli" gibi iddialar bu kanıtlarla desteklenmez.
Bu çalışma henüz hakemli makale değildir. Kod ve belge hazırlığında
yapay zekâ desteği, rapordaki katkı açıklamasıyla görünürdür.

GitHub bağlantısı özellikle çalışma branch'ine gider; main ile birleştirme
ve release yayımlama bu paylaşım paketinin parçası değildir. Gönderide
istersen rapor PDF'ini de ekleyebilirsin; ham veri/model dosyalarının
GitHub'dan indirilebildiği iddiasında bulunma.

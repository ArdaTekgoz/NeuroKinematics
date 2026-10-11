# Sonraki ayrım: yerel girdi ölçeği — PROPOSED / NOT_RUN

Bu belge tanı6 sonuçlarının ardından uygulanacak karar dalını tanımlar;
yeni bir eğitimin yapıldığını veya başarılı olacağını göstermez.

Tanı6'da aynı-kök yeni yönler de Profil A'yı sağlayamazsa ilk müdahale,
veri sayısını tekrar artırmadan yerel girdilerin train-only ölçeğini sınamak.
Tanı5'te mevcut karma local/wide normalizasyonuyla local pozisyon standart
sapması yaklaşık0.08–0.09; quaternion vektör bileşenleri yaklaşık0.04 idi.
Bu fark doğrulanmış bir neden değildir, kontrollü sınanacak hipotezdir.

Öneri: tanı6'nın aynı üretilmiş verisi, aynı kökleri ve aynı eğitim bütçesiyle
mevcut temsil vs yalnız göreli pose bileşenlerinin train-only standardizasyonu.
İstatistikler yalnız gerçek eğitim yönlerinden öğrenilir; probe/validation
kullanılmaz. q_current'ın residual toplamadaki [0,1] sözleşmesi korunur.
Veri, head, decoder, loss, optimizer ve kapasite birlikte değiştirilmez.
Kesin bütçe ve özellik dönüşümü yeni ADR/config ile ön kayıt gerektirir.

Aynı-kök başarısı belirgin artar, yeni-kök düşük kalırsa kökler arası geometriyi
öğrenme kapsamı incelenir. İkisi de düşükse finite-difference yön duyarlılığı
ve modelin öğrenilmiş yerel türevi, bağımsız Jacobian'ın öngördüğü ilişkiyle
ayrı tanıda karşılaştırılır. Sayısal düzeltici ekleyerek direct IK sonucu
başarılı gösterilmez. C1-07 ve yeni uzun kampanya için bu kanıt yeterli değildir.

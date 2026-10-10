# Öneri: sıfır hedef farkını yapısal olarak koruyan residual kontrol

10 Ekim 2026 · PROPOSED / NOT_RUN · Yeni model/eğitim yok

Tanı8'de iki ayrı mekanizma görüldü: hedef zaten sağlandığında model
ofseti pose'u bozuyor; hedef değişimine türev tepkisi de doğru değil.
Yeni veri/epoch/scaler taramasından önce ilk mekanizmayı doğrudan kontrol
eden bir mimari ablasyonu anlamlıdır.

Önerilen tek değişken: normalize residual düzeltmeyi
`g(x) - g(x_zero(q_current))` biçiminde tanımlamak. x_zero, aynı current
eklemler için sıfır göreli pose girdisidir; yalnız normal inference girdileri
ve mevcut FK kullanılır, teacher kullanılmaz. Hedef current pose olduğunda
ağ düzeltmesi sıfır olur. Mevcut float32 current normalizasyonu/float64
decoder round-trip hatası ayrıca ölçülür; fiziksel q'nun bit düzeyinde
eşitliği test edilmeden vaat edilmez.

Kontrol mevcut `g(x)` residual başlık; müdahale centered başlık. Aynı veri,
ölçek, model kapasitesi, initialization, loss ve güncelleme bütçesi gerekir.
Centered başlık iki ağ değerlendirmesi gerektirir; eşit update eşit compute
demek değildir. İki kolun gerçek süre/forward-backward maliyeti açık raporlanmalı,
seçilen bütçe adaleti yeni ADR/config'te eğitimden önce belirlenmelidir.
Kesin seed/örnek/adım ve checkpoint seçimi henüz yeni protokol olarak yazılmadı.

Bu değişiklik niçin tek başına yeterli sayılmaz? Sabit q_current'ta,
`d[g(x)-g(x_zero)]/d q_target = d g(x)/d q_target`.
Aynı ağırlıkların çıktısını sadece merkezlemek hedefe göre türevi değiştirmez.
Yeni eğitim farklı ağırlıklar öğrenebilir; bunun probe ve yeni-kök başarısını
iyileştirdiği ölçülmelidir. Sıfır testini geçmek ürün başarısı değildir.

Gerekli rapor: train, aynı-kök yeni yön, yeni-kök local/wide, sıfır kontrolü,
yerel task türevi ve maliyet. Sonuç yalnız sıfır testini düzeltirse daha uzun
eğitime atlanmaz; öğrenilmiş yerel teğeti kullanan bir yapı ayrı ADR ile
değerlendirilir. Sayısal IK refinement eklenirse ayrı hybrid sonuçtur.

A/B ve ürün hedefi değişmez. Negatif araştırma kapanışı alternatifi önceki
Core sözleşmesindeki anlamını korur. C1-07/G1 bu öneriyle başlamaz.

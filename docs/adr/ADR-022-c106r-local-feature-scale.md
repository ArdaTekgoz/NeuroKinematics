# ADR-022 — Yerel göreli pose ölçeği ve validation kıstası denetimi

10 Ekim 2026 · Kabul edildi · C1-06R tanı7

## Soru ve sabit deney

Tanı6'nın C1-02R/v1 verisi değiştirilmeden mevcut göreli13 girdi ile yalnız
ilk7 pose bileşeni train-only z-score yapılmış girdi karşılaştırılır.
64 ve512 kök, kök başına8 eğitim yönü ve ayrı8 probe yönü. REPEAT kolu yok;
kontrol tanı6 DIRECTIONS kolunun birebir tekrarını sağlamalıdır.

Yeni istatistikler her ölçekte yalnız o ölçekteki512/4096 gerçek eğitim
yönünden, float64 raw relative position3 + canonical quaternion4 üzerinde
population std (ddof0) ile hesaplanır. Sıfır/nonfinite std hata verir.
Son6 q_current [0,1] bileşeni değiştirilmez; residual toplama ve float64
decoder sözleşmesi aynıdır. Standardizasyon yeni mimari veya fizik kaybı
değildir; mevcut quaternion bilgisini koruyan affine dönüşümdür.

İki ölçekte RAW ve LOCAL_Z: aynı yeni başlangıç,13-512³-6 SiLU, Q loss,
AdamW1e-3/weight_decay0.01, cosine eta_min1e-6,5000 fullbatch update,
seed2026100901. Son checkpoint; erken durdurma/uzatma yok. Toplam20000 update.
Probe/validation normalization fitting'e girmez, teacher feature olarak kullanılmaz.
Sonuç öncesi config/kod/test/ADR ve veri SHA kaydı gerekir.

## Teknik denetim ve yorum

Tanı6 exact-byte veri/kinematik kanıtı ve122 frozen kaynak kontrol edilir.
Yeni feature testleri fitting sınırı, teacher bağımsızlığı, dönüşümün geri
alınabilirliği ve residual current korunumunu sınar. RAW kontrol ağırlıkları
ve dört küme metrikleri tanı6 DIRECTIONS ile birebir olmalıdır. Bütün yeni
checkpointler weights_only=True ile okunur ve bütün kümelerde replay yapılır.

Validation denetimi ayrı, model seçimi yapmayan açıklayıcı çalışmadır:
bağımsız FK + atan2 geodezik açıyla ham tahminleri tekrar değerlendir;
Profil A/B birlikteliği, metre/derece, limit ve tam paydayı sınır çevresi
sentetik mutantlarla doğrula. Validation local root teacher'ını aynı root'un
wide satırına tanık çözüm olarak eşle; eksik wide teacher'ın erişilemezlik
olmadığını test et. Bu oracle öğrenilmiş model veya sayısal baseline değildir.

Valid/position-only/rotation-only/joint-success ve local/wide ayrımı; ayrıca
A toleranslarının0.5,1,2,5,10 katında açıklayıcı duyarlılık tablosu raporlanır.
Bu tablo kabul eşiklerini değiştirmez, sonuçları yeniden PASS etiketlemez.
Ürün A2mm/1°,B1mm/0.5°, üç seed main A≥%95; H2-R ayrı kalır.
Kaynak Core §6'daki araştırma kapanışı/ürün hedefi ayrımı açıkça korunur.
Eski final NOT_READ, yeni final NOT_CREATED; bu deney C1-07'yi başlatmaz.

# Yerel ölçekleme tanısı — genelleme iyileşmedi

10 Ekim 2026 · DIAGNOSTIC7_COMPLETE / VALIDATION_TARGET_NOT_MET

Train-only göreli pose standardizasyonu train A'yı artırdı; aynı-kök yeni
yönlerde ve yeni-kök validation'da A/B yine0, çoğu sürekli hata daha kötü.
Bu müdahale mevcut bütçede tercih edilen yeni temsil değildir.

## Kontrollü koşullar

ADR-022/config üretimden önce kayıtlı. Tanı6'nın hashli C1-02R/v1 verisi;
64/512 kök ×8 eğitim yönü, kök başına8 ayrı probe yönü. RAW önceki
DIRECTIONS kolunu birebir tekrarlar. LOCAL_Z yalnız raw relative position3
ve canonical quaternion4'ü o hücrenin train mean/std (ddof0) ile dönüştürür.
Son6 normalized q_current, residual head ve decoder aynıdır. Probe/validation
istatistiklere girmez. Model13-512³-6, yeni aynı seed/ağırlıklar, Q loss,
AdamW/cosine,5000 fullbatch update sabit; terminal checkpoint kullanıldı.

| Kök / temsil | Train A | Train B | Aynı-kök probe A | Validation A |
|---|---|---|---|---|
|64 RAW|105/512|19/512|0/512|0/3600|
|64 LOCAL_Z|259/512|129/512|0/512|0/3600|
|512 RAW|3/4096|0/4096|0/4096|0/3600|
|512 LOCAL_Z|60/4096|14/4096|0/4096|0/3600|

Probe/validation B de0; validation main A/B0/3000. Eşit maruziyet her64
kolda2.560.000, her512 kolda20.480.000; toplam20000 update/46.080.000 satır.

| Kök / temsil | Train medyan mm / ° | Probe medyan mm / ° | Validation local medyan mm / ° |
|---|---|---|---|
|64 RAW|3,548 / 0,807|21,270 / 5,179|56,148 / 12,216|
|64 LOCAL_Z|1,885 / 0,360|42,698 / 8,171|58,302 / 10,703|
|512 RAW|12,013 / 3,677|13,792 / 4,341|16,040 / 4,840|
|512 LOCAL_Z|10,661 / 2,664|20,491 / 4,893|23,131 / 5,594|

Tam paydalı nearest-rank medyan; invalid +∞.512 LOCAL_Z valid oranları:
probe3974/4096, validation local1699/1800, wide1/1800. RAW karşılıkları
3999/4096,1737/1800,446/1800. Local-only normalizasyon wide için ciddi
ölçek dışı girdiler oluşturur; bu model global ürün adayı sayılmaz.

## Teknik kanıt

64 test PASS (57 mevcut+7 yeni);7 yeni test train-only fit sınırı,
invertibility/current, teacher bağımsızlığı, RAW eşliği, metre/derece
sınır çevresi, AND kuralı, invalid/tam payda ve quaternion işareti.
İki RAW kontrolün tanı6 terminal tensorleri ve bütün dört küme metrikleri
birebir. Dört yeni checkpoint güvenli weights_only yükleme ve33984 satırda
çıkarım replay PASS. Aynı33984 tahmin bağımsız FK ve alternatif atan2
geodezik açıyla incelendi; A/B sınıfları birebir, hata farkları audit.json'da.
Tanı6 exact-byte kinematik kanıtı yeniden kullanıldı; yeni uzun Foundations
test kampanyası çalıştırılmadı.122 frozen kaynak aynı.

Validation root oracle A/B3600/3600:351 eksik wide teacher dahil her hedef
için mevcut local validation kökünden limit içi tanık çözüm doğrulandı.
Bu oracle label kullanır; model girdisi/eğitimi veya öğrenilmiş başarısı değildir.
Final NOT_CREATED; eski final NOT_READ.

## Karar

Ölçekleme train optimizasyonunu etkiliyor ancak tek başına hedef hassasiyeti
ve genellemeyi çözmüyor. Yalnız64-kök train A artışına bakarak uzun eğitime
geçmek yanlış olur.512-kök train A60/4096 hâlâ düşük; probe0 ve yeni-kök0.
Bu sonuç, aynı modelin daha uzun bütçeyle asla öğrenemeyeceğini kanıtlamaz.
Tek seed ve sınırlı/sıralı kök kapsamı sınırları korunur.

Kullanıcının “kıstas mı, gidişat mı?” sorusu için ayrıntılı değerlendirme:
[CRITERION_AND_DIRECTION_REVIEW](CRITERION_AND_DIRECTION_REVIEW.md).
Ürün eşiği değiştirilmedi; yeni eğitim başlatılmadı.

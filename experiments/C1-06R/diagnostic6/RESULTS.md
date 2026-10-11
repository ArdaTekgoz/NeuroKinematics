# C1-02R/v1 yerel yön deneyi — hedef karşılanmadı

10 Ekim 2026 · DIAGNOSTIC6_COMPLETE / VALIDATION_TARGET_NOT_MET

Veri yönlerini çoğaltmak sürekli hataları azalttı, fakat Profil A başarısı
aynı-kök yeni yönlerde ve yeni-kök validation'da sıfır kaldı. Salt örnek
tekrarıyla küçük train başarısı, yerel IK ilişkisini öğrenme kanıtı değildir.

## Sabit karşılaştırma

ADR-021/config, veri üretiminden önce SHA ile kaydedildi. İlk64/512 matched
main train kökü; kontrol mevcut örneği8 kez, müdahale orijinal+7 yeni yönü
kullanır. Her kökte ayrı8 yön yalnız tanı içindir. Her model yeni ve aynı
başlangıç ağırlıklarıyla 5000 full-batch AdamW güncellemesi alır; genişlik512,
Q loss, relative/residual temsil, normalizasyon ve decoder aynıdır.
Tek model seed; terminal checkpoint, erken durdurma/uzatma yok.

64-kök kolları512 satır/update, 2.560.000 maruziyet;512-kök kolları4096
satır/update,20.480.000 maruziyet. Maruziyet her ölçekte eşittir; benzersiz
eğitim örnekleri kontrol64/512, müdahale512/4096'dır. Ölçekler arası bütçe
aynı değildir; nedensel kıyas her ölçekte iki kol arasında yapılır.

## Ölçümler

| Kök | Eğitim verisi | Train A (maruziyet satırları) | Ortak orijinal A | Aynı-kök yeni yön A | Validation A |
|---|---|---|---|---|---|
|64|Tek yön ×8|512/512|64/64|0/512|0/3600|
|64|Sekiz farklı yön|105/512|13/64|0/512|0/3600|
|512|Tek yön ×8|1928/4096|241/512|0/4096|0/3600|
|512|Sekiz farklı yön|3/4096|1/512|0/4096|0/3600|

Kontrolün tekrarlı train paydası bağımsız örnek sayısı değildir. Aynı-kök
tanı satırları train köklerini paylaşır; final veya yeni-kök validation sayılmaz.
Validation A/B ve main A/B dört modelde de0; main payda3000, tam payda3600.

| Kök / kol | Aynı-kök yeni yön medyan mm / ° | Bu kümede limit dışı | Validation local medyan mm / ° |
|---|---|---|---|
|64 tekrar|54,862 / 9,615|27/512|76,354 / 12,463|
|64 yön|21,270 / 5,179|10/512|56,148 / 12,216|
|512 tekrar|40,095 / 8,189|266/4096|45,983 / 9,595|
|512 yön|13,792 / 4,341|97/4096|16,040 / 4,840|

Medyanlar tam paydalı nearest-rank; invalid +∞.512 kökte yön çeşitliliği
aynı-kök pozisyon medyanını yaklaşık%65,6, orientation medyanını%47,0 azaltır.
Bu anlamlı sürekli-hata iyileşmesi,2mm ve1° ortak kabulünü sağlamamıştır.
Profil A oranı değişmediğinden yalnız A üzerinden veri müdahalesi etkisiz
denemez; sürekli metrikler yön bilgisinin faydalı olduğunu gösterir.

## Doğrulama

- C1-06R57 test PASS; bu turn'de tüm repo testleri yeniden çalıştırılmadı.
-8192 türetilmiş satırın bütün alanları/provenance deterministik tekrar eşliği.
- Öğrenme öncesi16384 current/teacher FK ve Jacobian çapraz kontrolü PASS;
  teacher oracle her iki4096 kümede A/B4096. Root/group leakage yok.
- İlk FD32 satırı aynı dört teacher kökünü tekrarlar; kapsam gizlenmedi.
  Son audit ayrıca512 farklı kökte1024 FK/Jacobian,32 farklı teacher FD PASS.
- Dört checkpoint weights_only=True ve bütün33984 değerlendirme satırı
  replay eşliği; aynı tahminlerin bağımsız FK hata karşılaştırması PASS.
-122 frozen giriş korunur; eski final NOT_READ, yeni final NOT_CREATED.

Sayısal ayrıntılar [audit](audit.json), [preflight](preflight.json),
[sonuç JSON](results.json); büyük veri/ağırlıklar Git dışı hashli ham yollarda.

## Yorum ve sınır

64 tek örnekte64/64, yeni yönlerde0/512: daha önceki küçük overfit başarısı
genelleme kapısı olamaz. Sekiz yönle train bile64 kökte105/512,512 kökte3/4096:
problem yalnız görülmemiş global köklere indirgenemez. Mevcut temsil/amaç ve
5000 update bütçesinde yerel hassasiyet ve yerel genelleme de çözülmemiştir.
Bu bulgular daha uzun eğitimle çözülemeyeceğini veya tek nedenin ölçek olduğunu
kanıtlamaz. Tek seed, sıralı matched-root seçimi ve sınırlı yön sayısı genelleme
sınırlarıdır. Aynı kökün sekiz yönü aynı etiketi paylaşır; yön etkisi ve optimuma
ulaşma hızı eşit güncellemede birlikte gözlenir.

Yeni bir Foundations FK/Jacobian hatası bulunmadı; bu test kapsamı fiziksel
kalibrasyonu veya bütün Foundations tasarımını doğrulamaz. Eski kanıt değişmedi.
Yeni uzun eğitim ve C1-07 başlatılmaz. Karar dalı: veri büyütmeden yerel göreli
girdilerin train-only ölçeği için tek değişkenli kontrol; [sonraki tanı](NEXT_DIAGNOSTIC.md)
PROPOSED / NOT_RUN.

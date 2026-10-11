# ADR-021 — C1-02R/v1 yerel yön kapsamı tanısı

10 Ekim 2026 · Kabul edildi; C1-06R içinde kontrollü araştırma.

## Bağlam ve karar

Tanı5 REPLAN uygulanır. Dondurulmuş F0-04/C1-02/v1 değiştirilmez.
Yeni C1-02R/v1 türevi yalnız train main köklerinden oluşur. Önceki tanılarla
aynı seçim: local ve wide etiketi bulunan köklerin sıralı ilk64 ve ilk512'si.
Bu seçim tüm main popülasyonunun rastgele örneklemi değildir.

Her kökte mevcut local satır + yedi yeni perturbasyon eğitim desteğini oluşturur.
Ayrı sekiz perturbasyon aynı-kök tanısıdır; validation veya bağımsız final değildir.
Her yeni yön SHA256(namespace, data_seed, root, role, index) ilk16 baytından
little-endian PCG64 seed alır. q_current = q_target + U(-0.1,0.1)^6;
limit dışında tüm vektör reddedilir; clipping yok. Hedef pose ve etiket aynı
dondurulmuş köktür. Seed, deneme sayısı ve root provenance saklanır.

Kontrol mevcut satırı sekiz kez; müdahale sekiz farklı yönü kullanır. Her kol
full-batch 8N satır × 5000 güncelleme; eşit maruziyet, farklı benzersiz örnek sayısı.
İki ölçekte aynı sıfır çıkışlı 13-512-512-512-6 SiLU, residual head,
Q kaybı, float64 endpoint-exact decoder; tanı3 normalizasyonu sabit.
AdamW lr0.001/weight_decay0.01, cosine eta_min1e-6; seed2026100901.
Yeni optimizer ve yeni model; warm-start yok. Terminal checkpoint kullanılır.
Ara kayıtlar yalnız Q kaybıdır; erken durdurma, sonuç sonrası uzatma yok.

## Doğrulama ve karar

Üretimden/eğitimden önce config/source SHA kaydı; her yeni satırda bağımsız
FK/current-teacher, analitik Jacobian çapraz kontrolü; ilk32 iç teacher'da FD.
Toleranslar tanı5/F0-03/C1-03'ten aynen alınır. Öğrenme öncesi teacher oracle
A/B ve deterministik üretim replay gerekir. Root/group validation ayrımı,
yön kümelerinin ayrıklığı ve orijinal satır eşliği zorunludur.

Her terminal model için: eğitim, ortak orijinal satırlar, aynı-kök yeni yönler,
tam3600 validation (local/wide ve aile kırılımı). Limit dışı tahmin paydadan
çıkarılmaz. Her checkpoint weights_only=True ile ve bütün küme metrikleriyle
tekrar doğrulanır. Yeni yönlerin başarı oranı ve hata dağılımı eşli raporlanır;
tek seed sonuçları nedensel genellemeyi veya ürün kabulünü kanıtlamaz.

Aynı-kök yeni yönlerde artış, validation düşük: kökler arası genelleme yönünde
kanıt. Aynı-kök de düşük: veri büyütmeden train-only temsil ölçeği/optimizasyon
ayrı deney. İyileşme tanımı yüzde puan farkıdır; yeni bir ürün eşiği değildir.
Ürün kapısı üç seed main A≥%95; A2mm/1°, B1mm/0.5° değişmez.
Eski final okunmaz, yeni final oluşturulmaz; C1-07 başlamaz.

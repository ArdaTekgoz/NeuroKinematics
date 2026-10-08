# C1-05 validation sonuçları

8 Ekim 2026 · r1. T-C04 PASS; doğrudan IK NO-GO.

| Deney | Seed | Model | Epoch / best | Q loss | Limit dışı /3600 | Konum medyan m* | Yönelim medyan °* | Profil A |
|---|---|---|---|---:|---:|---:|---:|---|
| E-C03 | 2026100201 | Q | 200 / 199 | 0.170779 | 269 | 0.206133 | 74.6479 | 0/3600 |
| E-C03 | 2026100201 | FK | 200 / 194 | 0.532655 | 413 | 0.215197 | 26.9175 | 0/3600 |
| E-C03 | 2026100202 | Q | 200 / 200 | 0.174430 | 184 | 0.207037 | 78.1453 | 0/3600 |
| E-C03 | 2026100202 | FK | 200 / 193 | 0.399691 | 230 | 0.162541 | 25.8861 | 0/3600 |
| E-C03 | 2026100203 | Q | 200 / 199 | 0.173756 | 188 | 0.210282 | 76.5182 | 0/3600 |
| E-C03 | 2026100203 | FK | 200 / 197 | 0.406346 | 328 | 0.170027 | 27.2912 | 0/3600 |
| E-C04 | 2026100201 | FK | 26 / 6 | 0.655523 | 529 | 0.373346 | 48.7464 | 0/3600 |
| E-C04 | 2026100201 | FK_LIMIT | 26 / 6 | 0.653034 | 468 | 0.374393 | 50.1352 | 0/3600 |
| E-C04 | 2026100202 | FK | 200 / 193 | 0.399691 | 230 | 0.162541 | 25.8861 | 0/3600 |
| E-C04 | 2026100202 | FK_LIMIT | 200 / 197 | 0.416644 | 302 | 0.173222 | 27.9912 | 0/3600 |
| E-C04 | 2026100203 | FK | 26 / 6 | 0.615468 | 490 | 0.353900 | 68.8564 | 0/3600 |
| E-C04 | 2026100203 | FK_LIMIT | 26 / 6 | 0.614631 | 446 | 0.351608 | 69.6905 | 0/3600 |
| E-C05 | 2026100201 | FK | 27 / 6 | 0.655523 | 529 | 0.373346 | 48.7464 | 0/3600 |
| E-C05 | 2026100201 | FK_TANH | 27 / 7 | 0.395505 | 0 | 0.283412 | 62.0037 | 0/3600 |
| E-C05 | 2026100202 | FK | 200 / 193 | 0.399691 | 230 | 0.162541 | 25.8861 | 0/3600 |
| E-C05 | 2026100202 | FK_TANH | 200 / 186 | 0.377647 | 0 | 0.137031 | 25.3105 | 0/3600 |
| E-C05 | 2026100203 | FK | 28 / 6 | 0.615468 | 490 | 0.353900 | 68.8564 | 0/3600 |
| E-C05 | 2026100203 | FK_TANH | 28 / 8 | 0.394650 | 0 | 0.277780 | 60.2167 | 0/3600 |

*Medyanlar yalnız geçerli ham q altkümesindedir. Payda her koşuda 3600; 351 etiketsiz wide satırı dahil. Profil B de bütün koşularda 0/3600. Geçersizlere sonsuz atayan tam-payda medyan/P95/P99, geçerli-altküme P95/P99, local/wide, boundary/singularity ve etiket kırılımları ilgili validation.summary.json ve results-audit.json dosyalarındadır.

| Deney | Seed | Δbaşarı /3600 | Δlimit dışı | Δkonum medyan m* | Δyönelim medyan °* |
|---|---|---:|---:|---:|---:|
| E-C03 | 2026100201 | 0 | +144 | +0.009064 | -47.7304 |
| E-C03 | 2026100202 | 0 | +46 | -0.044497 | -52.2592 |
| E-C03 | 2026100203 | 0 | +140 | -0.040255 | -49.2270 |
| E-C04 | 2026100201 | 0 | -61 | +0.001046 | +1.3888 |
| E-C04 | 2026100202 | 0 | +72 | +0.010682 | +2.1051 |
| E-C04 | 2026100203 | 0 | -44 | -0.002292 | +0.8341 |
| E-C05 | 2026100201 | 0 | -529 | -0.089934 | +13.2573 |
| E-C05 | 2026100202 | 0 | -230 | -0.025509 | -0.5756 |
| E-C05 | 2026100203 | 0 | -490 | -0.076120 | -8.6397 |

Fark müdahale eksi aynı deneyin **taze eşli kontrolü**dür; eksi hata daha iyidir.
E-C03'ün üç Q checkpointi C1-04 conditioned q çıktılarıyla birebir aynı (max fark0).
E-C04/05'teki taze FK comparator, E-C03'ün 200 epoch sonucuyla karıştırılmaz.
Ön kayıtlı ortak patience20 kuralı E-C04 seed1/3'ü26, E-C05 seed1/3'ü27/28 epoch'ta
durdurdu. Her çiftin iki arm'ı aynı epoch/step bütçesine sahiptir; deneyler arası
farklı gerçekleşen bütçe korunur. Yeniden uzun eğitimle bu negatif/karma sonuçlar
değiştirilmedi. Üç seedin mean/std/min/max farkları `results-audit.json` içinde.

FK eklemek yönelimi her seed'de yaklaşık48–52° iyileştirdi; konum etkisi
seed1'de +9 mm kötüleşme, seed2/3'te yaklaşık−44/−40 mm iyileşmedir. Ham limit
ihlali üç seed'de de arttı. Başarı artışı **0 çözüm /3600**. Limit cezası ihlali
iki seed'de61/44 azalttı, bir seed'de72 artırdı; tutarlı yarar kanıtlanmadı.
Tanh sıfır ham limit ihlali sağladı; yönelim etkisi karma, başarı artışı yine0.
Sınır içinde çıktı üretmek hedef poza ulaşmak veya fiziksel güvenlik değildir.


## Tam payda ve seçim yanlılığı

Geçersiz q satırlarına +∞ atanır; P95 değeri sınırsızsa aşağıda ∞ gösterilir.
Bu tablo bütün 3.600 satırı kapsar; yukarıdaki geçerli-altküme medyanlarıyla
birlikte okunmalıdır. Başarı paydası hiçbir durumda daraltılmadı.

| Deney | Seed | Model | Konum medyan / P95 (m) | Yönelim medyan / P95 (°) |
|---|---|---|---|---|
| E-C03 | 2026100201 | Q | 0.223689 / ∞ | 81.6425 / ∞ |
| E-C03 | 2026100201 | FK | 0.238897 / ∞ | 30.0963 / ∞ |
| E-C03 | 2026100202 | Q | 0.220931 / ∞ | 82.3755 / ∞ |
| E-C03 | 2026100202 | FK | 0.173722 / ∞ | 27.7458 / ∞ |
| E-C03 | 2026100203 | Q | 0.225868 / ∞ | 81.1445 / ∞ |
| E-C03 | 2026100203 | FK | 0.184947 / ∞ | 29.904 / ∞ |
| E-C04 | 2026100201 | FK | 0.425813 / ∞ | 55.3325 / ∞ |
| E-C04 | 2026100201 | FK_LIMIT | 0.422892 / ∞ | 55.965 / ∞ |
| E-C04 | 2026100202 | FK | 0.173722 / ∞ | 27.7458 / ∞ |
| E-C04 | 2026100202 | FK_LIMIT | 0.187545 / ∞ | 30.5957 / ∞ |
| E-C04 | 2026100203 | FK | 0.400267 / ∞ | 78.0012 / ∞ |
| E-C04 | 2026100203 | FK_LIMIT | 0.393839 / ∞ | 77.8964 / ∞ |
| E-C05 | 2026100201 | FK | 0.425813 / ∞ | 55.3325 / ∞ |
| E-C05 | 2026100201 | FK_TANH | 0.2834 / 0.75248 | 61.9979 / 153.867 |
| E-C05 | 2026100202 | FK | 0.173722 / ∞ | 27.7458 / ∞ |
| E-C05 | 2026100202 | FK_TANH | 0.137001 / 0.423229 | 25.3071 / 99.828 |
| E-C05 | 2026100203 | FK | 0.400267 / ∞ | 78.0012 / ∞ |
| E-C05 | 2026100203 | FK_TANH | 0.277761 / 0.733618 | 60.2079 / 154.97 |

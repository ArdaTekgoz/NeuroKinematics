# ADR-018 — Göreli girdide optimizasyon, ölçek ve kapasite tanısı

Tarih: 10 Ekim 2026 · Durum: ACCEPTED FOR DIAGNOSIS · C1-06R / REQ-C02–05

## Bağlam

Tanı3'te 2048-local-relative-residual train A0/2048 ve validation A0/3600.
Yerel validation medyanı 19,57mm/4,98°; eğitim hassasiyeti de yetersiz.
Kullanıcı dört kontrollü koşulu birlikte gerçekleştirmeyi onayladı.

## Karar ve ön kayıt

`experiments/C1-06R/diagnostic4/config.json` sonuç görülmeden sabitlenir.
Aynı 2048 train satırı, tanı3 train-only normalizasyonu, 13 göreli girdi,
residual çıktı, ADR-016 decoder, Q hedefi, seed ve tam batch korunur.
Yeni modül önceki kayıtlı kodun hiçbir baytını değiştirmez.

1. Referans: 3×256 SiLU, AdamW5000, lr.001/wd.01, cosine eta1e-6.
2. Optimizer paketi: aynı başlangıçtan ölçeklenmemiş Q ile L-BFGS;
   max_iter5000/max_eval6250, history30, strong_wolfe, toleranslar config'te.
3. Sayısal ölçek: aynı AdamW ile Q×1e6. Amaç/minimum değişmez. Adam'ın
   ölçek değişimine yaklaşık duyarsızlığı nedeniyle iyileşme garanti değildir.
4. Kapasite: aynı AdamW/Q/bütçe ile üç gizli katman genişliği256→512.

Üç dar modelin başlangıç parametreleri aynı; dört modelin başlangıç
fonksiyonu q_current. Geniş modelin parametreleri ve hesap maliyeti farklı.
L-BFGS sıfırdan başlar, warm-start yoktur. Optimizer karşılaştırması
algoritma + durma/schedule/decay paketini karşılaştırır; L-BFGS'te decay
yoktur. Tek tek optimizer iç etkenleri ayırdığı iddia edilmez. Gerçek
closure sayısı, internal iterasyon ve süre raporlanır. Line search son
iterasyonda nominal max_eval'i aşabilir; eşit hesap bütçesi iddiası yoktur.

## Kabul ve yorum

Referans son ağırlıkları tanı3 kontrolüyle tensor bazında aynı olmalı.
Model genişliği/başlangıç ve kayıp türevi testleri çalıştırılır. Dört son
checkpoint güvenli weights_only reload ve train/validation yeniden ölçümü
ile doğrulanır. Tüm başarısızlıklar ve ham çıktılar korunur.

Profil A/B eşikleri değişmez; 3600 validation tam paydası ve local/wide
alt grupları raporlanır. Yalnız terminal checkpoint kullanılır. Validation
sonucuna göre bütçe uzatılmaz. Sürekli hata düşüşü, hedefin sağlanması ve
genelleme ayrı yorumlanır. Tek seed tanısı üç seed ürün kapısını geçiremez.
Yeni final oluşturulmaz; eski final açılmaz; C1-07 başlamaz.

## Sonuçları

Bu deney yalnız seçilen genişlik, optimizer paketi ve bütçe için kanıt
üretir. Başarısızlık daha büyük ağın veya tüm optimizerların başarısızlığını
kanıtlamaz. Uzun eğitim kararı bu tanı ve geniş train hassasiyet kanıtından
sonra ayrıca değerlendirilir; eski eğitim komutu yeniden kullanılmaz.

# ADR-024 — Sıfır hedef farkında merkezlenmiş residual başlık

Tarih: 10 Ekim 2026. Durum: deney öncesi kabul; ürün mimarisi kararı değil.

ADR-023 sıfır hedef farkında bile TCP ofseti ve yerel türev sapması buldu.
Sabit ağırlıklarda merkezleme hedefe göre türevi düzeltmez; yeni eğitimde
genelleme etkisi bilinmiyor. Bu nedenle tek değişkenli eşli kontrol uygulanır.

RAW: q_norm = current_norm + g(x).
CENTERED: q_norm = current_norm + g(x) - g(x_zero_current).
x_zero_current aynı current girdisini, sıfır göreli konum ve birim quaternion
ile kullanır. İlk üç bileşen mevcut normalizasyona göre -mean/std'dir.
Etiket, teacher, hedef kökü veya sayısal IK çözümü inference'a girmez.
Çıkış dönüşümü mevcut endpoint-exact float64 sözleşmesidir.

512 kök, ADR-021'in 4096 directions satırı; RAW ölçeği; 512 genişlik;
her seed içinde aynı başlangıç tensorları; üç seed 2026100901/02/03;
her kolda 5000 full-batch AdamW update, lr .001, wd .01, cosine eta 1e-6.
Altı koşu, 30000 update, 122880000 satır maruziyeti. Son checkpoint;
validation ile checkpoint seçimi veya koşu uzatma yok. Parametre sayısı eşit,
CENTERED iki ağ değerlendirmesi yapar; eşit hesaplama süresi iddiası yok.
İlk RAW koşusu ADR-022 n512 RAW ile exact tensor ve dört ölçüm kümesi replay.

Her koşu train4096, original512, same-root probe4096, validation3600,
zero train-current512 ve zero validation-current1800 üzerinde ölçülür.
Profil A 2mm/1°, B 1mm/.5°, sıkı limitler; paydalar değişmez.
32 train ve32 validation interior current anchor'da task-space türevi;
shadow float64 merkezi FD h=1e-5 ile atanmış atol1e-5/rtol.001 kontrolü.
Geometri ADR-023 ile aynı seçim ve bağımsız FK/Jacobian denetimiyle doğrulanır.

Araştırmaya devam kapısı (ürün kabul eşiği değildir): üç seed'in HER BİRİNDE
local validation A ve same-root probe A kesin artmalı; local A-severity
max(position/.002, angle/1) medyan ve P95 kötüleşmemeli; centered bütün zero
A/B geçmeli; bütün bütünlük kontrolleri geçmeli. Üç seed ortalaması üzerinden
eşli1800 root bootstrap5000 tekrar, seed2026101009, yüzde95 aralığının altı
pozitif olmalı. Bu aralık sabit üç seed'e koşulludur; eğitim seed popülasyonu
belirsizliğini ölçmez. Invalid severity sonsuz, sonluya çevrilmez.

Kapı başarısızsa bu MLP başlık/ölçek mikro-deney ailesi burada durur.
Kapı geçerse en fazla BİR ön kayıtlı kök kapsamı doğrulaması önerilebilir;
otomatik uzun kampanya veya ürün başarısı ilan edilmez. Wide genelleme ayrıca
çözülmeden local-only başarı 50/50 ürün hedefini karşılamaz.
Kullanıcının hibrit ürün tercihi doğrultusunda sonraki karar Core bilimsel
kapanışı ve G1 teslimi; onlardan sonra adil sayısal/neural-seed kıyaslamasıdır.
Bu ADR C1-07/G1 veya Hybrid fazını başlatmaz, eşikleri ve Foundations'ı değiştirmez.

# ADR-025 — Core araştırma durdurma ve hibrit devir hazırlığı

10 Ekim 2026 · Durum: kabul edilmiş hazırlık kararı; G1 kabulü değildir.

## Bağlam

Kullanıcı üç aşamalı çalışmanın ikinci aşamasını yetkilendirdi: Core
kapanışına hazırlık ve hibrit devir kararını somutlaştırma. Üçüncü aşamadaki
akademik başarısızlık raporu bu kapsamda yazılmayacak. Kullanıcının ürün
önceliği gerektiğinde neural destek kullanan güvenilir hibrit IK'dir.

C1-06 T-C05 PASS/H2 REJECTED/direct IK NO_GO değişmez. C1-06R/ADR-024'ün
ön kayıtlı devam kapısı başarısız: CENTERED validation A1/0/0, yeni-kök
hata ve türev daha kötü. Aynı mimari ailesinde yeni küçük deney başlatılmaz.

## Karar

1. C1-06R araştırması **CLOSED_WITH_UNMET_PRODUCT_TARGET** olarak sonlandırılır.
   R0–R4 ve tanı kanıtları korunur. İlk planın R5 yeni bağımsız finali,
   validation geçiş koşulu sağlanmadığı için NOT_CREATED/NOT_RUN kalır.
   Bu, bütün plan teslimlerinin PASS olduğu anlamına gelmez. H2-R yeni
   bağımsız finalde NOT_EVALUATED; eski H2 reddi yerine yeni sonuç yazılmaz.
   Kapsamın durdurulması açıkça bu ADR ile kaydedilir; eşikler düşürülmez.
2. C1-07 hazırlığında C1-06'nın önceden seçilmiş **FK_TANH üç seed'i** ana
   araştırma devir adayı olarak korunur. Seçim sebebi tarihsel devamlılık,
   local/wide eğitim kapsamı, bounded head ve mevcut tekrar üretim kanıtıdır;
   ölçülmemiş hibrit üstünlük değildir. Tek bir şanslı seed seçilmez.
3. ADR-024'ün **RAW 512 üç seed'i** ek, keşifsel, local-only araştırma adayıdır.
   Aynı üç seed'de yeni-kök continuous hatası CENTERED'dan daha düşük, ancak
   direkt validation A/B0 ve limit ihlalleri vardır. H1 birincil modelinin
   yerini sonuçlara bakarak alamaz; H2-03 ön kaydında ikincil rolü açık kalır.
   Local/wide kapsamı gizleyen üretim yönlendirmesi veya yeni eşik kurulmaz.
4. CENTERED ve LOCAL_Z ürün/default seçilmez. Sonuçları negatif mekanizma
   kanıtı olarak arşivlenir. H1 faydası bütün adaylar için NOT_MEASURED.
5. Aday yolları/SHA, robot/TCP, girdiler, çıkış dönüşümü, veri/seed/seçim
   protokolü bir devir manifestinde sabitlenir. Hazırlık sırasında mevcut
   ortamda altı modelin weights_only yüklenmesi ve validation çıkarımı/FK
   denetlenir. Bu sonuç T-C06 temiz ortam testi diye sunulmaz.
6. G1 açık kalır: nihai model kartı, taze kilitli ortam/temiz checkout'ta
   çıkarım ve değerlendirme, gerekli regresyon kapsamı ve gerekçeli G1
   kararı C1-07'nin kalan işleridir. Hybrid uygulaması bu hazırlıkta başlamaz.

## Sonuçlar ve sınırlar

Eski kaynak raporlar, Foundations, checkpointler ve orijinal sonuçlar
değişmez. Core araştırma kapanışı olumlu ürün sonucuna eşitlenmez.
Ağırlıklar LOCAL_ONLY; Git push ağırlık/raw arşivlemesi değildir. Yeniden
indirme adresi veya bağımsız uzak yedek varmış gibi davranılmaz.
Hibrit sayısal motor aynı tolerans/algoritma/toplam bütçede kıyaslanır;
neural maliyeti dahil edilir. Faydası gösterilmezse sayısal motor varsayılan.
G1 sonrasında H2-01 → H2-02 → H2-03 sırası korunur.

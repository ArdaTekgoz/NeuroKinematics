# Core kapanışına hazırlık ve hibrit devir kararı

10 Ekim 2026 · İkinci kullanıcı aşaması COMPLETE_PREPARATION.
G1 OPEN; Hybrid NOT_STARTED. Bu belge nihai G1 kabulü değildir.

## Araştırma kapanış kararı

C1-06 T-C05 PASS, H2 REJECTED ve doğrudan IK NO_GO kararı değişmez.
C1-06R, **CLOSED_WITH_UNMET_PRODUCT_TARGET** olarak sonlandırıldı:
validation'da ürün hedefine ulaşılamadı; ADR-024 devam kapısı geçmedi.
Küçük başlık/ölçek varyasyonları ve yeni uzun eğitim durduruldu.

İlk C1-06R planındaki yeni bağımsız final (R5) NOT_CREATED/NOT_RUN.
Validation geçişi sağlanmadığından finali üretip tüketmiyoruz. H2-R için
yeni final sonucu NOT_EVALUATED; eski H2 reddi yeni deney sonucu değildir.
Araştırma durdurma kararı bütün orijinal plan teslimlerini PASS yapmaz.
Kapsam değişikliği [ADR-025](../../../docs/adr/ADR-025-core-research-stop-and-hybrid-handoff.md)
ile kayıtlıdır. Kaynak raporlar, eski sonuçlar ve eşikler korunur.

## Devirde seçilen adaylar

| Rol | Aile ve seed'ler | Gerekçe | Kullanım sınırı |
|---|---|---|---|
| Ana araştırma adayı | FK_TANH;2026100201/02/03 | C1-06'nın mevcut seçimi; local+wide eğitim; bounded çıktı; geçmiş temiz tekrar kanıtı | Doğrudan IK NO_GO; hibrit fayda ölçülmedi |
| İkincil keşifsel aday | LOCAL_RAW;2026100901/02/03 | Son tanılarda CENTERED'a göre daha iyi yeni-kök sürekli hata; üç seed birlikte | Yalnız local eğitim; wide dağılım dışı; limit ihlali yüksek; H1 birincil seçiminin yerine geçmez |
| Devirde aday değil | CENTERED,LOCAL_Z | Genelleme/ön kayıtlı kapı başarısız | Negatif araştırma kanıtı olarak korunur |

Bu seçim hazırlık çıkarımından önce config/ADR ile sabitlendi. En iyi
seed seçilmedi; validation'daki tek başarılı CENTERED örneği tercih nedeni
yapılmadı. FK_TANH'ın seçilmesi bütün modellerden daha iyi hibrit sonuç
vereceği iddiası değildir; bu H1'in ölçmesi gereken sorudur.

Altı checkpoint'in yolu, SHA256, mimari, normalization, seed ve rolü
[handoff-manifest.json](handoff-manifest.json) içinde. Açıklaması
[MODEL_CARD_DRAFT.md](MODEL_CARD_DRAFT.md) içinde. Modeller değiştirilmedi;
refinement, clamp, yeni seed seçimi veya model birleştirmesi yapılmadı.

## Hazırlık doğrulaması

Mevcut Windows/Python3.12.14/Torch2.10.0+cu128 ortamında CPU/thread1;
her model için3600 validation, toplam21600 tahmin. weights_only yükleme,
metadata/hash ve bağımsız FK/Profil A-B sınıflandırması PASS.
FK backend farkı en fazla4,63e-16m ve1,04e-15 rotation Frobenius;
ön kayıt1e-9/1e-9 sınırları içinde.

Altı modelde A/B0/3600 yeniden ölçüldü. FK_TANH limit dışı0/0/0;
LOCAL_RAW1417/1404/1397 (her biri3600; local/wide ayrı manifestte).
455 eski teslim dosyası,122 frozen girdi ve3–9 tanı ön kayıtları doğrulandı.
Bu mevcut ortam denetimi, temiz ortam T-C06 testi değildir. CPU/GPU exact
tensor eşliği veya batch1 latency üstünlüğü iddia edilmez.

## Hibrit faza koşullu devir

Hedef, bütçe içinde doğrulanmış IK çözümü; neural destek ölçülen faydaya bağlı.
G1 kapanmadan H2 başlatılmaz. G1 sonrası sıra H2-01 kabul/hata sözleşmesi,
H2-02 bütçeli sayısal refinement/fallback, H2-03 aynı motorda başlangıç
politikası kıyasıdır. q_current/merkez/neural/klasik restart aynı algoritma,
residual/Jacobian,tolerans,iterasyon tavanı ve toplam süreyle karşılaştırılır.
Neural ön işlem/çıkarım/doğrulama maliyeti bütçeye dahildir.

H1 mevcut hedefi P95 sürede≥%20 azalma, başarı kaybı≤1 yüzde puanı ve
eşli%95 CI desteği. Mutlak A/B, zor kümeler ve deadline ayrıca raporlanır.
İkincil LOCAL_RAW taraması ana H1 kontrastını sonuçtan sonra değiştiremez.
H1/G2 koşulları karşılanmazsa sayısal motor varsayılan kalır. Hata sınıfı
başına en fazla iki hedefli düzeltme turu; sonra ayrı araştırma kapsamı.

Limit dışı neural değer geçerli çözüm değildir. Gelecekte seed projection
kullanılırsa yalnız başlangıç üretir ve son çözüm tekrar doğrulanır.
Bu hazırlık projection politikasını uygulamadı; H2-01/02 ön kaydı gerekir.
Collision NOT_CHECKED; kinematik sonuç fiziksel robot güvenliği değildir.

## Açık işler ve aşama sınırı

[G1_READINESS.md](G1_READINESS.md) kalan T-C06/G1 işlerinin somut listesidir.
Ağırlıklar/raw LOCAL_ONLY; push onların uzak yedeği değildir.
C1-06R yeni finali oluşturulmadı; bu modeller için bağımsız final üstünlük
iddiası yok. C1-07 tamamlandı veya Hybrid başladı denmez.

İkinci kullanıcı aşamasının bütün hazırlık çıktıları tamamlandı. Sıradaki
kullanıcı aşaması3, C1-06/C1-06R akademik başarısızlık ve deney süreci raporudur;
bu aşamada yazılmadı ve komut beklenir. T-C06/G1 yürütmesi, üç aşamalı
istekten sonraki teknik iş olarak açık kalır.

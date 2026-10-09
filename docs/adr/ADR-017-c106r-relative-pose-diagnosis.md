# ADR 017 — Aynı kapasitede göreli pose girdi tanısı

10 Ekim 2026 · Kısa kontrollü tanı için kabul edildi

## Karar ve gerekçe

Tanı2'de residual çıktı tek başına validation başarısı sağlamadı. Kullanıcı
göreli pose deneyini çalıştırmamızı istedi. Ham/göreli pose girdisi ×
absolute/residual çıktı, local/mixed veri ve 512/2048 örnekle 16 kısa
koşu karşılaştırılacak. Tek seed, her hücre 5000 full-batch güncelleme;
aynı MLP 13-256-256-256-6 ve parametre başlangıcı, Q kaybı ve optimizer.
Kesin değerler diagnostic3/config.json içinde eğitimden önce sabittir.

Ham girdi tanı2/C1-04 conditioned sözleşmesidir. Göreli giriş:
base çerçevesinde p_target−FK(q_current).p, q_current TCP çerçevesinde
R_current.T @ R_target'ın kanonik wxyz quaternion'u ve aynı normalize
q_current. Konum farkının mean/std'si tüm 16.800 train girdisinden
hesaplanır; validation fit için kullanılmaz. Quaternion ayrıca normalize
istatistiğine sokulmaz. Her iki girdi 13 boyutludur; hedef teacher q,
mode, family veya split model girdisi değildir.

FK(q_current) sabit ön işlemdir; tahmine IK çözücüsü/düzeltme adımı
uygulanmaz. Bu Hybrid değildir. Göreli dönüşüm ve train-only konum
ölçeklemesi birlikte temsil paketidir; sonucu tek başına quaternion'a
veya tek başına ölçeklemeye atfetmeyiz. Raw/göreli farkı bu paketle sınırlı.

## Geçiş ve kanıt

Aynı hedef kökleri ve local/mixed satır seçimi ADR-015'i izler. Mixed
teacher başarısına koşullu tanı örneklemidir. Bütün 3600 validation
satırı, local/wide/main/zor grupları ve limit dışı tahminler raporlanır.
Son adım seçilir; validation en iyi epoch/seed seçimi yapılmaz. Yeni
endpoint-exact decoder ADR-016 bütün hücrelerde ortaktır. Ham kontrol
aynı config altında yeniden çalışır; eski tanı2 sayıları değiştirilmez.

512 ve 2048 ölçekleri genellemeyi küçük overfit başarısından ayırır.
Ürün kapısı üç seed main validation ≥%95 olarak kalır; bu tek-seed
tanı ürünü/finali kabul ettiremez. Yeni uzun eğitim veya final yoktur.

# ADR-020 — Aynı ağırlıklarda Profil A ölçekli pose amacı

10 Ekim 2026 · ACCEPTED FOR DIAGNOSIS · C1-06R / REQ-C03–05

## Bulgudan karara

Tanı5'te FK/Jacobian kontrolü PASS. Geniş modelin düşük condition
quartile'ında A1/512; tekillik tek açıklama değil. Q ile pose eşik aşımı
Pearson0,304; bütün2048 Q–P/Q–R çıktı gradyanları pozitif cosine verir.
Global parametre gradyan normu Q6,91e-5, P8,93e-3, R1,46e-2. Bu ölçüler
kayıpların etkilerinin farklı olduğunu gösterir; tek başına en iyi ağırlığı
veya kesin kök nedeni belirlemez. Eski raw-model Q/FK çatışması bu modele
aktarılmaz. Kullanıcı devam eden tanı ve kapsamlı incelemeyi yetkilendirdi.

## Tek müdahaleli kısa deney

Tanı4 geniş modelinin aynı son ağırlıklarından iki koşu başlatılır. Her
ikisi aynı yeni AdamW/schedule, aynı5000 tam-batch adım ve aynı2048 satır.
Girdi, temsil, model, decoder ve metrikler aynıdır. Kontrol Q'ya devam eder.
Müdahale yalnız amaç fonksiyonudur: konum kare hatası/(2mm)^2 ile matris
dönüş hatası/[8 sin²(1°/2)] toplamı. Her terim kendi Profil A sınırında1.
Bu düzgün surrogate, iki eşiğin ayrı ayrı sağlanması veya limit geçerliliği
ile eşdeğer bir kabul testi değildir. Ürün metrikleri aynen ayrıca hesaplanır.

Q terimi POSE_A arm'ına eklenmez. Limit cezası/clip/projeksiyon ve çıkarımda
sayısal solver eklenmez; geçersiz çıktılar başarısız kalır. TrainingFK'nin
ideal finite-revolute extension alanı yalnız türevli eğitim içindir.

## Kanıt ve sınırlar

Config/kod/test kaynakları eğitimden önce hashlenir. Dört eşik/sıfır/kayıp
tanımı kontrolü, başlama ağırlığı eşliği, güvenli reload ve bütün train/val
replay yapılır. Terminal model kullanılır; sonuçla bütçe uzatılmaz. Toplam
10.000 update; fizik kaybı daha maliyetli, eşit compute iddiası yok.
Bu deney tanı5 bulgusundan sonra tasarlanmıştır; başlangıçtan beri ön
kayıtlıymış gibi gösterilmez. Yeni final yok; eski final açılmaz. Validation
çok kez görülmüştür; olası iyileşme bağımsız final/genelleme kanıtı sayılmaz.

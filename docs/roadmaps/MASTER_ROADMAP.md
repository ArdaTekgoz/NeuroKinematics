# NeuroKinematics ana roadmap

> 11 Ekim 2026: Core COMPLETE / G1 PASS / ACCEPTED (araştırma kapanışı). Ürün NOT_MET; Hybrid NOT_STARTED.

Plan r2 · Başlangıç kapasitesi haftada 8–12 saat · İlk robot KUKA KR6 R900 sixx varsayımı

1. [Foundations](F0_Foundations.md) — 60–90 saat — G0 kinematik/veri kapısı.
2. [Core](C1_Core.md) — 100–160 saat — G1 araştırma ve H2 kararı.
3. [Hybrid](H2_Hybrid.md) — 90–140 saat — G2 servis ve H1 kararı.
4. [Studio](S3_Studio.md) — 80–130 saat — G3 taşınabilirlik ve kullanıcı işi.

Toplam 330–520 etkin saat; yüzde 25 payla 413–650 saat. Sabit bitiş sözü değildir. İlk iki teknik görev sonunda tahmin güncellenir. 10 saat/haftada yaklaşık 42–65 hafta, 8 saatte 52–82, 12 saatte 35–55 hafta. Gerçek robot ve ileri araştırma bu toplama dahil değildir.

## İlk 90 gün için esnek hedef

İlk 2 hafta kapsam, ortam ve robot tanımı. Sonraki 3–5 hafta FK ve Jacobian. Ardından veri fabrikası, benchmark ve G0 devri. Kalan kapasiteyle C1-01/02/03 başlatılır. Bu dağılım görev saatlerine göre yeniden hesaplanır; 3. ayda Hybrid veya GUI tamamlama garantisi verilmez.

## Her adımın döngüsü

Görev planını oku → girdileri doğrula → en küçük uygulamayı yap → ilgili testi çalıştır → ham kanıtı sakla → sonucu değerlendir → görev kaydını kapat → sonraki göreve devret. Günlük işte esas kayıt [25 görev dosyası ve izlenebilirlik](../TRACEABILITY.md) olur.

## Kapı kararları

G0 yanlışsa eğitim açılmaz. G1 araştırma sonucu olumsuz olabilir; geçerli model H1 için aday olabilir. G2 neural avantaj göstermiyorsa sayısal motor varsayılandır. G3 tamamlandığında desteklenen robot/platform/kontrol kapsamı açıkça yayımlanır. Yeni araştırmalar [ayrı birikim](../RESEARCH_BACKLOG.md) üzerinden seçilir.

## 24 Eylül 2026 · G0 kapanışı

Foundations COMPLETE; F0-06 COMPLETE; T-F09 PASS; G0 PASS / ACCEPTED.
[Karar](../../experiments/F0-06/G0_DECISION.md) ve
[hashli Core devri](../../experiments/F0-06/CORE_HANDOFF.md) hazırdır.
Core READY / NOT_STARTED. C1-01, C1-02 ve C1-03 NOT_STARTED;
ortak F0-06/G0 ön koşulu sağlandı. Sonraki faz çalışması bu kapanışta başlatılmadı.
Yazılım hedefi v0.1.0 korunur; tag/release oluşturulmadı.


## 10 Ekim 2026 · Core devir hazırlığı

ADR-025: C1-06R CLOSED_WITH_UNMET_PRODUCT_TARGET; yeni bağımsız final
NOT_CREATED/NOT_RUN,eski C1-06 H2 REJECTED/T-C05 PASS korunur.
C1-07 IN_PROGRESS/PREPARATION_COMPLETE: FK_TANH3 ana + RAW3 local-only
ikincil aday,model kartı taslağı,hashli manifest ve G1 hazırlık matrisi hazır.
Mevcut ortam21600 prediction bağımsız FK PASS; T-C06 temiz ortam PENDING,
G1 OPEN. Hybrid başlamadı. Kullanıcının ikinci aşaması tamamlandı;
üçüncü aşama akademik süreç raporu ayrı komut bekler.
[Devir kararı](../../experiments/C1-07/preparation/HANDOFF_DECISION.md),
[G1 kalan işler](../../experiments/C1-07/preparation/G1_READINESS.md).


## 11 Ekim 2026 · Core/G1 araştırma kapanışı

**C1-07 COMPLETE / T-C06 PASS; Core COMPLETE / G1 PASS / ACCEPTED.**
Ürün hedefi NOT_MET, direct IK NO_GO, H2 REJECTED. C1-06R durdurma kararı
değişmez: yeni final NOT_CREATED, H2-R final NOT_EVALUATED. Hybrid
NOT_STARTED; sonraki görev H2-01 / REQ-H01 / T-H01.

REQ-C06 → portable inference, nihai model kartı ve hashli devir → temiz
c7798ef clone + taze kilitli ortam → 127 test PASS (0 skip/fail/error),
6 checkpoint × 48 sorgu = 288 birebir q/bağımsız FK metriği.
122 frozen kaynak, 455 C1-06R teslimi ve 16 hazırlık dosyası değişmez.
Akademik negatif sonuç raporu, iki analiz grafiği, LinkedIn taslağı ve
ara verme/geri dönüş paketi tamam. 607 büyük dosyalık yerel arşiv,
üye SHA kontrolü ve boş klasöre geri yükleme ile doğrulandı; uzak arşiv
ve ikinci aygıt NOT_CONFIRMED. Release/tag ve LinkedIn yayını yapılmadı.

[G1 kararı](../../experiments/C1-07/G1_DECISION.md),
[geri dönüş rehberi](../records/CORE_RESUME.md).

# NeuroKinematics ana roadmap

Plan r1 · Başlangıç kapasitesi haftada 8–12 saat · İlk robot KUKA KR6 R900 sixx varsayımı

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

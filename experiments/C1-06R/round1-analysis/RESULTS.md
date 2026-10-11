# C1-06R round1 — Validation sonucu

9 Ekim 2026 · Belge r1 · **Eğitim COMPLETE; bütünlük PASS; validation hedefi NOT_MET.**

Kullanıcı 12 koşuyu tamamladı. Kampanya kaydı 10.250,003 saniye
(2 saat 50 dakika 50 saniye) gösteriyor. Önceki 6–11 saatlik kısa pilot
tahmini bu makinedeki tam koşuyu fazla tahmin etmişti. Bu süre başka
algoritmaya karşı hız üstünlüğü veya sonraki turun garantisi değildir.

## Doğrulanan sonuç

Dört model ailesi × üç seed, her koşuda 2000 epoch ve 120.000 güncelleme;
toplam 1.440.000 güncelleme tamamlandı. 85 ham çıktı dosyasının SHA ve
envanteri, config/runtime/source kimlikleri, epoch kayıtları, optimizer
adım sayısı, başlangıç eşliği ve 24.000 epoch permütasyonu doğrulandı.
Seçilen ve son checkpointlerin validation sonuçları yeniden çıkarımla
**birebir** tekrarlandı. Seçim ön kayıtlı sıralamayı ve erken epoch eşitlik
kuralını izliyor. Hiçbir seed elenmedi; tüm 3600 satır paydada tutuldu.

| Aile | Profil A, her seed | Profil B, her seed | Medyan konum hatası, seed aralığı | Medyan yönelim hatası, seed aralığı | Limit dışı sayısı, seed sırasıyla |
|---|---|---|---|---|---|
| Q | 0/3600 | 0/3600 | 283,2–285,9 mm | 92,1–94,8° | 185 / 81 / 95 |
| FK | 0/3600 | 0/3600 | 165,9–247,7 mm | 27,5–36,2° | 310 / 276 / 309 |
| Q_TANH | 0/3600 | 0/3600 | 197,4–204,5 mm | 73,9–79,7° | 0 / 0 / 0 |
| FK_TANH | 0/3600 | 0/3600 | 128,7–134,7 mm | 25,1–26,0° | 0 / 0 / 0 |

Tablo her koşunun **ön kayıtlı kuralla seçilmiş** checkpointini gösterir.
Seedler: 2026100901, 2026100902, 2026100903. Profil A ≤2 mm ve ≤1°;
Profil B ≤1 mm ve ≤0,5°; sonlu ve limit içi q birlikte zorunludur.
Her modelde main 0/3000, boundary 0/300, singularity 0/300; local 0/1800,
wide 0/1800. Main validation ≥%95 kapısı karşılanmadı.

12 × 80 = 960 kayıtlı validation değerlendirmesinin tamamında A/B sıfır.
Bunlar aynı sorguların tekrarlarıdır; 960 bağımsız test gibi havuzlanmaz.
Son epochu seçmek veya kayıtlı başka bir epochu almak başarı sağlamıyor.
Sürekli hata metriklerinde FK ve tanh katkısı var; bu, operasyonel pose
başarısı veya H2-R desteği değildir. H2-R yeni final kararı **NOT_EVALUATED**.

## Neden incelemesi

1. **Daha uzun eğitim tek başına yeterli olmadı.** Son checkpointlerde
   etiketli train üzerinde Profil A, Q'da 0/0/4, FK'da 0/0/1,
   Q_TANH'da 5/2/2 ve FK_TANH'da 2/0/1; her payda 15.204.
   Küçük 64 örnek tanısında öğrenebilmek tam veri hassasiyetini garanti etmedi.
2. **Eğitim ve validation ayrışıyor.** Örneğin Q_TANH seed1 son epochta,
   tüm train medyanı 38,9 mm/6,78° iken validation 362,0 mm/108,65°.
   Etiketli satırlarda teacher q RMSE train 0,0689 rad, validation 1,7039 rad.
   Bu gözlem aşırı uyumla tutarlı; aynı zamanda train pose hassasiyeti de
   yetersiz. Sorunu yalnız veri azlığına, yalnız kapasiteye veya tek kayba
   atfetmek için henüz nedensel deney yok.
3. **Yakın hedeflerde mevcut durum bilgisi yeterince korunmuyor.** Hiç
   öğrenme yapmadan q_current döndürmek local validation'da medyan
   47,44 mm/7,16° verir. Seçilmiş FK_TANH modelleri aynı 1800 local sorguda
   108,71–116,37 mm/21,58–22,42°. Her ikisi de A 0/1800.
   Bu baseline ürün adayı değildir; mevcut duruma eklem düzeltmesi öğrenen
   bir çıktının kontrollü olarak sınanmasını gerekçelendirir.
4. **Kayıplar arasında ölçülen gradyan çatışması var.** Önceden bu tanı
   için belirlenmiş 64 main/local ve 64 etiketli main/wide train örneğinde,
   FK/FK_TANH × best/last × üç seed = 24 prob çalıştırıldı. Son checkpoint
   problarında Q ile yönelim gradyanının kosinüsü −0,877…−0,550.
   Bu, iki terimin o noktalarda karşıt güncelleme yönleri istediğini gösterir;
   hangi çözüm dalının veya kayıp ölçeğinin nedeni olduğunu tek başına
   kanıtlamaz. Probe sıfır eğitim güncellemesi yaptı.

## Checkpoint uyumluluk kusuru

İlk audit denemesi, production contract içindeki `torch.__version__`
değerinin düz string yerine `TorchVersion` sınıfı olarak pickle'a yazılması
nedeniyle `weights_only=True` yüklemesinde durdu. Hash denetimi geçti;
bu dosya bozulması veya eğitim hatası değildir. Analiz yalnız kurulu
TorchVersion sınıfını geçici izin listesine alarak güvenli yüklemeyi korur.
Eski checkpoint ve dondurulmuş training kodu değiştirilmedi.

Önceki resume testleri basit test contract'ı kullanmıştı; production
metadata kapsamındaki bu kusuru yakalamamıştı. Yeni regresyon testi
kusuru ve dar kapsamlı okuma çözümünü doğrular. Eski launcher'ın kesilmiş
production koşusunu sürdürme yolu bu kusura hâlâ açıktır; mevcut 12 koşu
tamamlandığı için yeniden başlatma gerekmez. Sonraki eğitim sürümünde
metadata düz string olarak kaydedilecek ve gerçek production contract
ile kesinti/devam testi zorunlu olacak.

## Karar

**Round1 bitti; C1-06R araştırması bitmedi.** Yeni uzun eğitim paketi henüz
hazır değil. Mevcut kampanyayı tekrar çalıştırmak gerekmiyor. Yeni final
NOT_CREATED, eski final raw NOT_READ. C1-06 H2 REJECTED ve doğrudan IK
NO-GO korunur; C1-07/G1 açık kalır. Sonraki iş [kontrollü tanı](NEXT_DIAGNOSTIC.md).

Makine kanıtı: [audit.json](audit.json), seed bazlı JSON dosyaları,
[gradient-probe.json](gradient-probe.json), [çalışma kaydı](RUN_REPORT.md).
Ham ağırlıklar ve kaynak eğitim kayıtları `data/generated/C1-06R/round1`
altında Git dışında korunur; bu rapor validation araştırmasıdır.

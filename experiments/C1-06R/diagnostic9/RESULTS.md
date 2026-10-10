# ADR-024 sonucu — Sıfır ofseti düzeldi, yeni köklere genelleme düzelmedi

10 Ekim 2026. DIAGNOSTIC9 COMPLETE. Ürün hedefi NOT_MET.
Ön kayıtlı araştırmaya devam kapısı FAIL; bu MLP başlık/ölçek deney ailesi
STOP. Bu karar bütün neural IK yaklaşımlarının olanaksızlığı anlamına gelmez.

## Eşli sonuçlar

Her hücre aynı512 kök/4096 yön,5000 full-batch update; üç eğitim seed'i.
Profil A: konum≤2mm, geodesic açı≤1°, sonlu ve limit içinde eklemler.
Profil B:1mm/.5°. Checkpoint son adım; validation ile seçim yok.

| Seed sonu | Başlık | Train A /4096 | Aynı kökte yeni yön A /4096 | Validation A /3600 | Validation B /3600 |
|---|---|---:|---:|---:|---:|
| 01 | RAW | 3 | 0 | 0 | 0 |
| 01 | CENTERED | 568 | 60 | 1 | 0 |
| 02 | RAW | 0 | 1 | 0 | 0 |
| 02 | CENTERED | 657 | 70 | 0 | 0 |
| 03 | RAW | 0 | 2 | 0 | 0 |
| 03 | CENTERED | 667 | 82 | 0 | 0 |

Seed'ler2026100901/02/03. Bir başarılı validation örneği main/local içindedir:
ilgili koşuda main A1/3000, local A1/1800, wide A0/1800. Diğer iki seed
bütün validation strata'da A0. Bütün seed'lerde wide A/B0. 351 eksik wide
teacher satırı paydadan çıkarılmadı. Tek başarı %0,0278 overall eder;
tekrarlanabilir genelleme başarısı veya %95'e yakınlaşma ilan etmiyoruz.

| Seed | Local medyan konum RAW→CENTERED | Local medyan açı RAW→CENTERED | Local limit dışı RAW→CENTERED |
|---|---|---|---|
| 01 | 16,04→19,82mm | 4,84→6,83° | 63→112 /1800 |
| 02 | 16,09→20,89mm | 4,84→6,04° | 73→102 /1800 |
| 03 | 16,53→20,59mm | 5,00→6,29° | 70→120 /1800 |

CENTERED local limit dışı oranı her seed'de%5'i aştığı için geçersizleri
sonsuz hata sayan P95 sonlu değil (JSON null). Bunlar eksik ölçüm değil;
sonlu alt kümeyle değiştirilmedi. Local severity medyan ve P95 tüm seed'lerde
kötüleşti. Aynı-kök probe konum medyanı yaklaşık13,3–13,8mm'den7,2–7,3mm'ye
iyileşti; bu iyileşme yeni-kök validation'a aktarılmadı.

## Mekanizma ve türev

CENTERED her seed'de zero train-current512/512 ve zero validation-current
1800/1800 A/B sağlar. Bu2312 sentetik sorgu orijinal validation değildir.
Sıfır hedef farkında yapısal koruma sağlandı; fiziksel float64 dönüşümünde
yaklaşık4,83e-8m medyan yuvarlama farkı kaldı, toleransların çok altında.

Prediction Jacobian ile task-space yerel türev bağıl hatası (ideal0,
hareketsiz düzeltme1; bu sayı bir başarı yüzdesi değildir):

| Seed | Train current medyan RAW→CENTERED | Yeni-kök current medyan RAW→CENTERED |
|---|---|---|
| 01 | 0,556→0,259 | 0,604→0,934 |
| 02 | 0,578→0,254 | 0,635→0,808 |
| 03 | 0,546→0,233 | 0,583→0,806 |

Her grup32 anchor. Seed01 RAW train/validation31/32 geçerli prediction
Jacobian; diğerleri32/32. Bu tabloda geçerli alt kümeler kullanıldığı için
tam eşli32/32 karşılaştırma iddiası yok. Yeni-kök türevi bütün seed'lerde
kötüleşti. Shadow float64 FD h1e-5, atol1e-5/rtol.001 eşliği PASS.
Sabit ağırlıkta merkezleme hedef türevini değiştirmez; burada fark yeniden
eğitimden doğdu. Sıfır ofsetin giderilmesi doğru inverse haritayı öğrenmek
için yeterli değil. Bulgular yerel haritanın yeni konfigürasyonlara aktarım
sorununu destekliyor; veri kapsamı/mimari/kayıp etkilerini tek nedene indirmiyor.

## Ön kayıtlı karar

- Her seed'de local A ve probe A artışı: **FAIL**, local yalnız seed01 artıyor.
- Her seed'de local severity medyan/P95 kötüleşmeme: **FAIL**.
- Üç sabit seed ortalaması için eşli1800 kök bootstrap: ortalama+0,01852
  yüzde puanı, %95 aralık[0;0,05556]; pozitif alt sınır: **FAIL**.
- Centered bütün zero A/B: **PASS**.
- Yazılım, geometri, replay ve bütünlük: **PASS**.

Bootstrap5000 tekrar/seed2026101009; eğitim seed popülasyonu belirsizliğini
ölçmez. Aynı validation tekrar tekrar araştırmada kullanıldı; bağımsız final
değildir. Ön kaydın kapsam büyütme koşulu sağlanmadı; otomatik root-scaling
veya uzun eğitim başlatılmayacak. İyileşen train/probe sonuçları korunur,
başarısız genelleme sonucu gizlenmez.

## Maliyet ve doğrulama

30000 update,122880000 satır maruziyeti;535558 parametre her hücrede aynı.
RAW eğitim20,76–21,12s;CENTERED32,14–32,76s (GPU synchronize ölçümü).
Eşit update/parametre; CENTERED iki ağ değerlendirmesi yapar, eşit compute
değil. Bunlar inference veya CPU servis gecikmesi benchmarkı değildir.
Tüm komut191,69s; ölçülen modül187,13s. Donanım RTX5060 Laptop.

72 test PASS. İlk RAW önceki n512 RAW ile exact tensor/dört set metrik replay.
Altı checkpoint weights_only reload ve altı set exact replay.87696 prediction
bağımsız FK/atan2,384 model-anchor türevi replay;122 frozen kaynak değişmez.
[Audit](audit.json),[kayıt](RUN_REPORT.md),[config](config.json),
[yön ve durma kuralları](RESEARCH_DIRECTION_AND_STOP_RULES.md).
Yeni final NOT_CREATED; eski final NOT_READ. C1-07/G1 başlamadı.

Önerilen sonraki iş artık yeni bir küçük neural deneyi değil: C1-06R'nin
negatif ürün sonucu ve yararlı tanılarını kapanış/devir kararında birleştirmek;
C1-07 model kartı/T-C06/G1 koşullarını tamamlamak; ardından kullanıcının
hibrit önceliğine uygun aynı sayısal motor üzerinde neural başlangıç faydasını
H1 protokolünde ölçmek. CENTERED, tek validation başarısı nedeniyle en iyi
ürün veya en iyi hibrit aday seçilmedi; bu tur hiçbir hibrit fayda ölçülmedi.

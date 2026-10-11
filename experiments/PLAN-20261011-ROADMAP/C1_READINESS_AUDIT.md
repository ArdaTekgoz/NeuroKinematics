# C1 hazırlık denetimi — 11 Ekim 2026

Karar: **H2-01'e giriş uygun; genel neural ürün başarısı ve tüm sistem stabilitesi kanıtlanmış değil.**
Yeni eğitim bu çalışmada yapılmadı. Denetim kod, test ve mevcut ham kanıt bütünlüğünü kapsar.

| Bileşen / iddia | Kanıt ve sonuç | H2 kararı |
|---|---|---|
| Robot, birimler, frame/TCP, limitler | Kilitli kimlikler ve 277 F0-01/02/03 regresyon testi yeniden PASS | Değişmez referans |
| Araştırma kapanışı | C1-07 G1 PASS; eski doğrudan neural sonuç NO_GO, H2 hipotezi REJECTED | Geçerli negatif sonuç; Hybrid'e girişe engel değil |
| Eski neural modeller | Altı devredilmiş checkpoint'in hash'i doğrulandı; yeniden eğitim yok | Kontrollü H1 karşılaştırması; otomatik varsayılan değil |
| C1-06S geometrik IK | 38 birim testi yeniden PASS; 12.000 final sorgu ve 60.000 ölçümün ham hash'leri eş | KUKA için doğrulanmış aday/öğretmen/baseline |
| Final başarı | Kayıtlı ana 10.000/10.000, sınır 1.000/1.000, tekillik 1.000/1.000; A/B PASS | Örneklenmiş dağılımda başarı; bütün uzay için ispat değil |
| Süre | Kayıtlı P95 0,5029 ms; maksimum 103,1543 ms; 59.999/60.000 ölçüm 10/50 ms içinde | Ortak deadline servisi H2-02'de zorunlu |
| Çarpışma ve yol | C1-06S NOT_CHECKED; dal yakınlığı yörünge sürekliliği garantisi değil | H2-04 ve sonraki TR/MP zorunlu |
| Taşınabilirlik | 12.000 clean-source replay EXACT, aynı kilitli runtime; taze ortam oluşturulmamış | H2-06 yeni ortam/servis kabulü zorunlu |
| Genel robot desteği | AnalyticIK sabit KR6 geometrisine bağlı; base-axis singular durumda sonlu q1 taraması var | Diğer robotlarda kullanılamaz; MR capability registry gerekir |

## Bu tur gerçekten çalıştırılanlar

`pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_roadmap_readiness.py`

- 343 hash karşılaştırması PASS: tarihsel training freeze, geometrik freeze ve
  kanıt manifesti, C1-07 kaynak/teslim listeleri, altı checkpoint ve yeni final ham dosyaları.
  Bu sayı farklı manifestlerde yinelenen dosyaları da içerir; 343 benzersiz dosya iddiası değildir.
- Ayrı süreçlerde C1-06S 38, F0-01 16, F0-02 102, F0-03 159: **315 PASS**, skip/fail/error 0.
- [Makine çıktısı](audit/summary.json), [bütünlük](audit/integrity.json), aynı klasörde JUnit ve loglar.
- Yeni eğitim, yeni benchmark, tüm Core eğitim testleri, yeni temiz ortam,
  ONNX, servis deadline/collision, fiziksel robot bu tur **NOT_RUN**.

## Sıfır neural sonucun yorumu

Sıfır sonuç C1-06'da aniden başlamadı: C1-04/05'te de doğrudan IK başarı sayacı
sıfırdı. Decoder sınır kusuru ve metadata kusuru bulundu; düzeltmeler genelleme
problemini çözmedi. Sentetik pose'ların çözülebilir olduğu C1-06S validation
3.600/3.600 ile gösterildi; sentetik veri dağılımının her öğrenme amacı için yeterli
olduğu veya fiziksel dünyaya aktarımın doğrulandığı sonucu çıkmaz.

8.400 eğitim kökü/16.800 çift, eksik wide teacher etiketleri, çoklu çözüm dalları,
kayıp hedeflerinin çatışması ve yerel inverse davranışı birlikte değerlendirilmelidir.
Tek bir neden kesinleştirilmedi. 64 örnekte ezberleme kapısı, yeni kök/yön
genellemesini kanıtlamıyordu. Yeni eğitimde bu ölçümler ayrı kapılar olacaktır.
[Kök neden kaydı](../C1-06S/ROOT_CAUSE_AND_RECOVERY.md).

## Başarılı çözücü ne işe yarayacak?

1. KUKA üzerinde hedef pose için bağımsız FK ile doğrulanmış joint adayı sağlar.
2. Yeni veride eksik teacher'ı doldurmaya ve dal etiketlemeye yardımcı olur.
   Mevcut API tek aday döndürür; bütün dallar için ayrıca denetlenmiş API gerekir.
3. Neural/hybrid yöntemin geçmesi gereken güçlü bir ürün karşılaştırması sağlar.
4. Neural katkı yokken de kinematik araştırma ve servis geliştirmesinin sağlam
   bir bileşenle ilerlemesini sağlar. Neural bilimsel katkının yerine geçmez.

Eski başarısız ağı zorunlu çözüm motoru yaparak ilerlemiyoruz. H2-N1/N2 yeni
eğitimi açık iş olarak tanımlar; H2-03 katkıyı ölçer. Güvenilirlik deterministik
denetimde, öğrenilmiş fayda ayrı deney kapısındadır.

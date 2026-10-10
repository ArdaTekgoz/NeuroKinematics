# NeuroKinematics hedefi, araştırma sınırı ve hibrit ilerleme kararı

10 Ekim 2026 · Belge r1 · Core araştırma yönü; G1/Hybrid kabulü değildir.

Kullanıcının bu turdaki açık tercihi: hibrit bir sistem; ters kinematikte
yüksek başarı ve gerektiğinde yapay zekâ desteğiyle güvenilirlik. Buna göre
ürün amacı, ağın tek başına her sorguyu çözmesini şart koşmadan, doğru kabul
ve hata davranışıyla bütçe içinde doğrulanmış IK çözümü döndürmektir.
Profil A/B ve limit koşulları korunur. Neural katkı, toplam sisteme ölçülen
faydasına göre seçilir. Bu tercih ürün başarı eşiğini düşürme izni değildir.

Tamamlanan ADR-024 kararı: CENTERED validation A1/0/0 (her biri3600),
local medyan/P95 ve türev daha kötü; ön kayıtlı devam kapısı FAIL.
Bu nedenle aşağıdaki koşullu yolun durma kolu uygulanmıştır. Yeni bir
MLP mikro-deneyi veya kök büyütme kampanyası başlatılmayacaktır.
[Tam sonuç](RESULTS.md), [audit](audit.json).

## Üç farklı soru

1. **Doğrudan neural IK:** Ağın tek çıktısı sıkı FK/limit testini sağlıyor mu?
   Önceki 0/3600 bu sorunun cevabıdır. Sıfır hareket sentetik testi veya
   ortalama hatanın azalması onun yerine konamaz.
2. **Öğrenilmiş başlangıç:** Ağın önerisi, aynı sayısal motorun hedefe daha
   hızlı/güvenilir ulaşmasına yardım ediyor mu? Küçük FK hatası bunu garanti
   etmez; limitler, çözüm dalı, conditioning ve yakınsama davranışı önemlidir.
3. **Güvenilir servis:** Son çözüm bağımsız doğrulamadan geçiyor mu; geçmezse
   doğru başarısızlık nedeni ve süre durumu veriliyor mu? Sayısal veya neural
   alt bileşenin tek başına başarısı uçtan uca servis başarısı değildir.

## 0/3600 ne söyler, ne söylemez?

Önceki sabit modellerin o 3600 sorguda 2mm ve1° koşulunu birlikte, limitler
içinde sağlayamadığını söyler. Bütün tahminlerin rastgele olduğunu veya IK
probleminin çözümsüz olduğunu söylemez. ADR-022 tanık kontrolünde mevcut
validation köklerinin geçerli çözümleri 3600/3600 A/B sağlamıştır. Tanık,
modelin erişebileceği öğrenme başarısına ilişkin bir üst sınır tahmini değildir.

Önceki en yararlı yerel kontrolün medyanı 16,04mm/4,84° idi. Aynı modelde
eşikleri 10mm/5° yapmak bile yalnız271/1800 local başarı veriyordu; problem
yalnız eşik çevresindeki yuvarlama değildir. Bu geniş tolerans bir duyarlılık
analizidir, kabul kuralı değildir. Modeller hedefe kısmen yaklaşıyor fakat
gerekli hassasiyet ve yeni köklere genelleme eksik kalıyor.

%95'e çıkılabilir, %50'de kalınır veya bu mimari asla çalışmaz diyebilecek
kanıtımız yok. Donanım/süre artırımı bilimsel bir başarı garantisi değildir.
Yerel-only öğrenmede wide başarısı sıfır kalırsa yarı local/yarı wide üründe
local başarı %100 olsa bile toplam başarı %50'yi aşamaz. Bu aritmetik sınır,
tüm olası mimarilerin kapasitesine ilişkin bir iddia değildir.

## Nerede yanlış düşündük, hangi nedenler açık?

Küçük64 örneği ezberleme kontrolünden uzun kampanyaya geçiş fazla iyimserdi.
Bu kontrol optimizasyonun çalışabildiğini gösterir; yeni yön ve yeni kök
genellemesini göstermez. Bu ayrımı sonradan ayrı ölçümlere böldük.
Round1'in uzun eğitimi genel başarıyı getirmedi; bazı validation hataları
büyüdü. Dolayısıyla aynı eğitimi daha uzun çalıştırmak için mevcut gerekçe yok.

Gerçek bir float32 sınır dönüşümü kusuru bulundu ve yeni sürümde düzeltildi;
tek başına 0/3600'ü açıklamadı. Robot, birimler, eklem sırası, TCP, FK,
Jacobian, gradient, label ve veri kökeni çoklu kontrollerden geçti. Bu
sonuçlar kontrol edilen kapsamda yeni bir Foundations kusurunu desteklemiyor;
projenin her satırının kusursuz olduğunu kanıtlamıyor. Foundations'ı varsayımla
değiştirmek yerine somut bir karşı örnek bulunursa sürümlü kusur kaydı açılır.

Yerel yön çeşitliliği sürekli hatayı düşürdü; yalnız z-score değişimi train
başarısını yükseltirken genellemeyi kötüleştirdi. Sıfır ofseti ve türev tanısı
yapısal eksiklik gösterdi. ADR-024 bu eksikliklerden yalnız sıfır ofsetini
izole eder. Veri kökü kapsamı, yerel inverse haritasının karmaşıklığı,
temsil/kayıp ve model kapasitesinin ortak etkisi hâlâ ayrıştırılmış değildir.
Wide çoklu çözüm/dal problemi olasıdır; bunu local başarısızlığın kesin tek
nedeni diye sunmuyoruz.

## Bu deneyler nereye kadar?

ADR-024'ün altı koşusu sabittir: üç seed × RAW/CENTERED, 5000'er update.
Validation sonucuna bakarak adım sayısını artırma veya başarılı seed seçme yok.
Araştırma devam kapısı config/ADR'de eğitim öncesi yazılmıştır: her üç seed'de
aynı-kök probe ve local validation A artışı, local medyan/P95 severity'nin
kötüleşmemesi, eşli kök bootstrap alt sınırının pozitifliği, tüm zero A/B ve
bütünlük kontrolleri. Bu kapı %95 ürün kabulünün yerine geçmez.

- Kapı geçmezse bu MLP başlık/ölçek mikro-deney ailesi durur. Yeni mimari,
  geometri güdümlü öğrenme veya büyük veri kampanyası ayrı araştırma kapsamı
  gerektirir; C1-07'nin belirsiz süreli ön koşulu yapılmaz.
- Kapı geçerse en fazla bir ön kayıtlı kök kapsamı doğrulaması düşünülebilir.
  O deneyde kök sayıları, bütçe, seed'ler, seçim ve durma ölçütleri baştan
  dondurulur. Bu tur o deney tasarlanmış/çalıştırılmış sayılmaz. Kullanıcının
  hibrit önceliği nedeniyle olumlu kapı bile yeni doğrudan-IK kampanyasını
  zorunlu kılmaz; G1 teslimine geçiş önerisi ayrıca değerlendirilebilir.
- Hibrit H1 için mevcut faz raporu hata sınıfı başına en fazla iki hedefli
  düzeltme turu öngörür. Her başarısız sonuçtan sonra sınırsız yeni deney yok.
  Araştırma sorusu/kanıt değişmedikçe aynı başarısız kampanya tekrarlanmaz.

## Core kapanışı ve sonraki yol

Core raporu §6, %95'i başlangıç ürün hedefi olarak tanımlar; araştırmanın
bilimsel kapanışı için olumlu sonuç zorunlu değildir. Kritik doğruluk,
tekrar üretim, baselinelar ve açık hipotez kararı zorunludur. Dolayısıyla
doğrudan neural hedef NOT_MET olarak kalırken geçerli negatif araştırma
sonucuyla Core kapanışı mümkündür. Bu kapanış otomatik verilmez.

Önerilen sıra:

| Sıra | İş | Tamamlanma kanıtı |
|---|---|---|
| Şimdi | ADR-024 sonlandırma ve C1-06R araştırma kararını yazma | Bağımsız audit, tüm seed sonuçları, açık sınırlar |
| Sonra | C1-06R kapsamının kapanış/devrini netleştirme | Olumsuz ürün sonucu, aday seçimi gerekçesi; çalıştırılmayan final açık |
| C1-07 | Model kartı, temiz ortam çıkarım/değerlendirme, manifest, G1 | T-C06 ve gerekçeli G1 kararı; yalnız .pt dosyası yeterli değil |
| G1 sonrası H2-01/02 | Hata/kabul sözleşmesi ve bütçeli sayısal+neural çözüm hattı | Her adayın doğrulanması; fallback ve deadline davranışı |
| H2-03 | Aynı çözücüde farklı başlangıçları eşli kıyaslama | Başarı, P50/P95/P99, iterasyon ve güven aralıkları |

H1'de motor, residual/Jacobian, tolerans, iterasyon ve toplam süre bütçesi
aynı kalır. q_current, merkez, klasik restart ve neural başlangıç politikaları
karşılaştırılır. Neural ön işlem/çıkarım/doğrulama maliyeti toplam süreye
dahildir. CPU batch1 gerçek servis ölçümü, GPU eğitim süresinden çıkarılamaz.

Mevcut H1 hedefi ana kümede P95 toplam sürede en az%20 azalma, başarı
kaybında en fazla1 yüzde puanı; eşli%95 güven aralıklarıyla destek. Bu,
başlangıç motorunun mutlak başarısı düşükken sistemi yeterli ilan etme
ölçütü değildir. Mutlak Profil A/B, zor kümeler, deadline ve yanlış-kabul
sonuçları ayrıca raporlanmalıdır. “Güvenilir” başarısızlığı gizlemek veya
her isteğe çözüm garantisi vermek değildir.

Neural fayda sağlarsa desteklenen hibrit aday olur. Fayda sağlamazsa sayısal
motor varsayılan kalır; neural araştırma modu açık etiketlenir. Sayısal motor
da ürün hedefini karşılamazsa çözücü/iş yükü/bütçe sorununa dönülür; eşik
düşürülüp başarı ilan edilmez. Gerekirse desteklenen görev kapsamı açık bir
yeni ürün sözleşmesiyle daraltılır ve tüm kapsam sonucu ayrıca korunur.
Studio, ilgili faz kapıları sağlandıktan sonra güvenilir sayısal servisle
ilerleyebilir. Kinematik geçerlilik çarpışmasızlık veya fiziksel güvenlik değildir.

Öğrenilmiş yaklaşık çözümleri sayısal yöntemle iyileştirme yolu literatürde
de vardır: [IKFlow](https://arxiv.org/html/2111.08933v3), özellikle VI-B/VI-C.
Bu çalışma bizim robotumuz, batch1 iş yükümüz veya sıkı ortak başarı
ölçütümüz için performans garantisi vermez; yerel H1 deneyi gerekir.

## Kaynaklar ve durum sınırı

[Core §6–7](../../../docs/raporlar/02_Core_v1_0_r1.md),
[Hybrid §4,6–7](../../../docs/raporlar/03_Hybrid_v2_0_r1.md),
[C1-07](../../../docs/tasks/C1-07.md),
[Hybrid roadmap](../../../docs/roadmaps/H2_Hybrid.md),
[önceki kıstas denetimi](../diagnostic7/CRITERION_AND_DIRECTION_REVIEW.md),
[ADR-024](../../../docs/adr/ADR-024-c106r-centered-residual.md).
Yeni bağımsız final NOT_CREATED;
eski final NOT_READ. Bu belge C1-07/G1/H2 testlerinin çalıştırıldığı anlamına gelmez.

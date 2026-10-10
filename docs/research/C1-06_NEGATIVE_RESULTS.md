# NeuroKinematics Core negatif sonuç araştırma raporu

Arda Tekgöz · 11 Ekim 2026 · Belge r1 · C1-06 ve C1-06R

Bu metin proje sonunda hazırlanacak akademik makale için yöntem, negatif
bulgu ve geçerlilik sınırlarını koruyan teknik araştırma raporudur. Hakemli
yayın değildir. Deneyler proje sahibi tarafından yürütülmüş; kod, denetim
ve belge hazırlığında yapay zekâ desteği kullanılmıştır. Makale yazarlığı,
katkı beyanı ve son bilimsel sorumluluk yayımdan önce insan yazar tarafından
gözden geçirilmelidir. Aşağıdaki sayılar sürümlü depo kanıtlarına dayanır.

## Özet

KUKA KR6 R900 sixx için durumla şartlandırılmış doğrudan neural ters
kinematik modelleri, kinematik doğruluk kontrollerinden geçen bir yazılım
zincirinde değerlendirildi. Sıkı ortak başarı; eklem limitleri içinde sonlu
çıktı, en fazla 2 mm konum ve 1 derece yönelim hatası olarak tanımlandı.
Özgün C1-06 değerlendirmesinde 21 checkpoint'in her biri 12000 sorguda
Profil A/B sıfır başarı verdi [K1]. Sonraki C1-06R uzun validation
kampanyasında 12 koşu ve 1440000 optimizer güncellemesi başarı sağlamadı
[K2]. Temsil, optimizasyon, kapasite, kayıp, yerel yön kapsamı, ölçekleme
ve sıfır-hareket davranışı kontrollü tanılarla incelendi [K3–K10].

Son üç-seed merkezlenmiş residual deneyinde eğitim başarısı artmasına
karşın validation başarıları 1/3600, 0/3600 ve 0/3600 kaldı; yeni köklerde
medyan hatalar ve limit uyumu kötüleşti. Ön kayıtlı devam kapısı geçilmedi.
Sonuç, incelenen yöntemlerin veri ve bütçe koşullarında ürün hedefini
karşılamadığını gösterir. Neural IK'nin genel olarak olanaksız olduğu,
tek bir kesin kök neden bulunduğu veya hibrit sistemin başarısız olacağı
sonucu çıkarılamaz. Araştırma, olumsuz sonucu koruyarak durduruldu [K11].

## Araştırma soruları ve kapsam

Birinci soru, supervised eklem tahminine FK/limit bileşenleri eklenmesinin
operasyonel IK başarısını iyileştirip iyileştirmediğidir. Özgün H2 kontrastı
ve kabul kuralı C1-06 ön kaydında sabittir; bu kontrastın kayıp ve çıktı
başlığı bileşik etkisini içerdiği sınırlılık olarak korunur [K1]. İkinci
soru, başarısızlık sonrasında belirli mekanizmaların ayrı kontrollerle
açıklanıp açıklanamayacağıdır. Üçüncü soru, negatif doğrudan-IK sonucundan
hangi geçerli araştırma çıktılarının sonraki hibrit faza taşınabileceğidir.

Yüzde 95 başlangıç ürün hedefi, bilimsel araştırma kapanışıyla eşanlamlı
değildir. Core raporu kritik doğruluk, tekrar üretim, baseline ve açık
hipotez kararını zorunlu kılar [K12]. Hedefe ulaşılmadığı için eşikler
gevşetilmedi. Bu rapor collision-free çözüm, fiziksel robot doğruluğu,
gerçek zaman garantisi veya başka robotlara genelleme iddiası içermez.

## Robot ve ölçüm sözleşmesi

Altı döner eklem manifest sırasıyla joint_1–joint_6; konum metre, eklemler
radyan; base_link–tool0 TCP sözleşmesi kullanılır. URDF, robot spec,
manifest ve TCP dosyaları SHA256 ile kilitlidir. Bağımsız NumPy FK,
Pinocchio referansından ayrı uygulanmıştır; iki uygulamanın aynı URDF'ye
dayanması ortak model hatasını dışlamaz. Jacobian ve finite-difference
kontrolleri öğrenme öncesinde ve seçili tanılarda yürütülmüştür [K5,K12].

Profil A: e_p = norm(p_pred-p_target) ≤ 0,002 m ve geodezik dönme hatası
e_R ≤ 1 derece; ayrıca bütün eklemler sonlu ve gerçek limitler içindedir.
Profil B aynı koşulları 0,001 m ve 0,5 derece ile uygular. Bir bileşenin
tek başına geçmesi ortak başarı sayılmaz. Geçersiz sonuçlar paydadan
çıkarılmaz. Tam paydalı hata özetlerinde geçersizler sonsuzdur; sonlu
olmayan quantile JSON'da null ile temsil edilir. Bu, eksik satır değildir.

## Veri ve değerlendirme tasarımı

C1-02 train 16800, validation 3600 satırdır. Train'de 15204, validation'da
3249 teacher etiketi mevcuttur. Validation'ın 1800 satırı local, 1800'ü
wide; 3000'i main, 300'ü boundary, 300'ü singularity grubudur. Eksik351
wide teacher, FK ile hedef başarısı ölçülebileceği için validation
paydasında tutulur. Train-only normalizasyon ve kök grubu ayrımı korunur.
Öğretmenin başarısız olması hedefin erişilemez olduğu anlamına gelmez.

Özgün C1-06'nın 12000 sorguluk değerlendirmesi ile C1-06R'nin3600
validation sorgusu farklı protokollerdir; sayıları birleştirilmez. C1-06R
yeni bağımsız finali hiç oluşturulmadı. Araştırma boyunca validation
tekrar kullanıldı; son tanılar bağımsız doğrulayıcı test olarak sunulamaz.
Bu rapor için eski final raw yeniden açılmadı; yayımlanmış kabul ve özet
kayıtları kullanıldı. Bir sorgunun çok model/epoch/geçişte ölçülmesi yeni
bağımsız örnek oluşturmaz. Üç eğitim seed'i geniş seed popülasyonunu
temsil etme gücü sınırlı bir tasarımdır.

## Deney dizisi

Aşağıdaki kronoloji toplam bir hiperparametre taramasını tek ön kayıtlı
deney gibi sunmaz. Her tanı önceki bulgudan sonra ayrı revizyon ve sınırla
tanımlanmıştır. Aynı gün yapılan deneylerin numarası nedensel kanıtın
gücünü tek başına artırmaz. Erken overfit kontrolleriyle genelleme testleri
ayrı tutulmalıdır.

| Çalışma | Sınanan değişiklik | Temel gözlem |
|---|---|---|
| C1-06 | Özgün ablation ve final | 21 checkpoint, her birinde A/B0/12000; H2 reddedildi |
| Round1 | Q/FK × linear/tanh ×3 seed; 2000 epoch | 12 koşu, 1440000 update; bütün validation A/B0/3600 |
| Tanı2 | 64/512, local/mixed, absolute/residual; optimizer takibi | 64 örnek ezberlenebildi, validation0; gerçek decoder kusuru bulundu |
| Tanı3 | Ham/göreli pose,512/2048 örnek | 16 koşu; bazı local hatalar düştü, bütün validation0 |
| Tanı4 | Optimizer, Q ölçeği, genişlik256/512 | Geniş model train A3/2048; validation0 |
| Tanı5 | FK/Jacobian, koşulluluk, loss; Q/POSE_A takibi | Geometri PASS; train A57/33, validation iki kolda0 |
| Tanı6 | Tek yön tekrarı / kök başına yön çeşitliliği | Probe sürekli hatası iyileşti; A0 |
| Tanı7 | RAW / local train-only z-score | Train arttı, probe ve validation hatası kötüleşti |
| Tanı8 | Sabit ağırlıkta sıfır hareket ve türev | Gereksiz düzeltme ve eksik yerel tepki ölçüldü |
| Tanı9 | RAW / merkezlenmiş residual ×3 seed | Zero düzeldi; validation1/0/0; genelleme kapısı başarısız |

Round1'in kayıtlı süresi10250,003 saniyedir. Eğitim zamanı, farklı
platformdaki sayısal baseline'a göre hız üstünlüğü değildir. Tanı2–8'in
çoğu tek seed üzerinde yürütülmüştür. Tanı9 üç-seed kontrolü daha önceki
tek-seed hipotezlerin tamamını bağımsız biçimde doğrulamaz [K2–K10].

## Doğrulanmış kusurlar ve açıklayamadıkları

Float32 normalized-to-joint dönüşümü, doğru teacher çıktılarının100/15204
train ve19/3249 validation satırını katı limitlerin çok az dışına taşıdı.
Yeni endpoint-exact float64 decoder ile teacher'ların tamamı A/B geçti.
Ancak aynı ağırlıklarla eski12 ve tanı2'nin8 modelinin validation başarısı
yine sıfır kaldı. Kusur gerçektir; yaygın başarısızlığın yeterli açıklaması
değildir. Sırf daha çok başarı saymak için limit toleransı genişletilmedi [K3].

Round1 checkpoint metadata'sındaki TorchVersion sınıfı weights-only
yükleme/resume uyumsuzluğu oluşturdu. Yeni tanı checkpointleri sürümü düz
string kaydetti. Tarihsel launcher korunur ve yeni çalışma için önerilmez.
Bu kusur dosya kullanımını etkiler; öğrenilmiş geometrik hatayı açıklamaz.
Diğer başarısız komutlar ve test ortamı sorunları loglarda korunmuştur;
çalıştırma altyapısı hataları model performans başarısızlığıyla karıştırılmaz.

## Mekanizma bulguları

Göreli pose gösteriminde local2048 residual modelinin medyan local
validation hatası47,82mm/8,05 dereceden19,57mm/4,98 dereceye indi;
başarı yine sıfırdı. Train A da0/2048 olduğundan sorun yalnız unseen-root
genellemesi değildir; incelenen model/kayıp/bütçe eğitim hassasiyetini de
karşılayamadı [K4].

Genişlik512 Q kaybını azaltırken yüksek pose hassasiyeti sağlamadı.
Geometri tanısında4096 current/teacher FK ve Jacobian karşılaştırması
geçti;32 finite-difference kontrolü de geçti. İyi koşullu alt kümede
başarısızlık sürdü; local quaternion'lar pi süreksizlik sınırına yakın
değildi. Bu bulgular belirli açıklamaları zayıflatır, bütün Foundations
kodunun kusursuz olduğunu kanıtlamaz [K5].

Normalized Q hatasıyla operasyonel pose hatası aynı amaç değildir.
Profil A ölçekli pose loss'a geçiş bazı sürekli ölçümleri iyileştirdi,
fakat aynı başlangıç checkpoint'inden eşli5000 adım sonunda validation
başarısı getirmedi. Yalnız kayıp değişiminin çözüm olduğu desteklenmedi.
Veri kökü kapsamı ve haritanın temsili açık hipotezler olarak kaldı [K5].

Tanı6'da512 kökte yön çeşitliliği, aynı-kök probe konum medyanını40,10mm'den
13,79mm'ye düşürdü. Tanı7'de yerel z-score train A3'ten60/4096'ya
çıkarken probe konum medyanı13,79'dan20,49mm'ye kötüleşti. Daha iyi train
uyumu daha iyi genellemeyi garanti etmedi. Tek seed ve belirli bütçe
sınırı korunmalıdır [K6,K7].

Tanı8'de512 RAW, target=current sentetik sorgularda validation-current
A96/1800, medyan5,61mm/1,25 derece verdi; current'ı aynen döndüren tanık
bütün zero sorgularını çözdü. Türev denetimi ağın kısmi yerel tepki
öğrendiğini fakat ideal inverse tepkiyi vermediğini gösterdi. Çok küçük
perturbasyonlarda hareketsiz baseline da tolerans içinde kalabildiğinden
tek başına küçük-hedef başarı oranı doğru türev kanıtı değildir [K8].

## Üç seed ile merkezleme kontrolü

Tanı9:512 kök,4096 yön,her hücre5000 full-batch AdamW adımı;
535558 parametre. RAW current_norm+g(x), CENTERED current_norm+g(x)-g(x0)
kullanır; x0 aynı current ve sıfır göreli pose'dur. Aynı seed'de başlangıç
tensorları eşittir. CENTERED iki forward yapar; eşit update eşit hesap
süresi değildir. İlk RAW eski kontrol tensor ve metriklerini birebir
üretmiştir. Son checkpoint kullanılmış, validation'da en iyi seed seçilmemiştir.

| Seed sonu | Train A RAW / CENTERED | Yeni yön A RAW / CENTERED | Validation A RAW / CENTERED |
|---|---:|---:|---:|
| 01 | 3 /568 | 0 /60 | 0 /1 |
| 02 | 0 /657 | 1 /70 | 0 /0 |
| 03 | 0 /667 | 2 /82 | 0 /0 |

Train ve yeni yön paydaları4096; validation3600. Eğitim seed'leri
2026100901/02/03. Validation B bütün hücrelerde0. CENTERED zero sorguların
2312/2312'sinde A/B sağladı. Buna karşılık local validation konum medyanı
RAW16,04/16,09/16,53mm'den CENTERED19,82/20,89/20,59mm'ye; yönelim
4,84/4,84/5,00 dereceden6,83/6,04/6,29 dereceye yükseldi.

Local limit dışı sayıları63/73/70'ten112/102/120'ye çıktı. Son durumda
geçersiz oranı yüzde5'i aştığı için tam-payda P95 sonlu değildir. Yeni-kök
task türev bağıl hatası medyanları0,604/0,635/0,583'ten0,934/0,808/0,806'ya
kötüleşti. Bu hata bir yüzde başarı ölçüsü değildir; ideal0, hareketsiz1'dir.
İlk RAW seed'de geçerli prediction Jacobian kapsamı31/32, diğerlerinde32/32
olduğu için bu medyanlar tamamen aynı geçerli alt küme iddiası taşımaz.

Ön kayıtlı devam kapısı her seed'de local ve yeni-yön A artışı, local
severity medyan/P95 kötüleşmeme, zero ve integrity geçişi ve pozitif
bootstrap alt sınırı gerektiriyordu. Local ortalama kazanç0,01852 yüzde
puanı; sabit üç seed'e koşullu1800 kök bootstrap yüzde95 aralığı
[0;0,05556] oldu. Gate FAIL. Bu aralık tüm eğitim seed'lerinin
belirsizliğini veya validation adaptasyonunun etkisini ölçmez [K9,K10].

## Bilimsel yorum ve başarısızlık nedenleri

Desteklenen sonuç: incelenen MLP aileleri yeterli hassasiyet ve yeni
konfigürasyonlara genelleme sağlayamadı. Bazı değişiklikler continuous
hatayı veya train başarısını iyileştirdi; ortak operasyonel kriteri
sağlamadı. Sıfır residual koşulu yapısal olarak düzeltilebildi; doğru
yerel haritanın öğrenilmesine yeterli olmadı. Yapısal invariant ile
öğrenilmiş genelleme farklı değerlendirme hedefleridir.

Kesinleşmeyen açıklamalar: daha yoğun kök kapsamı, geometri güdümlü
temsil, farklı kapasite/optimizer, tek çıktı ile dal seçimi ve kayıp
tasarımının ortak etkisi. Bunların hiçbirine tek başına nedensel suç
atanamaz. Wide başarısızlık çoklu çözümle ilişkili olabilir; local train
başarısızlığını tek başına açıklamaz. Daha uzun eğitimin imkânsızlığı
kanıtlanmadı; mevcut sonuçlar tekrar uzun kampanya için gerekçe sağlamadı.

Önceki yöntemsel aşırılık,64 örnek üzerinde kusursuz öğrenmeyi uzun
kampanya için yeterli geçiş işareti saymaktı. Sonraki analiz bunu düzeltti:
ezberleme, aynı kökte yeni yön ve yeni kökte genelleme ayrı ölçüldü.
Bu süreç değişikliği gelecek deney tasarımı için somut bir ders olarak
korunmalı; başlangıç kararları sonradan kusursuz tasarlanmış gösterilmemelidir.

## Geçerliliğe yönelik tehditler

Tek robot ve sentetik model; fiziksel kalibrasyon eksikliği; aynı URDF'den
kaynaklanan ortak hata; teacher seçimi yanlılığı; bazı tanılarda etiketli
mixed köklere koşullama; az seed; adaptif validation kullanımı; unequal
compute ve farklı gerçekleşmiş eğitim bütçeleri temel sınırlılıklardır.
Local-only sonuçlardan wide ürün başarısı çıkarılamaz. Hiçbir solver
başarısızlığı erişilemezlik kanıtı değildir. Yaklaşık veya geçerli-only
hata ortalamaları strict başarı yerine kullanılamaz.

Özgün C1-06 eşli bootstrap'ın sıfır farkı ve [0,0] aralığı ampirik örnekte
sıfır gözlemden kaynaklanır; popülasyonda başarı olasılığının kesin sıfır
olduğunu kanıtlamaz. Bu rapor yeni p-değeri üretmez ve tanılar arasından
yalnız olumlu sonuç seçip doğrulayıcı sonuç ilan etmez. Etki büyüklükleri,
kapsam, başarısız sonuçlar ve denenmemiş hipotezler birlikte sunulur.

## Proje kararı ve makalede kullanılabilecek iddialar

C1-06R CLOSED_WITH_UNMET_PRODUCT_TARGET; yeni final NOT_CREATED ve H2-R
yeni finalde NOT_EVALUATED. Özgün C1-06 H2 REJECTED değişmez. Core'un
bilimsel kapanışı, C1-07 temiz tekrar ve G1 kararı ayrı belgelerde izlenir;
bu rapor tek başına G1 kabulü değildir [K11,K12].

Makalede savunulabilir katkı; yanlış başarı üretmeyen doğrulama zinciri,
denetimli negatif sonuç dizisi, düzeltilen fakat ana sonucu açıklamayan
sayısal kusur, eğitim/yön/kök genellemesinin ayrılması ve açık durma
kararıdır. Başarısızlık tek başına yayın yeniliği değildir; ilgili
literatürle kapsamlı konumlandırma, açık artifact erişimi ve insan
bilimsel değerlendirmesi gerekir. Bu rapor bu işleri tamamlandı saymaz.

Öğrenilmiş yaklaşık çözümleri sayısal yöntemlerle iyileştirmek literatürde
mevcuttur. IKFlow bu yaklaşımın örneğidir [K13]. Bu literatür bizim robot,
batch1 ve süre bütçemiz için kazanç garantisi vermez. Sonraki H1 sorusu,
aynı sayısal motor ve toplam bütçede neural seed'in fayda sağlayıp
sağlamadığıdır. Bu raporda hibrit üstünlük ölçülmedi.

## Kanıt ve kaynak dizini

Kaynak yolları depo köküne göredir. Tam SHA envanteri bu raporla birlikte
üretilen evidence-index.json'da bulunur; büyük raw/checkpointler Git dışıdır.

- K1: experiments/C1-06/stage2/RESULTS.md ve final-001/acceptance.json.
- K2: experiments/C1-06R/round1-analysis/RESULTS.md ve audit.json.
- K3: experiments/C1-06R/diagnostic2/RESULTS.md; ADR-016.
- K4: experiments/C1-06R/diagnostic3/RESULTS.md ve audit.json.
- K5: experiments/C1-06R/diagnostic4/RESULTS.md; diagnostic5/RESULTS.md ve loss-followup.
- K6: experiments/C1-06R/diagnostic6/RESULTS.md ve audit.json.
- K7: experiments/C1-06R/diagnostic7/RESULTS.md ve CRITERION_AND_DIRECTION_REVIEW.md.
- K8: experiments/C1-06R/diagnostic8/RESULTS.md ve audit.json.
- K9: experiments/C1-06R/diagnostic9/config.json,results.json,audit.json,RESULTS.md.
- K10: docs/adr/ADR-024-c106r-centered-residual.md.
- K11: docs/adr/ADR-025-core-research-stop-and-hybrid-handoff.md; experiments/C1-07/preparation/C1-06R-closure.json.
- K12: docs/raporlar/02_Core_v1_0_r1.md; experiments/C1-07/preparation/G1_READINESS.md.
- K13: Ames B, Morgan J, Konidaris G. IKFlow Generating Diverse Inverse Kinematics Solutions. arXiv:2111.08933v3,2022. https://arxiv.org/html/2111.08933v3

## Proje sonunda makaleye aktarım kontrolü

Bağımsız final ile validation'ı ayır; tüm seed'leri göster; denominatörü
koru; hipotezlerin ön kayıt ve keşifsel durumunu yaz; veri/robot/runtime
kimliklerini sabitle; yeni H1 sonuçlarını bu rapora geriye dönük ekleme,
ayrı sürümle bağla. Grafiklerin kaynak JSON/CSV'sini ve üretim betiğini
sakla. Yeni bir çözüm bulunursa eski negatif sonuçları silme; hangi
koşulun değiştiğini açık bir karşılaştırmayla göster.

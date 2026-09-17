# NeuroKinematics Foundations teknik tasarım raporu

Hedef yazılım v0.1.0 · Belge r1 · 17 Eylül 2026

## 1 Fazın amacı ve teslimatı

Foundations, sonraki makine öğrenmesi çalışmalarının dayandığı robot modelini, kinematik hesapları, veriyi ve ölçüm altyapısını doğrular. Yanlış eklem sırası veya TCP dönüşümü üzerinde eğitilen model, eğitim kaybı azalsa bile yanlış problemi öğrenir. Bu nedenle G0 kapısı kapanmadan Core eğitimi başlamaz.

Faz çıktısı; robot tanımını yükleyen bir komut satırı aracı, Pinocchio referans hesabı, küçük bağımsız seri zincir FK uygulaması, Jacobian denetimi, deterministik veri üretimi ve sayısal baseline çalıştıran benchmark iskeletidir. Bu rapor bir uygulama tasarımıdır; kabul testleri henüz çalıştırılmamıştır.

| Alan | Başlangıç kararı | Sınır |
|---|---|---|
| Robot | KUKA KR6 R900 sixx varsayımı | Tam URDF varyantı F0-01 ile sabitlenecek |
| Mekanizma | Sabit tabanlı açık seri zincir, altı döner eklem | Floating base, kapalı çevrim, mimic ve çoklu kol yok |
| Görev | Base çerçevesinde TCP konumu ve yönelimi | Yalnız konum testi ayrı görev profili gerektirir |
| Referans | Pinocchio FK ve geometrik Jacobian | Referans kütüphane gerçek robot ölçümü değildir |
| Kendi uygulamamız | URDF eklem dönüşümlerini çarpan sınırlı FK | Genel amaçlı yeni robotik kütüphane yazılmaz |
| Ortam | Linux ve CPU doğrulaması | Kesin paket sürümleri kurulumda kilitlenecek |

### Girdi ve çıktı sözleşmeleri

Girdiler URDF veya önceden çözümlenmiş Xacro çıktısı, base link, tip link, sabit tip–TCP dönüşümü, etkin eklem listesi ve model manifestidir. Eksik limit veya desteklenmeyen eklem türü açık hata üretir; gizli varsayılanla devam edilmez. Xacro çözümleme ayrı bir hazırlık adımıdır.

Çıktılar `RobotSpec`, FK dönüşümü, geometrik Jacobian, limit doğrulaması, veri manifesti ve sorgu başına benchmark kaydıdır. Modüller robot markasına göre koşul dalları içermez; robot farkları konfigürasyondan gelir. F0-01 içinde lisans ve geometri kaynağı kaydedilir; boyutlar bellekten doldurulmaz.

### Gereksinimler

REQ-F01 model kimliği ve birimlerin tutarlılığı; REQ-F02 FK doğruluğu; REQ-F03 Jacobian ve metrik doğruluğu; REQ-F04 deterministik ve sızıntı denetimli veri; REQ-F05 karşılaştırılabilir benchmark; REQ-F06 temiz kurulum ve faz devri. Gereksinim–görev–test eşleştirmesi faz roadmapinde bulunur.

<!-- pagebreak -->

## 2 Kinematik sözleşme

İç hesapta metre, radyan ve saniye kullanılır. Eklem vektörü manifestteki `joint_names` sırasındadır. Poz hedefi base çerçevesinde ifade edilir. Dönüşüm matrisleri sütun vektörlerini çarpar; T_A_B, B koordinatlarını A koordinatlarına taşır. Sağ elli çerçeve kullanılır. Bu sözleşme her dış kütüphane adaptöründe test edilir.

<!-- equation:FK -->
$$
T_{base,TCP}(q)=T_{base,0}T_{0,1}(q_1)\cdots T_{n-1,n}(q_n)T_{n,TCP}
$$

URDF `origin` dönüşümü ile eklem hareketi ayrı tutulur. Sabit eklemler kinematik zincirde kalır; eklem vektöründe yer almaz. Eksenler doğru yerel çerçevede yorumlanır. Tip link ile gerçek TCP aynı kabul edilmez; sabit takım dönüşümü her FK ve Jacobian hesabına dahil edilir.

Dosya/API sınırında quaternion sırası **w, x, y, z** olarak seçilir; ters sıra kullanan kütüphanelerde açık dönüşüm yapılır. Normalize edilmemiş girişler tolerans içinde normalize edilir; sıfır norm ve sonlu olmayan değerler reddedilir. r ve −r aynı yönelimi temsil eder. Açı hatası hesaplanırken mutlak iç çarpım veya göreli dönme matrisi kullanılır; işaret değişimi hareket olarak yorumlanmaz.

### FK referansı ve bağımsızlık

Pinocchio'nun robot modeli ve kinematik hesapları referans olarak kullanılır [K01]. Bağımsız FK uygulaması, aynı kütüphanenin çıktısını tekrar paketlemek yerine seri dönüşümleri kendisi hesaplar. Böylece algoritma/adaptör hataları yakalanır. Ancak iki uygulama aynı yanlış URDF'yi kullanıyorsa ikisi de aynı yanlış fiziksel modeli üretebilir. Model geometrisi ayrıca üretici veya güvenilir model kaynağıyla denetlenir.

DH parametreleri öğretici bir küçük zincirde kullanılabilir. Ana URDF için otomatik ve kayıpsız DH dönüşümü zorunluluğu konmaz. DH tablosu eklenirse standart/modifiye seçimi ve eksen atamaları ayrıca açıklanır.

### Jacobian sözleşmesi

Jacobian satır sırası çizgisel hız ardından açısal hızdır. Her iki hız TCP noktasında ve base eksenlerinde ifade edilir. Pinocchio adaptöründe karşılık gelen çerçeve seçimi açıkça kaydedilir; `WORLD`, yerel ve dünya yönelimli yerel ifadeler birbirinin yerine geçirilmez. Açısal satırlar Euler açılarının türevi değildir.

Önce basit geometrilerde işaret ve yön kontrol edilir. Sonra her eklem küçük miktarda değiştirilip sonlu fark ile Jacobian sütunu doğrulanır. Rotasyon türevi, R artışı ve SO(3) logaritmasıyla aynı çerçevede hesaplanır; matris elemanları doğrudan açısal hız sayılmaz.

<!-- pagebreak -->

## 3 Matematik ve sayısal doğrulama

Poz hatası ve yönelim hatası birbirinden ayrı ölçülür. Aşağıdaki yönelim metriğinde `clip`, kayan nokta yuvarlanması nedeniyle arccos tanım kümesinin dışına çıkmayı önler. Sonuç tablolarında konum milimetre, yönelim derece olarak sunulur.

<!-- equation:POSE -->
$$
e_p=\|p(\hat q)-p_d\|_2,\quad e_R=\arccos\left(\operatorname{clip}\left(\frac{\operatorname{tr}(R_d^TR(\hat q))-1}{2},-1,1\right)\right)
$$

FK regresyonunda çok küçük yönelim farkları için arccos yerine dönme matrisleri farkının Frobenius normu kullanılır. Böylece sıfır yakınındaki sayısal hassasiyet, yanlış model farkı olarak görünmez. Tam sıfır, 90 derece, 180 derece ve quaternion işaret çifti için ayrı örnekler vardır.

| Test | Veri ve yöntem | G0 kabul hedefi |
|---|---|---|
| T-F01 Model | Eklem sırası, limit, eksen, base ve TCP denetimi | Manifest tam; sessiz dönüşüm yok |
| T-F02 FK | Önce 1.000, kapanışta 10.000 konfigürasyon; float64 | En büyük konum farkı ≤ 1e−9 m; matris farkı ≤ 1e−9 |
| T-F03 Jacobian | En az 100 konfigürasyon; merkezi fark | Normalize mutlak fark ≤ 1e−5; max ve medyan kayıtlı |
| T-F04 Metrikler | Bilinen dönüşümler, işaret ve birim örnekleri | Hatalar beklenen analitik değerle uyumlu |

Normalize Jacobian farkı, aynı ℓ ile ölçeklenmiş iki matrisin farkının Frobenius normunun, 1 ile referans matris normundan büyük olanına bölünmesidir. Bu sayılar sayısal doğrulama eşikleridir; robotun fiziksel hassasiyeti değildir. İlk perturbasyon 1e−6 rad seçilir; 1e−5 ve 1e−7 rad ile duyarlılık kontrol edilir. Başka model veya hesap tipi farklı eşik gerektirirse eğitimden önce protokol güncellenir; başarısız deneyi geçirmek için eşik yükseltilmez.

### Tekillik ve ölçek

Çizgisel ve açısal Jacobian satırlarının birimleri farklıdır. Karşılaştırmada karakteristik uzunluk ℓ kullanılır. ℓ, robot manifestine yazılan sabit geometrik ölçektir; test verisinden öğrenilmez.

<!-- equation:JAC -->
$$
\widetilde J=\operatorname{diag}(1/\ell,1/\ell,1/\ell,1,1,1)J,\quad \kappa(\widetilde J)=\sigma_{max}/\sigma_{min}
$$

En küçük tekil değer, koşul sayısı ve tekil değerlerin çarpımı birlikte raporlanır. Sıfır tekil değerde koşul sayısı sonsuz olarak işaretlenir. Altı boyutlu görevde DoF altıdan küçükse 6×n matris için determinant tabanlı tam görev manipulability sıfır olur; böyle bir robot için görev boyutu yeniden tanımlanmalıdır. Yedi DoF olması tek başına bu metriği geçersiz kılmaz.

<!-- pagebreak -->

## 4 Veri fabrikası ve örnekleme

Veri modeli `robot_id`, `model_hash`, `sample_id`, `group_id`, `split`, q, hedef konum, quaternion, örnekleme sınıfı ve referans metrikleri içerir. Core için `q_current` ve `pair_mode`, Hybrid için `trajectory_id`, zaman damgası ve önceki durum alanları eklenebilir. Etiketlerin robot ve TCP kimliği kaybolmaz.

Veri büyüklüğü aşamalıdır: önce 10.000 örnek üzerinde boru hattı ve disk düzeni doğrulanır; sonra 100.000 örnekle ilk öğrenme deneyi yapılır. Bir milyon örneğe ancak küçük veri deneyinin hata analizi gerekçe sağlıyorsa geçilir. Shard biçimi ve içerik hashleri manifestte tutulur; dosya metadata değişimi ile sayısal içerik farkı ayrılır.

### Bölümlendirme

Eğitim/doğrulama/test oranı başlangıçta yüzde 70/15/15 olarak seçilir. Grup kimliği split öncesinde atanır. Aynı kök konfigürasyondan üretilen yakın varyantlar, aynı yörünge ve aynı çift ailesi tek split içinde kalır. Normalizasyon yalnız eğitim verisinden öğrenilir. FK ile oluşturulan sabit bağımsız benchmark hedefleri eğitim kaynağı değildir.

Tam tekrarlar hash karşılaştırmasıyla aranır. Yakın örnekler, normalize eklem uzaklığı ve poz uzaklığı dağılımlarıyla incelenir. Birbirine yakın tüm noktaları kaldırmak gerçekçi bir zorunluluk değildir; amaç bağımlı örnek ailesinin sızmasını önlemek ve testin eğitim komşuluğunu görünür kılmaktır. Mekânsal ayrılmış test, rastgele splitten ayrı raporlanır.

### Kapsama ve zor alt kümeler

Eklem uzayında uniform örnekleme, çalışma uzayında uniform yoğunluk üretmez. LHS veya katmanlı örnekleme seçenekleri karşılaştırılır. Çalışma uzayı kapsaması, referans örnek havuzunun dolu voxel sayısı üzerinden ampirik ölçülür. Voxel boyutu, yönelim bölmeleri, örnek sayısı ve doygunluk eğrisi birlikte kaydedilir. Evrensel “yüzde 95 erişilebilir uzay kapsandı” iddiası kullanılmaz.

Ana test, en az 10.000 FK ile üretilmiş ve eklem limitleri içindeki hedefi kapsar. Ayrı zor testler en az 1.000 sınır yakınlığı örneği ve 1.000 tekillik yakınlığı örneği içerir. Örnekler örtüşüyorsa bu açıkça kaydedilir ve toplam başarı oranında iki kez sayılmaz.

Sınır etiketi için birinci tanım eklem limitine normalize uzaklığın yüzde 2'den küçük olmasıdır. Bu etiket gerçek Kartezyen çalışma alanı sınırıyla özdeş değildir. Kartezyen sınır vekili, yalnız eğitim referans havuzundan elde edilen seyrek/dış voxel bölgeleriyle ayrıca tanımlanır. Tekillik alt kümesi, eğitim referans havuzundaki ölçeklenmiş en küçük tekil değerlerin alt yüzde 5 eşiğiyle dondurulur.

Erişilemez hedef, bir çözücü başarısız oldu diye etiketlenmez. Analitik dış sınırla kanıtlanan örnekler `PROVEN_UNREACHABLE`; diğer çözülemeyen hedefler `UNRESOLVED` olur. Çarpışma katmanı henüz yoksa veri kinematik olarak ulaşılabilir sayılır; çarpışmasız olduğu söylenmez.

<!-- pagebreak -->

## 5 Sayısal baseline ve benchmark sözleşmesi

İlk zorunlu çözücü DLS'dir. LM, sönümleme ve adım kabulünü uyarlayan ayrı bir varyant olarak eklenecekse bu fark açıkça yazılır. İki farklı isim altında aynı algoritmanın sonuçları bağımsız başarı gibi sunulmaz. Sayısal IK'nin başlangıca bağımlılığı ve yakınsamama olasılığı korunur [K06].

<!-- equation:DLS -->
$$
\Delta q=\widetilde J^T(\widetilde J\widetilde J^T+\lambda^2I)^{-1}\widetilde e
$$

Bu ifade yerel lineer düzeltme adımını gösterir. Uygulamada açık matris tersi hesaplamak yerine lineer sistem çözülür. Artık vektörüyle Jacobian'ın çerçevesi ve işareti eşleşmelidir. Büyük yönelim hatalarında SO(3)/SE(3) logaritmasının türevi ve adım kabulü dikkate alınır; formül bütün hata büyüklüklerinde küresel çözüm garantisi değildir.

| Sözleşme alanı | Tasarım değeri | Kayıt kuralı |
|---|---|---|
| Görev toleransı | 2 mm ve 1 derece birlikte | İki koşul da sağlanır |
| Zaman bütçesi | 10 ms ve 50 ms ayrı profiller | Zaman aşımı ve gerçek geçen süre saklanır |
| İterasyon üst sınırı | DLS için 200 | Diğer çözücülerde mevcutsa ayrıca kayıt |
| Başlangıç | Aynı sorgudaki q_current | Rastgele tekrar seedleri sabit ve kayıtlı |
| Doğrulayıcı | Ortak Pinocchio tabanlı denetim | Solver başarı bayrağına kör güven yok |
| Hesap tipi | Float64 referans baseline | Diğer tipler ayrı varyant |

KDL, TRAC-IK ve pick_ik Core içinde eklenecektir [K03–K05]. Çözücülerin zaman aşımı uygulamaları ve yardımcı hedefleri farklı olabileceğinden uyarlama notları tutulur. Aynı thread zorunluluğuyla mimarileri değiştirmek yerine eşit kaynak tavanı altında gerçek thread sayısı raporlanır. Ayrıca tek sorgu CPU ve toplu GPU ölçümleri karıştırılmaz.

### Ölçüm kaydı

Her sorguda `query_id`, çözücü/sürüm, q_current, q_candidate, poz hataları, limit sonucu, durum kodu, toplam süre, varsa iterasyon sayısı ve kullanılan başlangıç tutulur. İterasyon bilgisi alınamıyorsa sıfır yazılmaz; `NOT_AVAILABLE` yazılır.

Geçerli bir çözümün timeout sonrasında dönmesi, geometri başarısı ile deadline başarısının ayrı raporlanmasını gerektirir. Bütün sorguların gecikme dağılımı ve başarılı sorguların gecikme dağılımı birlikte verilir. Eğitim, dosya yükleme ve ısınma süreleri ayrı kayıttır; darboğazı saklamak amacıyla toplam çözümden rastgele kalem çıkarılmaz.

<!-- pagebreak -->

## 6 Görev sırası ve doğrulama kapısı

| Görev | Çıktı | Bağımlılık |
|---|---|---|
| F0-00 Kapsam ve ortam | SPEC, ortam kararı, temel komutlar | Yok |
| F0-01 Robot modeli | Robot manifesti ve model hashleri | F0-00 |
| F0-02 FK doğrulama | Referans ve bağımsız FK karşılaştırması | F0-01 |
| F0-03 Jacobian | Sonlu fark ve metrik testleri | F0-02 |
| F0-04 Veri fabrikası | Shardlar, split ve kapsama raporu | F0-03 |
| F0-05 Benchmark | DLS ve sorgu başına sonuç kaydı | F0-04 |
| F0-06 Faz devri | G0 değerlendirmesi ve tekrar üretim tarifi | F0-05 |

F0-04 kapanışında T-F05 deterministik içerik testi, T-F06 grup/split denetimi ve T-F07 veri doğruluğu raporu tamamlanır. Eğitim ve test grup kesişimi sıfır olmalıdır. Test eşiği veya örnek seçimi, ilerideki model sonucuna göre değiştirilmez. Coverage sonucu düşükse veri tasarımı açıkça revize edilir; yüzdelik bir göstergeyi şişirmek için grid kaba hale getirilmez.

F0-05 kapanışında T-F08 baseline protokol testi tamamlanır. Bilinen kolay hedeflerde çözücü toleransa ulaşmalı; erişilemez hedef ve zaman bütçesi örneklerinde yanlış başarı vermemelidir. Kolay hedef testi bir performans üstünlüğü testi değildir. F0-06 içindeki T-F09, temiz ortamda küçük veri ve benchmarkın tekrar üretimini kontrol eder.

### G0 kabulü

Model kimliği ve koordinat sözleşmesi tam; FK/Jacobian testleri başarılı; veri manifesti ve bölümlendirme denetimi mevcut; DLS baseline sonuçları yeniden üretilebilir; tüm kritik hata örnekleri çözülmüş olmalıdır. Yazılım görevleri için bu rapordaki sayısal hedeflerin sonucu şu anda **ÖLÇÜLMEDİ** durumundadır.

### Risk ve geri dönüş

FK farkı varsa önce base/TCP, joint sırası, origin çarpım sırası, eksen yönü ve birimler incelenir. Pinocchio farkı varken yapay zekâ eğitimiyle hatayı örtmek denenmez. Jacobian farkı varsa çerçeve ve perturbasyon büyüklüğü ayrıştırılır. Büyük veri üretimi yavaşsa aynı protokol küçük shardlarla korunur. Model kaynağı doğrulanamıyorsa F0-01 engelli kalır veya yeni robot seçimi ADR ile yapılır.

### Core fazına devir

Doğrulanmış RobotSpec, FK/Jacobian servisleri, sabit eğitim/doğrulama/test manifestleri, DLS baseline konfigürasyonu, örnek ham sonuç dosyası ve G0 kararı devredilir. Core, hedef formatını veya eklem sırasını kendi içinde yeniden tanımlamaz. Diferansiyellenebilir FK, bu referansa karşı ayrıca test edilir.

<!-- pagebreak -->

## 7 Uygulama rehberi ve öğrenme hedefi

Fazın eğitim değeri, hazır kütüphaneyi kullanırken onun koordinat ve sayısal davranışını denetleyebilmektir. Önce iki eklemli basit bir zincirde dönüşüm çarpımı elle kontrol edilir; sonra gerçek robot zinciriyle aynı arayüz çalıştırılır. Bu küçük örnek portföy anlatımını güçlendirir ancak ana test setinin yerine geçmez.

İlk komutlar model inceleme, FK hesaplama, kinematik doğrulama, veri üretme ve benchmark çalıştırma işlevlerini hedeflemelidir. Komut adları uygulama sırasında kesinleşir. Raporda henüz var olmayan bir CLI'nin çalıştığı iddia edilmez. Her komutun girdi dosyası, beklenen çıktı ve hata kodu F0 görev dosyasına yazılacaktır.

Kodun `kinematics`, `data`, `solvers` ve `benchmark` modülleri arasında tek yönlü bağımlılık kurulması tercih edilir. Veri modülü FK servisini çağırabilir; FK modülü eğitim veya GUI modülünü import etmez. Robot dosyaları ile üretilen veri aynı klasöre karışmaz. Büyük datasetler Git geçmişine rastgele eklenmez; içerik kimliği ve üretim tarifi sürümlenir.

Faz sonunda gösterilebilir çıktı, rastgele bir hedef seçen animasyondan daha kapsamlıdır: modelin kimliği, 10.000 konfigürasyondaki FK hata dağılımı, Jacobian denetimi, veri kapsama görseli ve baseline başarı/başarısızlık dökümü birlikte sunulur. Bu görseller gerçek ölçümlerden üretilecektir.

## 8 Kaynaklar

K01 [Pinocchio resmi kinematik ve model altyapısı](https://github.com/stack-of-tasks/pinocchio).

K03 [MoveIt kinematik konfigürasyonu ve KDL](https://moveit.picknik.ai/main/doc/examples/kinematics_configuration/kinematics_configuration_tutorial.html).

K04 [TRAC-IK resmi MoveIt açıklaması](https://moveit.picknik.ai/main/doc/how_to_guides/trac_ik/trac_ik_tutorial.html).

K05 [pick_ik resmi MoveIt açıklaması](https://moveit.picknik.ai/main/doc/how_to_guides/pick_ik/pick_ik_tutorial.html).

K06 [Lynch ve Park Modern Robotics Bölüm 6](https://modernrobotics.northwestern.edu/chapters/chapter6/), analitik ve sayısal ters kinematik ders kaynağı.

K13 [Zhou ve diğerleri On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035), CVPR 2019. Quaternion ve sürekli 6D gösterimler Core içinde deneysel olarak karşılaştırılır.

Kaynak erişimi 17 Eylül 2026. Proje eşikleri literatür sonucu veya endüstri standardı olarak sunulmamıştır. Güncel uygulama ilerlemesi `docs/records/STATUS.md`, görev ayrıntıları `docs/roadmaps/F0_Foundations.md` üzerinden izlenir.

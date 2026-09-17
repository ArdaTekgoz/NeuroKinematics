# NeuroKinematics Core teknik tasarım raporu

Hedef yazılım v1.0.0 · Belge r1 · 17 Eylül 2026

## 1 Amaç ve araştırma sınırı

Core, öğrenilmiş IK modelinin katkısını ölçülebilir biçimde sınar. Aynı robot, veri ve değerlendirme sözleşmesi altında pose-only MLP, durumla şartlandırılmış model, residual mimari ve kinematik kayıp bileşenleri karşılaştırılır. Beklenen çıktı yalnız model ağırlığı değildir; hangi bileşenin hangi durumda yararlı olduğunu gösteren tekrar üretilebilir deney paketidir.

Bu faz G0 kapısının kapanmasını gerektirir. Yeni yazılım, eğitilmiş model ve sonuçlar henüz doğrulanmamıştır. v1.0 ifadesi hedef sürümdür; bu rapor uygulama öncesi tasarımı tanımlar.

| İçerik | Faz kararı | Gerekçe |
|---|---|---|
| Tek robot ve sabit görev | Zorunlu | Nedensel karşılaştırmayı sadeleştirir |
| Pose-only ve conditioned MLP | Zorunlu | Durum bilgisinin katkısını ayırır |
| Differentiable FK ve limitler | Zorunlu | Model tutarlılığını ve çıktı geçerliliğini sınar |
| Res-MLP, delta çıktı, 6D temsil | Kontrollü ablasyon | Mimari tercihi sonuçtan önce üstün saymaz |
| Tekillik cezası | İkinci deney dalgası | Önce sağlam FK ve limit eğitimi gerekir |
| Jerk, GUI, gerçek robot, MAML | Bu fazda yok | İlgili sonraki fazlarda ayrı veri ve test ister |

### Araştırma sorusu

H2, FK ve eklem limiti bileşenlerinin aynı kapasitedeki supervised conditioned modele göre zor alt kümelerde geçerli çözüm oranını artırıp artırmadığını sorar. State conditioning ise çoklu çözüm belirsizliğini azaltmayı hedefler; matematiksel birebirlik veya kesintisiz hareket garantisi sunmaz.

REQ-C01 adil baseline seti; REQ-C02 doğru diferansiyellenebilir FK; REQ-C03 izlenebilir model ve kayıp; REQ-C04 kontrollü ablasyon; REQ-C05 bağımsız değerlendirme; REQ-C06 model kartı ve faz devri olarak tanımlanır.

### Bilimsel konum

Bensadoun ve diğerlerinin çalışması çoklu çözüm dağılımlarını zaten ele alır [K07]. CycleIK, farklı robot tasarımlarına taşınan bir neural IK yaklaşımıdır [K09]. Bu faz yeni bir genel IK teorisi iddia etmez. Katkı adayı; tanımlı iş yükünde kontrollü deney, hata analizi ve robot başına tekrar üretilebilir eğitim hattıdır.

<!-- pagebreak -->

## 2 Veri eşleşmeleri ve çoklu çözüm

Conditioned model için yalnız (hedef poz, etiket eklem) çifti yeterli değildir; sorguda gerçekten mevcut olan q_current gerekir. Eğitim dosyası, q_current değerinin nasıl üretildiğini kaydeder. Etiket q_target değerini girdiye aynen vermek veya yalnız çok küçük gürültü ekleyerek iyi sonuç almak, genel IK başarısı olarak yorumlanmaz.

### İki açık görev dağılımı

**Yerel hareket dağılımı:** Limit içindeki q_target çevresinde, her eklem için en çok 0.1 rad perturbasyonla q_current üretilir; limit dışında kalanlar yeniden örneklenir. Aynı kök q_target ve varyantları tek gruptur. Bu, yakın hedefler için öğrenmeyi kolaylaştıran kontrollü bir sentetik görevdir; gerçek zaman serisi değildir.

**Geniş başlangıç dağılımı:** q_current ve ulaşılabilir hedef birbirinden bağımsız seçilir. Aynı hedefin birden fazla geçerli dalı olabilir. Etiket seçimi gerekiyorsa sabit bütçeli sayısal öğretmen adayları içinde q_current'a normalize uzaklığı en düşük geçerli çözüm alınır. Öğretmen başarısızlıkları ve seçim yanlılığı kaydedilir; öğretmenin bulamadığı test sorguları benchmarktan çıkarılmaz.

Başlangıç eğitim karışımı yüzde 50 yerel, yüzde 50 geniş başlangıç çiftidir. Öğretmen maliyeti ilk dalgada sınırlı alt kümede ölçülür. Yalnız yerel model eğitilirse model kartı bunu açıkça belirtir ve geniş başlangıç sonucu ayrı raporlanır. Belirli etikete eklem hatası, birden fazla doğru IK çözümü varken temel doğruluk ölçütü olamaz; bağımsız FK geçerliliği önceliklidir.

Ana benchmarkın 10.000 sorgusu aynı iki dağılımı eşit ağırlıkla taşır; yerel ve geniş başlangıç sonuçları ayrıca verilir. Sınır/tekillik alt kümelerinde de pair_mode korunur. Split, çiftler ve perturbasyonlar oluşturulmadan önce kök gruplara uygulanır. Eğitim ve test arasındaki aynı yörünge, aynı kaynak konfigürasyon veya öğretmen aday ailesi sızıntısı engellenir.

### Öğretmen etkisini ayırma

Sayısal öğretmen kullanılmış bir model, öğretmenin çözüm tercihlerini taklit edebilir. Öğretmenin kendi performansı ayrı baseline olarak verilir; etiket bütçesi ve üretemediği örnek oranı raporlanır. FK kaybının avantajı, öğretmenin hatalarını saklamayacak şekilde değerlendirilir. Test hedefleri model tarafından veya öğretmenin başarılı sonuçları arasından sonradan seçilmez.

### Ölçek ve temsil

Sınırlı döner eklemler kendi alt/üst sınırlarıyla normalize edilir. Büyük açısal aralıklarda yalnız sin/cos kullanmak tur bilgisini silebilir; bu nedenle başlangıç gösterimi limit tabanlı normalizasyondur. Dairesel uzaklık yalnız gerçek sürekli eklemler için uygundur; v1 kapsamındaki sınırlı eklemlere körlemesine uygulanmaz.

<!-- pagebreak -->

## 3 Model ve diferansiyellenebilir kinematik

İlk MLP, üç adet 256 genişlikli gizli katman ve SiLU aktivasyonu ile tanımlanır. Bu bir başlangıç konfigürasyonudur. AdamW, başlangıç öğrenme oranı 1e−3, üst sınır 200 epoch ve doğrulama ölçütünde 20 epoch ilerleme olmaması halinde erken durdurma kullanılabilir. Etkin batch 1024 hedeflenir; bellek yetmezse küçük mikro-batch ve gradient accumulation kullanılır. Kesin ayarlar deneyden önce config dosyasına yazılır.

Altı eklemli robotta quaternion kullanan pose-only girdi 7, conditioned girdi 13 bileşendir. Sürekli 6D rotasyonla bu sayılar 9 ve 15 olur. Çıkış altı eklem değeridir. Res-MLP ayrı varyanttır; katman sayısının artışı ile FK kaybının etkisi aynı deneyde değiştirilmez. Mimari karşılaştırmada parametre sayısı ve eğitim bütçesi raporlanır.

<!-- equation:DELTA -->
$$
\Delta q=f_\theta(x,q_{current}),\qquad \hat q=q_{current}+\Delta q
$$

Mutlak q çıktısı ile delta çıktı eşit veri ve eğitim bütçesinde karşılaştırılır. Delta hedefi çözüm dalı seçimi sorununu tek başına çözmez. MimicIK mevcut eklem, hedef poz, delta hareket ve FK tutarlılığını kullandığından bu kombinasyon özgünlük iddiası olarak sunulmaz [K11].

### Differentiable FK uygulaması

PyTorch içinde autograd akışı kesilmeden çalışan bir FK gerekir. `pytorch_kinematics` hazır adaydır [K02]; desteklenen zincir için küçük Torch FK alternatifi de mümkündür. Pinocchio'nun Python üzerinden NumPy hesabını çağırmak, kendiliğinden PyTorch gradyanı oluşturmaz.

T-C01, aynı q üzerinde Torch ve Pinocchio FK sonuçlarını karşılaştırır. Float64 için Foundations FK eşikleri uygulanır. Float32 deneyinde başlangıç eşikleri 1e−5 m konum ve 1e−5 dönme matrisi normudur. Float32 farkı ayrı raporlanır; referansın doğruluğu buna göre gevşetilmez.

T-C02, float64 ile en az 20 limit içi konfigürasyonda autograd ve merkezi fark gradyanını denetler. Sonlu fark epsilon 1e−6, mutlak tolerans 1e−5 ve göreli tolerans 1e−3 başlangıç hedefidir. Küçük bir joint perturbasyonunun eğitim kaybını beklenen yönde değiştirdiği de kontrol edilir. Türevlenemeyen sınır noktaları ayrı edge-case testidir.

### Rotasyon deneyi

Quaternion ve sürekli 6D giriş karşılaştırması, Zhou ve diğerlerinin gösterim tartışmasına dayanır [K13]. Quaternionda işaret tutarlılığı ve 180 derece çevresi incelenir. 6D gösterimin avantajı bu robottaki deneyle ölçülür; kaynak çalışmanın başka görevlerdeki sonucu doğrudan taşınmaz. API'nin quaternion biçimi değişmez; 6D dönüşüm model ön işlemidir.

<!-- pagebreak -->

## 4 Kayıp fonksiyonları ve çıktı geçerliliği

Core kaybı boyutsal ölçekleri açık olan eklem, konum, yönelim ve limit bileşenlerinden oluşur. Eğitim kaybı, nihai başarı oranıyla aynı ölçü değildir. Kinematik doğrulayıcı bütün model varyantlarında aynıdır.

<!-- equation:LOSS -->
$$
L=\lambda_qL_q+\lambda_pL_p+\lambda_RL_R+\lambda_{lim}L_{lim}
$$

<!-- equation:LOSSTERMS -->
$$
L_p=\|(p(\hat q)-p_d)/\ell\|_2^2,\qquad L_R=\|R(\hat q)-R_d\|_F^2/8
$$

Konum ℓ karakteristik uzunluğuyla ölçeklenir. Yönelim için dönme matrisi chordal kaybı kullanılır; bu derece veya radyan cinsinden açısal hata değildir. Nihai raporda ayrıca geodezik yönelim hatası hesaplanır. Eklem kaybı limit aralığıyla normalize edilen etiket farkıdır. Etiket farklı bir geçerli dalı temsil edebileceğinden etikete yakınlık ve hedef poza ulaşma ayrı incelenir.

Limit kaybı, alt limit altına veya üst limit üstüne taşmanın normalize karesel ReLU cezasıdır. Bu yumuşak ceza, sıfır ihlali garanti etmez. Ayrı varyantta tanh tabanlı sınırlı çıktı denenebilir. Sınırlı çıktı zaten limit cezasını büyük ölçüde sıfırlayacağı için tanh ile limit kaybının katkısı birbirine karıştırılmaz. Sınır yakınında tanh gradyan doygunluğu ayrıca izlenir.

### Kayıp ağırlığı seçimi

İlk ölçeklenmiş pilotta bileşen kayıpları ve gradyan normları kaydedilir. En fazla sekiz validation konfigürasyonundan oluşan arama bütçesi ayrılır. Aynı arama bütçesi kontrol modeline de verilir. Eski rapordaki FK ağırlığı adayları 0, 0.1, 0.5, 1, 2 ve 5, normalize kayıplar altında ayrı config olarak sınanabilir; eski ham kayıp ölçeğindeki sonuçlarla birleştirilmez.

Curriculum varsayılan zorunluluk değildir. Sabit ağırlıkla eğitim sağlıklıysa, yalnız validation üzerinde ayrı bir curriculum varyantı denenebilir. Epoch aralıkları ve ağırlık değişimleri config içinde açıkça yazılır. Test sonucuna bakıp kayıp ağırlığı seçmek yasaktır.

### Tekillik cezası

Önce tekillik yalnız değerlendirme metriğidir. Sonraki kontrollü varyantta ölçeklenmiş en küçük tekil değere hinge cezası denenebilir. Tekil değerlerin çakıştığı yerlerde gradyan kararsızlığı ve hedef doğruluğu kaybı incelenir. Eski ters manipulability cezası ölçek ve sıfır yakınında büyüme riski nedeniyle doğrudan varsayılan yapılmaz.

Tekillik cezası hedefte tekil olmayan başka bir çözüm bulunacağını garanti etmez. Yeterli serbestlik yoksa hedef doğruluğuyla çatışabilir. Core içinde jerk kaybı kullanılmaz. q_current'a yakınlık terimi varsa adı konfigürasyon yakınlığıdır; hız, ivme veya jerk sonucu olarak yorumlanmaz.

<!-- pagebreak -->

## 5 Deney matrisi ve baseline adaleti

| Deney | Değişen unsur | Sabit tutulan temel |
|---|---|---|
| E-C01 | Pose-only MLP → conditioned MLP | Veri, eğitim ve çıktı tipi |
| E-C02 | Conditioned MLP → Res-MLP | Kayıp; kapasite ve bütçe kayıtlı |
| E-C03 | Supervised → FK bileşenleri | Conditioned backbone ve etiketler |
| E-C04 | Limit cezası ekleme | Sınırsız çıktı ve diğer kayıplar |
| E-C05 | Sınırlı çıktı başlığı | Aynı backbone; limit cezası etkisi ayrı |
| E-C06 | Quaternion → 6D giriş | Model ailesi ve veri çiftleri |
| E-C07 | Mutlak q → delta q | Girdi, kayıp ve veri |
| E-C08 | Tekillik cezası | Doğrulanmış en sade Core modeli |

Deneyler aşamalıdır; bütün kombinasyonların Kartezyen çarpımı denenmez. İlk dalga E-C01, E-C03 ve E-C04'tür. İkinci dalga kapasite ve temsil tercihleridir. Her ana karşılaştırma en az üç bağımsız eğitim seed'iyle çalışır. Veri seed'i ile eğitim seed'i ayrı saklanır.

DLS, KDL, TRAC-IK ve pick_ik aynı sorgu setinde değerlendirilir [K03–K05]. pick_ik local ve global ayarları farklı varyantlar olarak etiketlenir. Parametre ayarı yalnız validation kümesinde yapılır. KDL veya başka bir solverın kendi iç başarı eşiği farklıysa bağımsız denetim ortak 2 mm / 1 derece eşiğini uygular ve bu fark rapora yazılır.

### Entegrasyon sınırı

Harici baseline entegrasyonuna ilk turda en fazla 20 saat ayrılır. Kurulum engeli sürerse sorun, ortam ve başarısız komutlarla kaydedilir. DLS üzerinde Core araştırması devam edebilir; fakat eksik baseline tamamlanmadan tam robotik ekosistem karşılaştırması veya üstünlük iddiası yayımlanmaz. Bu sapma `v1.0-rc` düzeyinde açık tutulur; nominal G1 kapanışı harici baseline kayıtlarını gerektirir.

### Yanlış karşılaştırmalardan kaçınma

Neural forward pass ile bütün sayısal çözüm döngüsü tek zaman tablosunda aynı isimle verilmez. Ana ölçü, giriş hazırlığından son doğrulama ve sonuca kadar toplam süredir. Model yükleme/ısınma ayrı verilir. Tek sorgu CPU gecikmesi ve toplu GPU throughput'u ayrı deneylerdir. Başarı oranı düşük bir modelin yalnız başarılı birkaç örneği üzerinden hız üstünlüğü kurulmaz.

H2 için ön kayıt: zor alt kümelerin eşit ağırlıklı birleşiminde en az 2 yüzde puanı geçerli çözüm artışı hedeflenir; genel kümede 1 yüzde puanından fazla düşüş istenmez. Etki farkının güven aralığı ve her alt küme ayrı sunulur. Bu değerler bilimsel sonuç değil, değerlendirme öncesi proje tercihidir.

<!-- pagebreak -->

## 6 Değerlendirme ve karar

Temel kinematik başarı, sonlu çıktı, eklem limitleri ve hem 2 mm konum hem 1 derece yönelim toleransının sağlanmasıdır. Core çıktısında çarpışma durumu `NOT_CHECKED` olur; collision-free başarı oranı verilmez. Zaman bütçesinde başarı ayrıca ölçülür. Geçersiz çıktı oranı, timeout, çözülememe ve limit ihlali ayrı kodlanır.

Sonuçlar genel, yerel başlangıç, geniş başlangıç, sınır ve tekillik kümelerinde raporlanır. Her biri için N, başarı oranı, konum/yönelim medyan-P95-P99, limit ihlali, toplam süre P50-P95-P99 ve hata kategorileri bulunur. Başarısız örnekler dahil bütün sorguların sonuçları saklanır.

### İstatistik ve testin korunması

Sorgular yöntemler arasında eşleştirilir. Aynı örnek ailesi veya yörünge içindeki kayıtlar bağımsız tekrar sayılmaz; bootstrap grup düzeyinde yapılır. Eğitim seed'i değişkenliği ayrıca gösterilir. Üç seed keşif niteliğindedir; dar bir güven aralığı tek başına geniş genelleme kanıtı sayılmaz.

Ana hipotez ve karşılaştırma önce sabitlenir. Çok sayıda ablation sonucunda yalnız en iyi p-değeri seçilmez; ikincil analizler keşifsel işaretlenir. Nihai test kümesi model seçimi bittikten sonra açılır. Test hatasına göre veri toplamak gerekirse yeni eğitim/validation sürümü ve yeni bağımsız nihai test hazırlanır; eski sonuç saklanır.

| Test | Beklenen kanıt | Kapanış ölçütü |
|---|---|---|
| T-C01–02 | FK eşliği ve gradyan raporu | Referans ve türev testleri geçer |
| T-C03 | Küçük veri üzerinde öğrenme kontrolü | Loss azalır; girdi/etiket karışıklığı yok |
| T-C04 | Config, seed, split, ham sonuç kayıtları | Ana varyantlar aynı protokolde koşar |
| T-C05 | Ablasyon ve hata analizi | Olumlu ve olumsuz sonuçlar birlikte raporlu |
| T-C06 | Model kartı ve yeniden çalıştırma | Temiz ortamda örnek çıkarım ve değerlendirme |

### G1 kararları

**Araştırma kapanışı:** Bütün kritik doğruluk ve tekrar üretim testleri geçer; baselinelar çalışır; hipotez sonucu desteklendi, reddedildi veya belirsiz olarak yazılır. Yüzde 95 kinematik başarı başlangıç ürün hedefidir; bütün deneyleri kapatmanın bilimsel şartı değildir.

**Hybrid adaylığı:** Model iyi doğrudan IK üretmese de geçerli bir sayısal başlangıç sağlayabilir. Model H1 deneyi için taşınır. Açık yazılım kusuru veya geçersiz FK gradyanı varsa model taşınmaz. Core, başarısızlık analizini saklayan geçerli bir araştırma çıktısı olarak da yayımlanabilir.

<!-- pagebreak -->

## 7 Roadmap ve faz devri

| Görev | Çıktı | Ön koşul |
|---|---|---|
| C1-01 Baseline entegrasyonu | DLS, KDL, TRAC-IK, pick_ik sonuçları | G0 |
| C1-02 Çift veri protokolü | q_current eşleşmeleri ve split denetimi | G0 |
| C1-03 Torch FK | Eşlik ve gradyan testleri | G0 |
| C1-04 Neural baseline | MLP ve conditioned model | C1-02, C1-03 |
| C1-05 Physics-aware deneyler | FK/limit ve seçili varyantlar | C1-04 |
| C1-06 Ablasyon ve nihai test | Eşleştirilmiş sonuçlar, hata analizi | C1-01, C1-05 |
| C1-07 Model kartı ve G1 | Sabit checkpoint ve faz devri | C1-06 |

İşler tek kişinin kapasitesinde sırayla yürütülür; tabloda bağımsız görünen görevler zorunlu eşzamanlı çalışma anlamına gelmez. Toplam etkin emek tahmini 100–160 saattir. Eğitim makinesi beklemeleri ayrıca kaydedilir.

Hybrid'e devredilen model kartı; robot hash'i, model mimarisi, pose gösterimi, joint sırası, normalizasyon, eğitim dağılımı, checkpoint hash'i, seed, desteklenen girişler, başarısızlık kümeleri ve çıkarım örneğini içerir. Sadece `.pt` dosyası teslimat değildir.

### Riskler ve alternatifler

Loss azalırken FK hatası yüksekse etiket-poz eşleşmesi, gradyan kopması ve loss ölçeği incelenir. Geniş başlangıçta dal ortalaması oluşuyorsa yerel görev başarısı ayrı sunulur; problem çözüldü denmez. Eklem sınırı ihlali varsa çıktı başlığı deneyi yapılır; son kontrol kaldırılmaz. Tekillik cezası doğruluğu bozuyorsa ceza varsayılan modelden çıkarılır ve sonuç kaydedilir.

## 8 Kaynaklar

K02 [PyTorch Kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics). K03 [KDL yapılandırması](https://moveit.picknik.ai/main/doc/examples/kinematics_configuration/kinematics_configuration_tutorial.html). K04 [TRAC-IK](https://moveit.picknik.ai/main/doc/how_to_guides/trac_ik/trac_ik_tutorial.html). K05 [pick_ik](https://moveit.picknik.ai/main/doc/how_to_guides/pick_ik/pick_ik_tutorial.html).

K07 [Bensadoun ve diğerleri](https://proceedings.mlr.press/v162/bensadoun22a.html), ICML 2022. K09 [CycleIK](https://arxiv.org/abs/2404.08825v2), IROS 2024. K11 [MimicIK](https://arxiv.org/abs/2606.15148v2), 2026 ön baskı.

K13 [Zhou ve diğerleri Rotation Representations](https://arxiv.org/abs/1812.07035), CVPR 2019. Kaynak erişimi 17 Eylül 2026. Sonuçlar şu anda ÖLÇÜLMEDİ; güncel kayıt `docs/records/STATUS.md` dosyasıdır.

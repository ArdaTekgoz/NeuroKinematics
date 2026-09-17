# NeuroKinematics Studio teknik tasarım raporu

Hedef yazılım v3.0.0 · Belge r1 · 17 Eylül 2026

## 1 Amaç ve kullanıcı değeri

Studio, doğrulanmış kinematik ve çözücü servislerini masaüstü mühendislik iş akışına dönüştürür. Kullanıcı robot tanımlar, hedef poz verir, çözümün geçerliliğini görür, çalışma uzayını inceler ve deney çıktısını dışa aktarır. Bu fazın başarısı görsel zenginlikten çok doğru sonucun anlaşılır ve tekrar üretilebilir sunulmasıdır.

Neuro şemsiyesi altında ürün adı NeuroKinematics Studio'dur. Önceki NeuroKinSim vizyonu, Studio içindeki kinematik simülasyon ve analiz işlevlerine dönüşür. Gerçek robotla canlı senkronizasyon veya kalibrasyon yoksa arayüz fiziksel dijital ikizin doğrulandığını iddia etmez.

Giriş koşulu G2 kararıdır. Nöral yöntem varsayılan motor seçilmemiş olsa da Studio doğrulanmış sayısal motorla çalışabilir. Mevcut teknik sınırlılıklar kullanıcıya sonuçların parçası olarak gösterilir.

| Gereksinim | Kullanıcı işi | Zorunlu çıktı |
|---|---|---|
| REQ-S01 | İkinci robotu tanıtmak | Aynı pipeline ile robot başına yeni model |
| REQ-S02 | Robot ve hedef yönetimi | Frame/birim hatasını engelleyen giriş |
| REQ-S03 | Sonucu ve çalışma alanını incelemek | Poz hatası, limit, collision ve coverage görünümü |
| REQ-S04 | Uzun işi çalıştırmak veya iptal etmek | Donmayan arayüz ve tutarlı görev durumu |
| REQ-S05 | Sonucu yeniden kullanmak | Manifestli proje ve veri dışa aktarımı |
| REQ-S06 | Temiz sistemde çalışmak | Test edilmiş paket, kılavuz ve lisans envanteri |

### İlk kullanıcı senaryosu

Bir robotik öğrencisi veya araştırmacı, desteklenen robotu açar; mevcut eklem durumunu görür; TCP hedefini seçer; IK sonucunda hedef ve ulaşılan pozu birlikte inceler. Başarısızlık varsa nedenini ve hangi kontrolün yapılmadığını görür. Aynı sorguyu başka çözücüyle çalıştırıp karşılaştırma dosyasını alır.

Arayüzün ilk sürümünde serbest biçimli CAD hücresi tasarlama, üreticiye özel robot programı üretme, tam dinamik simülasyon, bulut eğitim yönetimi veya fiziksel robota komut gönderme işlevi yoktur. Çevre geometrisi, desteklenen basit nesneler veya doğrulanmış mesh girdisiyle sınırlanır.

Bu rapor tasarım belgesidir; kullanıcı testi, ikinci robot deneyi veya paket dağıtımı henüz yapılmamıştır.

<!-- pagebreak -->

## 2 İkinci robot ve taşınabilirlik deneyi

İkinci robot adayı UR5'tir; kesin varyant ve model kaynağı S3-01 içinde doğrulanır. ABB IRB 120 alternatif aday olarak saklanır. Robot seçimi fiziksel donanım satın alınmasını gerektirmez. Aynı robotun farklı isimli kopyası ikinci robot kanıtı değildir.

H3 deneyi, mevcut kod commit'i üzerinde yeni URDF/manifest ve konfigürasyonla pipeline'ın yeniden çalışmasını sınar. Yeni robot için farklı ağırlıklar ve normalizasyon üretmek beklenen davranıştır. Eğitim maliyeti, insan müdahalesi, solver-specific kod değişikliği ve hata analizi kaydedilir.

| Adım | Denetim | Kabul |
|---|---|---|
| Model tanıtma | Varyant, frame, eksen, limit ve TCP | Model manifesti tam |
| Kinematik | İlk robotla aynı FK/Jacobian paketi | Aynı sayısal testler geçer |
| Veri ve eğitim | Robot başına bağımsız manifest ve ağırlık | Kopyalanmış etiket veya gizli sabit yok |
| Benchmark | Aynı görev profili, ayrı robot sonuçları | Başarı ve süre robot bazında raporlu |
| Paket | Model–robot eşleşme kontrolü | Yanlış ağırlık yükleme reddedilir |

**H3 kabul hedefi:** Desteklenen seri zincir sınıfında yeni robota geçiş yalnız varlık, manifest ve konfigürasyon değişikliğiyle yapılır; solver mantığına robot markasına özel dal eklenmez. Genel bir yazılım kusuru düzeltilmesi gerekirse ilk robotun regresyonları yeniden çalıştırılır ve iki robot aynı yeni commit'te ölçülür. Bu durumda sıfır kod değişikliği hedefinin ilk denemede sağlanmadığı açıkça kaydedilir.

İkinci robot neural doğruluk hedefini tutturamazsa hata analizi ve sayısal baseline yayımlanabilir. Aynı pipeline'ın çalışması, her robotta aynı başarı oranını veya aynı eğitim süresini garanti etmez. Sorun parametrizasyonda mı, veride mi, model kapasitesinde mi ayrılır.

### Genelleme kelimesinin kullanımı

Bu faz **pipeline taşınabilirliği** ve **robot başına yeniden eğitim** gösterir. Zero-shot, few-shot veya tek modelle bütün robotları çözme iddiası yoktur. Bu tür araştırmalar için eğitim, validation ve test robotları ayrılmış bir görev dağılımı gerekir. MAML'in özgün tanımı çok sayıda görevden hızlı adaptasyona yöneliktir [K17]; iki robotu yeniden eğitmek MAML deneyinin yerine geçmez.

Araştırma hattına aktarılacak soru, farklı robotlarda hangi veri/öğrenme bileşenlerinin tekrar kullanılabildiğidir. Bunu yanıtlayacak veri birikimi oluşmadan GNN veya meta-learning, Studio'nun zorunlu bileşeni yapılmaz.

<!-- pagebreak -->

## 3 Uygulama mimarisi ve ekran akışı

PySide6 uygulama kabuğu ve PyVista/VTK görselleştirmesi başlangıç adaylarıdır. İlk teknik prototip yalnız bir robot mesh'ini yükleyip FK dönüşümleriyle günceller. Lisans, paketleme ve performans sonuçları uygunsa kapsam genişletilir [K14, K18].

| Alan | Gösterilen içerik | Kullanıcı kararı |
|---|---|---|
| Robot paneli | Model kimliği, eklem sırası, limitler, base/TCP | Doğru robotla çalıştığını doğrulamak |
| Hedef ve çözücü | Poz, yönelim, q_current, tolerans profili | Hangi görevin çözüleceğini seçmek |
| 3B sahne | Mevcut, hedef ve ulaşılan poz; collision işaretleri | Sapmayı ve geometri durumunu anlamak |
| Sonuç paneli | Hata, süre, limit, tekillik ve durum kodu | Çıktıyı kullanmak veya yeniden denemek |
| Analiz ve dışa aktarım | Benchmark, heatmap, yörünge ve kayıt | Sonucu paylaşmak veya tekrar üretmek |

GUI, SolverRequest/Result üzerinden servise erişir. Ayrı bir “görsel FK” hesaplama sistemi kurmaz; doğrulanmış model dönüşümlerini gösterir. Ekranda derece seçilse bile iç API radyan kullanır ve dönüşüm giriş sınırında yapılır. Base/TCP değişikliği önceki model veya hedefi sessizce yeniden yorumlamaz; proje uyumsuzluğu açıkça gösterilir.

### İşlerin çalıştırılması

Eğitim, veri üretimi, workspace taraması ve uzun benchmark işleri arayüzün ana olay döngüsünde çalıştırılmaz. İş kimliği, ilerleme, günlük ve iptal durumu bulunan worker süreçleri kullanılır. İptal edilen işin geç dönen sonucu yeni proje üzerine yazamaz. Büyük çıktı dosyaları tamamlanınca atomik olarak görünür hale getirilir.

Render güncellemeleri arayüz thread'inde yapılır. Her solver çağrısında mesh dosyası yeniden yüklenmez; dönüşümler güncellenir. Aynı anda birden çok ağır eğitim işi açma varsayılan olarak sınırlandırılır. Kaynak bütçesi kullanıcıya anlaşılır işlem durumu olarak gösterilir.

### Hata iletişimi

Geçerli kinematik sonuç, çarpışma kontrolünün yapılıp yapılmadığıyla birlikte sunulur. Kırmızı/yeşil renge ek olarak metin ve simge kullanılır. “Çözülemedi” mesajı otomatik “ulaşılamaz hedef”e çevrilmez. Süre aşımı, eksik geometri ve yanlış model gibi kullanıcı kararını etkileyen nedenler görünür olur.

Geliştirme ayrıntıları varsayılan iş akışına yığılmaz. Commit, hash, thread sayısı ve ayrıntılı günlük gibi bilgiler deney ayrıntısı paneli veya dışa aktarılan raporda bulunur. Hedef poz ve sonuç geçerliliği ana akışta kalır.

<!-- pagebreak -->

## 4 Analiz araçları ve dışa aktarım

Workspace görünümü örneklenmiş verinin haritasıdır. Bir voxel'de başarılı bir IK örneği bulunması, o voxel'deki bütün yönelimlerin erişilebilir olduğu anlamına gelmez. Haritada örnek sayısı, yönelim koşulu, test dağılımı ve seçilen metrik gösterilir. Hiç örneklenmemiş bölge ayrı renkle `VERİ YOK` olarak işaretlenir.

Tek bir birleşik uygunluk puanı başlangıç varsayılanı değildir. Konum hatası, yönelim hatası, başarı oranı, tekillik ve collision farklı katmanlarda incelenir. Birleşik skor daha sonra eklenirse ağırlıkları görünür ve değiştirilebilir olur; fiziksel güvenlik puanı olarak adlandırılmaz.

### Yörünge inceleme

Yol oynatımında zaman, joint konumu, hız, ivme ve jerk eğrileri birlikte izlenebilir. Hedef poz dizisi ile elde edilen TCP izi ayrı gösterilir. Başarısız örnekler atlanıp düzgün görünen bir video üretilmez; başarısız adım ve son geçerli konfigürasyon görünür olur.

Çarpışma etiketinde tek konfigürasyon veya örneklenmiş yol denetimi olduğu açıkça belirtilir. Varsayılan çıktının fiziksel robotta çalıştırılabilir program olmadığı dosya türüyle de anlaşılır. İlk dışa aktarım genel JSON/CSV, konfigürasyon ve model manifestidir; üretici dili postprocessor ayrı proje kararıdır.

| Dışa aktarım | İçerik | Doğrulama |
|---|---|---|
| Proje paketi | Robot referansı, hedefler, konfigürasyon, şema sürümü | Aç–kaydet–aç eşliği |
| Benchmark | Sorgu başına sonuç, özet, donanım ve sürüm | Ham ve özet sayıları tutarlı |
| Yörünge | Zaman, eklemler, hedef ve gerçekleşen poz | Birimler, örnek sayısı, geçersiz adımlar korunur |
| Model paketi | Ağırlık, normalizasyon, robot/TCP hashleri | Yanlış robotla açma reddedilir |

### Yerel çalışma ve lisans kayıtları

İlk ürünün çalışma akışı çevrimdışıdır. Kullanıcı robot verisi veya mesh dosyası izni dışında başka yere gönderilmez. Ağ gerektiren opsiyonel indirme veya eğitim işlemi ayrı eylem olur. Kullanıcı kaynaklı URDF/mesh dosyalarını tekrar dağıtma hakkı, yazılım kütüphanesi lisansından ayrı kaydedilir.

Paketleme envanterinde bileşen, kullanılan sürüm, kaynak adresi, lisans metni, dağıtılan dosya ve bildirim gereksinimi tutulur. Qt for Python ve üçüncü taraf Qt bileşenleri için kullanılan modüller özelinde resmi bilgiler esas alınır [K14]. Projenin kendi kod lisansı ayrı ADR kararıdır; bütün bağımlılıkların aynı lisansa sahip olduğu varsayılmaz.

<!-- pagebreak -->

## 5 Test planı ve kullanıcı değerlendirmesi

İlk masaüstü dağıtım hedefi Windows 11 x64 CPU olarak planlanır; Linux araştırma/benchmark ortamı ayrı tutulur. Bu platform kararı S3-02 prototipinde uygulanabilirlikle doğrulanır. Windows paketlemesi bloke olursa Linux paketi ayrı destek profiliyle aday sürüm olabilir; Windows hedefi tamamlanmış gösterilmez.

| Test | Deney | Kabul hedefi |
|---|---|---|
| T-S01 | İkinci robot pipeline'ı | H3 sonucu ve iki robot regresyonu mevcut |
| T-S02 | Poz/frame/birim ve model uyumsuzluğu | Sessiz yanlış dönüşüm veya model seçimi yok |
| T-S03 | Eğitim/tarama sırasında arayüz | Etkileşim sürer; iptal isteği ≤ 2 s içinde görünür |
| T-S04 | Aç–kaydet–aç, kesilen iş, bozuk paket | Veri kaybı yok; hatalı paket açıkça reddedilir |
| T-S05 | Heatmap ve yörünge | Verisiz/başarısız bölgeler doğru temsil edilir |
| T-S06 | Temiz hedef işletim sistemi | Python geliştirme kurulumu gerektirmeden demo çalışır |
| T-S07 | En az üç hedef kullanıcıyla görev testi | Kritik görevlerde toplam ≥ yüzde 80 tamamlama hedefi |

Kullanıcı testi görevleri: robot açma, hedef girme, yanlış frame girdisini fark etme, başarısızlığı açıklama ve sonuç dışa aktarma. Süre, yardım ihtiyacı, hata ve tamamlanma kaydedilir. Üç kullanıcı küçük bir kullanılabilirlik çalışmasıdır; pazar doğrulaması veya genellenebilir kullanım istatistiği değildir. Katılımcı iletişimi ayrıca yürütülecek iştir; bu plan herhangi bir davetin gönderildiği anlamına gelmez.

### G3 kabulü

İkinci robot deneyi tamamlanmış, kritik GUI/servis testleri geçmiş, desteklenen işletim sisteminde temiz kurulum denenmiş, örnek proje ve kullanıcı rehberi mevcut olmalıdır. Kullanıcı değerlendirmesinde kritik yanlış yorum üreten sorunlar çözülür. Sayısal üstünlük, bu fazın arayüz kalitesini tek başına belirlemez.

### Roadmap

S3-01 ikinci robotu; S3-02 servis ve GUI kabuğunu; S3-03 analiz ve görev yönetimini; S3-04 dışa aktarım ve paketlemeyi; S3-05 kullanıcı testi, portföy demosu ve G3 kapanışını kapsar. Etkin emek tahmini 80–130 saattir. Önce ikinci robot ve servis sözleşmesi kapanır; kapsamlı arayüz bunların üzerine kurulur.

### Sürüm sonrası

MAML, generatif IK, GNN, ROS 2 entegrasyonu, Jetson, gerçek robot ve NeuroLocalization yeni ihtiyaç dosyalarıyla değerlendirilir. Her öneri giriş verisi, test, bütçe ve durdurma şartı taşır. Studio tamamlandı diye tüm bu araştırmaların zorunlu hale gelmesi önlenir.

<!-- pagebreak -->

## 6 Portföy ve akademik çıktı

Portföy sunumu bir çalışan demo, sürüm etiketi, kurulum tarifi, gerçek benchmark tablosu, bilinen sınırlılıklar ve yeniden üretim komutlarından oluşur. Mimari ve kanıt akışı açıklanır. İlk robot ve ikinci robot sonuçları birbirine karıştırılmaz; modelin başarılı olduğu dağılım açıkça belirtilir.

Akademik rapor, tasarım metninin sonuç alanlarına ölçümler eklenerek güncellenir. Araştırma sorusu, yöntem, baseline, veri, ablation, sonuç, geçerliliğe yönelik tehditler ve sınırlılıklar ayrı tutulur. Negatif veya karma sonuçlar silinmez. Yayına uygunluk danışmanla hedef dergi/konferansın kapsamı ve deney gücü üzerinden değerlendirilir; makale kabulü bir release koşulu değildir.

Ticari değerlendirme için sonraki adım, mühendislik iş akışının kim için ne kadar zaman kazandırdığını ölçmektir. Üç kullanıcıyla görev testi ürün kullanım sorunlarını bulur; ücretli talep, destek maliyeti veya lisans uygunluğu hakkında tek başına sonuç vermez. Bu başlıklar ayrıca araştırılır.

### Fazdan devredilecek kayıtlar

- İki robotun model/veri/checkpoint kimlikleri ve karşılaştırılabilir sonuç dosyaları.
- Desteklenen platform, kurulum paketi, örnek proje ve kullanıcı rehberi.
- Açık sorun listesi, hata kayıtları ve geometri doğrulama kapsamı.
- Kullanılabilirlik bulguları ve uygulanan düzeltmeler.
- Sonraki sürüme taşınan işlerin ön koşulları ve gerekçeleri.

## 7 Kaynaklar

K14 [Qt for Python lisansları ve üçüncü taraf bileşenler](https://doc.qt.io/qtforpython-6/licenses.html).

K17 [Finn, Abbeel ve Levine Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://proceedings.mlr.press/v70/finn17a.html), ICML 2017, PMLR 70, 1126–1135.

K18 [PyVista resmi belgeleri](https://docs.pyvista.org/), 3B görselleştirme bileşeni için uygulama kaynağı.

K01 [Pinocchio resmi depo](https://github.com/stack-of-tasks/pinocchio), ortak model ve kinematik servis.

Kaynak erişimi 17 Eylül 2026. G3 sonucu ve kullanıcı testi şu anda ÖLÇÜLMEDİ durumundadır. Güncel görevler `docs/roadmaps/S3_Studio.md`, araştırma birikimi `docs/RESEARCH_BACKLOG.md` dosyalarında tutulur.

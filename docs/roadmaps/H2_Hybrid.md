# Hybrid — geometrik servis ve yeni neural katkı planı

Roadmap r3 · 11 Ekim 2026 · Durum PLANLANDI / NOT_STARTED · Tahmin: 150–232 etkin saat.
Yazılım hedef sürümü belge revizyonundan ayrıdır; henüz release yapılmadı.

Hedef yazılım v2.0.0. Özgün H2-01–06 kimlikleri ve kabul eşikleri korunur;
iki yeni görev ve geometrik servis ekleri ADR-027 ile tanımlanır.
[Özgün rapor](../raporlar/03_Hybrid_v2_0_r1.md) tarihsel tasarım kaynağıdır;
[r3 ek şartname](H2_REVISION_SPEC.md) mevcut bileşen rollerini ve deney kollarını belirtir.
Giriş: C1-07 G1 araştırma PASS ve C1-06S geometrik bileşen PASS.
Bu giriş yeni neural başarı veya H1 desteği anlamına gelmez.

Ana sıra H2-01→02→N1→N2→03→04→05→06. Bağımlılık matrisi izin verse de tek
aktif ana görevle ilerlenir; örneğin 04, eğitim beklerken teknik olarak açılabilir.
Şu an tek sıradaki iş **H2-01**; bu plan revizyonu H2 uygulamasını başlatmamıştır.

## Uygulama sırası

| Sıra | Görev | Ön koşul | Saat |
|---|---|---|---|
| 1 | [H2-01 — Kabul politikası ve bileşen kimliği](../tasks/H2-01.md) | C1-07, C1-06S | 12–18 |
| 2 | [H2-02 — Ortak bütçeli çok motorlu IK servisi](../tasks/H2-02.md) | H2-01 | 26–40 |
| 3 | [H2-N1 — Yeni neural deney ve veri sözleşmesi](../tasks/H2-N1.md) | H2-02 | 16–24 |
| 4 | [H2-N2 — Yeni neural adayın eğitimi ve seçim kapısı](../tasks/H2-N2.md) | H2-N1 | 24–40 |
| 5 | [H2-03 — H1 ve geometrik ürün karşılaştırması](../tasks/H2-03.md) | H2-02, H2-N2 | 20–30 |
| 6 | [H2-04 — Solver rollout ve ortak geometri denetimi](../tasks/H2-04.md) | H2-02 | 28–42 |
| 7 | [H2-05 — Öğrenilmiş bileşenin ONNX eşliği](../tasks/H2-05.md) | H2-02, H2-N2 | 14–22 |
| 8 | [H2-06 — G2 servis kabulü ve MR devri](../tasks/H2-06.md) | H2-03, H2-04, H2-05 | 10–16 |

## Görev bazında kabul

### H2-01 — Kabul politikası ve bileşen kimliği

İstek ve sonuç sözleşmesini geometrik, sayısal ve öğrenilmiş adayların ortak denetimine açmak.

**Gereksinim:** REQ-H01, REQ-HG01. **Test:** T-H01, T-HG01.

**Çıktı:** C1-07 HYBRID_HANDOFF ve C1-06S HANDOFF hashlerini yükle; learned=false, robot/TCP/config/model kimliğini taşı. SolverRequest: q_current, hedef pose, robot_revision, motor politikası, toplam bütçe, doğrulama profili ve collision gereksinimi; Result: aday, metrikler, kaynak motor, süre, kontrol kapsamı ve tanılar. Model/robot uyumsuzluğu, bozuk quaternion, yanlış boyut, NaN/Inf, limit, UNRESOLVED, LATE_VALID ve iptal durumlarını tanımla. Kinematik/geometri/zaman/neural readiness alanlarını ayır; varsayılan motor seçimini henüz başarı ilan etmeden configte açık tut.

**Kabul:** T-H01: özgün hata/kabul durumlarının tamamında yanlış başarı 0. T-HG01: yanlış robot hashli analitik bileşen çağrılmaz; UNRESOLVED erişilemezlik değildir; collision istenip NOT_CHECKED ise tam kabul yok. Profil A 2 mm/1° ve B 1 mm/0,5° aynen korunur; 1e-8 rad yalnız sayısal limit işlem payıdır.

**Başarısızlıkta:** Kimlik veya geometri kapsamı belirsizse açık hata; bilinmeyen sonucu başarıya yükseltme.

**Kanıt:** `experiments/H2-01/`; PLANLANDI / NOT_RUN.

### H2-02 — Ortak bütçeli çok motorlu IK servisi

Doğrulanmış analitik adayı ve mevcut sayısal/neural yolları tek süre bütçesinde çalıştırmak.

**Gereksinim:** REQ-H02, REQ-HG02. **Test:** T-H02, T-HG02.

**Çıktı:** Uyumlu robotta geometry-first ürün adayı, numerical-only ve neural-seeded araştırma politikalarını ayrı seçilebilir yap. Analitik adayı bağımsız son doğrulamaya sok; başarısız/uyumsuz adayda kalan bütçe ile sayısal yola geç; geometri olmayan robota KUKA formülü uygulama. Neural politikasında bir refinement ve en fazla bir alternatif restart; tüm ön işlem, aday üretimi, inference, collision ve son denetimi aynı monotonic deadline içine al. İptal/late-result ve process worker yaşam döngüsünü kur; analitik çağrının kendisinin kesilebilir veya hard real-time olduğunu varsayma. Her motorun aday, iterasyon, refinement/restart, soğuk yükleme ve toplam sürelerini ayrı logla.

**Kabul:** T-H02: iyi/kötü/NaN seed ve bütçe tükenmesi doğru geçiş verir; yeni denemeye tam bütçe tahsis edilmez. T-HG02: fake clock ve gerçek geciken worker ile geç sonuç zamanında başarı sayılmaz; iptal sonrası sonuç yayımlanmaz. Her kabul edilmiş aday gerçek çıktı üzerinde FK/limit/etkin collision denetiminden geçer; geometry bypass yanlış kabulü 0.

**Başarısızlıkta:** Deadline servisi doğrulanmadan C1-06S bileşen süresini H2 servis garantisi diye raporlama; sorunlu motoru devre dışı bırak ve G2 açık kalsın.

**Kanıt:** `experiments/H2-02/`; PLANLANDI / NOT_RUN.

### H2-N1 — Yeni neural deney ve veri sözleşmesi

Yeni eğitimi eski sıfır başarı kampanyasından ölçülebilir değişikliklerle ayırmak.

**Gereksinim:** REQ-HN01. **Test:** T-HN01.

**Çıktı:** Rolü seed proposal olarak ön kaydet; birincil K=1 durum/pose ile şartlandırılmış öneri, ikincil çok adaylı K=4 yaklaşımını ve basit MLP kontrolünü karşılaştırma sırası ile yaz. Mimari/loss kararı uygulamadan önce yeni ADR ile sabitlensin. Yeni dataset sürümü üret: local/wide, bağımsız kök/yön, sınır/tekillik grupları; train-only normalizasyon. Analitik teacher sonuçlarını bağımsız FK ile doğrula; mevcut tek-aday API bütün dalları verdi varsayma. Dal/tur eşdeğerlerini bounded joint uzayında tut; multi-candidate teacher gerekiyorsa denetlenmiş branch API ekle. Etiketsiz veya çözülemeyen örnekleri silmek yerine açık maskeler ve oranlarla sakla. Data-size pilotu ve kaynak tavanını ölçerek tam eğitim örnek sayısını seç; sayıyı final sonucuna bakarak değiştirme. Train precision, aynı kökte yeni yön, yeni kökte validation ve zero-offset/türev tanılarını ayrı raporla. Eski C1 finalleri ve görülmüş C1-06S finali yeni seçim/final seti olamaz. Üç seed, maksimum iki hedefli düzeltme, seçme ölçütü, bütçe ve yeni final üretim protokolünü dondur.

**Kabul:** T-HN01: grup/kök sızıntısı ve çapraz split tekrar 0; doğrulanmış teacher kabul hatası 0; eksik etiketler sayılmış; teacher/model/hash zinciri tam. 64 örnekte overfit tek başına GO değildir. Ayrı train/new-direction/new-root pilotları ve aynı motorlu seed fayda ölçümü mevcut. Final hedefleri henüz üretilmeden selection policy ve veri/architecture/config freeze hazır; eski negatif deney aynı kimlikle yeniden yazılmamış.

**Başarısızlıkta:** Veri veya köken denetimi başarısızsa eğitim açılmaz; kaynak bütçesi gerçek pilotla yenilenir. Ek veri otomatik çözüm garantisi değildir.

**Kanıt:** `experiments/H2-N1/`; PLANLANDI / NOT_RUN.

### H2-N2 — Yeni neural adayın eğitimi ve seçim kapısı

Gerçekten yeni ağı eğitmek ve H1 deneyine hashli, rolü belirli bir aday vermek.

**Gereksinim:** REQ-HN02. **Test:** T-HN02.

**Çıktı:** Kayıtlı yeni model/loss/config ile üç bağımsız eğitim seed çalıştır; eski checkpoint continuation ile sıfırdan yeni eğitim etiketlerini ayır. Adayları train/validation üzerinde aynı toplam seed+refinement maliyetiyle değerlendir; neural direct A/B başarısını ayrıca raporla, seed başarısı yerine kullanma. K=4 denenirse tüm aday üretim/arama maliyeti sayılır; çoklu adaylı klasik kontrol de aynı bütçede çalışır. K=4 yararını K=1 H1 ile karıştırma. İki hedefli düzeltme tavanı ve kaynak tavanında dur; aday seçimi validation ile tamamlanınca model kartı, checkpoint, train/split kimliği ve ONNX giriş/çıkış sözleşmesini dondur. Her seed için kayıp, gerçek poz/joint hataları, finite/limit oranı, local/wide başarı, refinement ihtiyacı ve maliyeti kaydet.

**Kabul:** T-HN02 teknik teslimi: üç seed çalışması veya belgelenmiş teknik engel; checkpoint kimliği, tekrar inference ve veri denetimi; seçim yalnız validation ile yapılmış. G-N adayı ancak H2-03 bağımsız ölçümüne gidebilir; eğitim kaybı düşmesi, az örnekte ezberleme veya teacher başarısı neural ürün PASS değildir. Yeni aday bulunamazsa görev sonuç kaydı NEGATIVE/NOT_MET olur; H2-03 eski kontrolü ölçebilir fakat yeni model başarılı denmez.

**Başarısızlıkta:** İki düzeltmede yarar yoksa körlemesine uzun eğitim yok. G-N açık kalır; eksik katkı için hata sınıfına dayalı yeni ADR/araştırma ön kaydı gerekir.

**Kanıt:** `experiments/H2-N2/`; PLANLANDI / NOT_RUN.

### H2-03 — H1 ve geometrik ürün karşılaştırması

Neural katkıyı motor değişiminin etkisinden ayırıp bağımsız ölçmek.

**Gereksinim:** REQ-H03, REQ-HG03. **Test:** T-H03, T-HG03.

**Çıktı:** B-H01 q_current, B-H02 merkez, B-H03 neural seed, B-H04 klasik restart: aynı residual/Jacobian, sayısal motor, tolerans, iterasyon ve toplam bütçe. B-H03-old ve B-H03-new kimliklerini ayrı tut; local-only sonuçları genel ana sonuç gibi sunma. B-H05 geometrik servis ayrı ürün karşılaştırmasıdır. 10.000 ana + 1.000 sınır + 1.000 tekillik bağımsız sorgu; ana local/wide 50/50; üç eğitim seed; 10/50 ms, batch 1, beş timing geçişi; yöntem sırası seed ile dengeli. Model/config freeze sonrası taze final; C1 finalleri tuning verisi değil. Tüm başarısız/geç sorguları süre ve başarı paydasında tut. Sorgu/grup bootstrap %95 CI; tekrarları bağımsız hedef sayma; cold start/NN/refinement/son doğrulama ve bellek ayrıca raporla.

**Kabul:** T-H03: P95 toplam süre ≥%20 azalma, başarı kaybı ≤1 yüzde puanı; süre oranı CI üstü <1, başarı farkı CI altı ≥−1 yüzde puanı. Özgün H1 eşiği korunur. T-HG03: geometric-vs-neural/system sonuçları ayrı tablo; motor farkından neural kazanç çıkarılmaz. G-N yalnız dondurulmuş kapsamda doğrulanmış öğrenilmiş katkıyla kapanır; direct-neural iddiası için ayrı A/B testi gerekir.

**Başarısızlıkta:** H1 reddedilirse G-N NOT_MET; doğrulanmış geometri/sayısal motor ürün adayı olarak kalır. İstatistiksel belirsizlik başarıya çevrilmez.

**Kanıt:** `experiments/H2-03/`; PLANLANDI / NOT_RUN.

### H2-04 — Solver rollout ve ortak geometri denetimi

Zamanlı hedeflerde çözücü sürekliliğini ve geometri denetimini doğrulamak; genel planner kurmamak.

**Gereksinim:** REQ-H04, REQ-HG04. **Test:** T-H04, T-H05, T-HG04.

**Çıktı:** En az 30 yörünge × 200 zamanlı örnek; sonraki q_current önceki kabul edilmiş çıktı; teacher forcing yok. Geometrik dal değişimini, q5≈0 ve sınır/tur eşdeğerlerini test et; bounded joint mesafesi kullan, yapay 2π sarmalama yapma. Poz/joint hata, hız, ivme, jerk, >0,2 rad jump ve singularity metriklerini aynı zaman çizelgesinde ver; analitik türev referanslarını test et. ≥100 etiketli çevre/self collision örneği; geometry/visual mesh ayrımı, allowed-pair, ölçek/frame hashleri; state/edge API’lerini ileride MP’nin kullanacağı tek servis olarak kur. Örneklenmiş yol denetimi en fazla 1° joint ve 5 mm tahmini TCP adımıyla kapsamlı kayıt; geometri eksikse UNKNOWN. Eksik ivme limitine üretici limiti icat etme.

**Kabul:** T-H04/T-H05 özgün kabulü korunur; bilinen çarpışmalar kaçırılmaz; türevlerin analitik kontrolü geçer. T-HG04: branch flip/singular/sınır vakaları raporlanır; collision NOT_CHECKED fiziksel veya yol güvenliği sayılmaz. H2 rollout testi TR trajectory generator PASS yerine kullanılamaz; sonraki katman aynı geometri/robot kaynağını kullanır.

**Başarısızlıkta:** Bilinmeyen geometri veya limit deneysel/UNKNOWN; kesintisiz çarpışmasızlık iddiası yok. Kritik yanlış kabul G2 engelidir.

**Kanıt:** `experiments/H2-04/`; PLANLANDI / NOT_RUN.

### H2-05 — Öğrenilmiş bileşenin ONNX eşliği

Seçilmiş neural tensor hattını taşınabilir kılmak; geometrik kodu neural export diye sunmamak.

**Gereksinim:** REQ-H05. **Test:** T-H06.

**Çıktı:** Yeni seçilmiş aday varsa onu; yoksa açıkça araştırma olarak etiketlenen dondurulmuş kontrolü FP32 CPU ONNX’e aktar. ≥1.000 sabit girdiyle PyTorch/ONNX ham joint ve FK farklarını, sınırdaki kabul farklılıklarını kaydet. Ön/son tensor işlem, normalizasyon, quaternion/pose temsili, model/robot hashleri ve runtime sürümlerini pakete koy. Analitik, FK, collision, budget/fallback host tarafında kalır; bunlar ONNX export edilmiş sayılmaz. Her runtime gerçek çıktısında bağımsız son kabul uygula; cold start ve total latency ölç.

**Kabul:** T-H06: maksimum ham joint farkı ≤1e−5 rad; FK farkı ≤0,1 mm ve ≤0,01°; finite/limit kontrolleri ve final doğrulama eşliği. Eşik geçilmeden nominal v2 teslimi kapanmaz; research ONNX PASS, neural fayda PASS anlamına gelmez.

**Başarısızlıkta:** Export kusurunu düzelt veya runtime NOT_SUPPORTED kaydı; eşiği düşürme. Yeni model seçilince eski model eşlik testi otomatik taşınmaz.

**Kanıt:** `experiments/H2-05/`; PLANLANDI / NOT_RUN.

### H2-06 — G2 servis kabulü ve MR devri

Desteklenen motorları ve neural katkı durumunu ayrı kapılarla teslim etmek.

**Gereksinim:** REQ-H06, REQ-HG06. **Test:** T-H07, T-HG06.

**Çıktı:** Yeni temiz kilitli ortamda servis kurulum/replay; C1-06S mevcut-runtime replay ile karıştırma. G2 servis matrisi, G-N neural katkı matrisi, runtime/robot/geometry/budget kapsamı, hashli model/solver kartları ve ham loglar. Varsayılan KUKA motorunu H2-03/04 sonuçlarına göre seç; neural kazanç yoksa geometric/numerical, araştırma adayları açık etiketli. MR için capability registry gereksinimi, ortak SolverRequest/Result, örnek istekler, hata durumları, paket ve artifact geri yükleme tarifi devret.

**Kabul:** T-H07/T-HG06: kritik kabul/timeout/collision/kurulum/replay testleri PASS; H1 kararı, ONNX eşliği ve yörünge/geometri kapsamı kayıtlı. G2 yalnız özgün zorunlu şartlar ve r3 ek servis testleri kapanınca PASS; G-N ayrıca SUPPORTED/REJECTED/INCONCLUSIVE. H2-N2 teknik engeli giderilmeden tamamlandı yazılmaz; olumsuz bilimsel sonuç belgelenebilir. Kaynak/manifest paketlenmeden MR açılmaz.

**Başarısızlıkta:** ONNX veya kritik servis kusuru varsa G2 açık; G-N negatifse öğrenilmiş vizyonun tamamlandığı iddia edilmez.

**Kanıt:** `experiments/H2-06/`; PLANLANDI / NOT_RUN.

## Faz kapısı

**G2:** H2 servis, bütçe, rollout/geometri, ONNX ve temiz kurulum zorunlu
testleri PASS; H1 açık karara bağlanmış. **G-N:** öğrenilmiş katkı H2-03'te
kanıtlanmışsa kapanır. Negatif H1, G2'yi özgün kurallara göre engellemeyebilir;
G-N'yi kapatmaz. Eski MLP otomatik varsayılan değildir. Sonraki iş MR-01.

Tüm işler için gereksinim → değişiklik → test → ham kanıt bağı tutulur.
Uygulamada protokol/seed/kimlikler sonuç görülmeden dondurulur; başarısız kayıtlar
korunur. Yukarıdaki süreler tahmindir; GPU bekleme, donanım temini ve kullanıcı
testi takvimini kapsamaz. İlk iki görevden sonra yalnız tahminler güncellenir.

[Ana roadmap](MASTER_ROADMAP.md) · [Kaynak eşlemesi](SOURCE_MAPPING_r3.md)
· [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).

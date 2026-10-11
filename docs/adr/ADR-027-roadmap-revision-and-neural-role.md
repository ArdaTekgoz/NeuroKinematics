# ADR-027 — Geometrik Core sonrası neural rolü ve genişletilmiş roadmap

Durum: KABUL EDİLDİ (plan kararı; uygulama kabulü değildir) · 11 Ekim 2026 · r1.
Dayanak: kullanıcının dört yeni belgeyi projeye yerleştirme ve H2 ile devam etme talebi.

## Sorun

C1-07 araştırma kabulü ile neural ürün başarısı aynı şey değildir. C1-06S,
KUKA'ya özel geometrik IK başarısını göstermiştir; yeni neural model eğitmemiştir.
Eski H2 planı geometrik motoru tanımıyor; yeni belgeler çoklu robot, yörünge,
planlama ve fiziksel yürütme ekliyor. Eski görev kimliklerini silmeden birleştirmek gerekir.

## Karar

1. F0/G0, C1/G1, C1-06R ve C1-06S'nin dondurulmuş kanıtlarına dokunulmaz.
   Kaynak faz raporları r1 korunur; bu ADR ve roadmap r3 ileriye dönük ek kapsamı tanımlar.
2. AnalyticIK, yalnız doğrulanmış robot hash'i için capability bildirir. Generic
   serviste robot adıyla dallanmak yerine yetenek/uyumluluk kaydı kullanılır.
   KUKA geometrisini bütün robotlara zorlamak veya kanıtlanmış bileşeni silmek yoktur.
3. H2-01/02'de geometrik aday, sayısal motor ve neural öneri aynı kabul ve süre
   politikasına bağlanır. Eski neural modeller kontrol/araştırma adaylarıdır;
   varsayılan motor veya doğrudan robot komutu olmazlar.
4. H2-N1 veri/deney hazırlığı ve H2-N2 yeni neural eğitim kampanyası zorunlu plan
   görevleri olur. Öğrenmenin ilk rolü doğrulanabilir başlangıç önerisidir.
   Geometrik çözücü teacher ve güçlü baseline sağlar; faydayı otomatik kanıtlamaz.
   Dal seçimi, öğrenilmiş arama ve robotsuz analitik seed için ayrı deney kimliği gerekir.
5. H1'in aynı sayısal motorda ≥%20 P95 iyileşme, ≤1 yüzde puanı başarı kaybı ve
   güven aralığı şartları korunur. Geometrik motor kıyası ayrı B-H05 koludur.
   Geometrik başarı, H1 veya yeni direct-neural başarı olarak yazılmaz.
6. G2 servis kabulü ile G-N neural katkı kapısı ayrı tutulur. Yeni eğitim yapılır;
   sınırlı kampanya olumsuzsa G2, özgün rapordaki koşullarla kapanabilir, fakat
   G-N AÇIK/NOT_MET kalır. Neural vizyonu tamamlandı denmez; sonraki araştırma
   seçimi hata analizi ve ayrı ön kayıt gerektirir. Sonsuz eğitim döngüsü yoktur.
7. Sıra: H2 → MR (generic onboarding) → S3-01/02 → TR → MP → S3-03/04/05.
   S3-01'in H3 değerlendirmesi korunur; MR altyapısı tekrar yazılmaz.
   TR ve MP, Studio yetenek uzantılarıdır; ikinci bir GUI/faz çekirdeği oluşturmaz.
8. H2-04, solver rollout ve geometri doğrulamasıdır. TR zaman-parametreli hareket
   üretir; MP geometrik yol arar. Çalışma zamanında MP, TR'den önce gelir.
9. PE kaynak belgesinin sırası düzeltilir: PE-07A güvenlik gereksinimleri başta;
   PE-05A temel kalibrasyon, PE-06 HIL ve PE-07B yürütme kilitleri PE-04'ten önce.
   PE-05B ölçülmüş sim-to-real değerlendirmesi PE-04'ten sonra. Böylece gerçek
   hareket, temel kalibrasyon/hata testleri henüz yokken planlanmış olmaz.
10. Digital twin kararı yalnız yazılım yürütme ön koşuludur; fiziksel güvenlik
    onayı veya ölçülmüş fiziksel doğruluk yerine geçmez. TK56 referans platform
    adayıdır; mevcut donanım, limit, denetleyici ve kalibrasyon özellikleri varsayılmaz.

## Sonuçlar ve alternatifler

Eski MLP'yi değiştirmeden varsayılan yapmak kanıtsız fayda iddiasına yol açardı.
Her robot için analitik IK yazmak generic hedefi bozardı. Baştan her şeyi yeniden
yazmak doğrulanmış varlıkları kaybettirirdi. Seçilen yol, ortak doğrulayıcı altında
birden çok motor ve deneyle ölçülen neural katkıdır.

Yeni mimari öğrenilmiş başarının garantisi değildir. G2 ve G-N ayrı raporlanır;
profil kapsamı daraltılarak genel neural başarı iddia edilmez. Yeni robotlar için
NUMERICAL_READY, NEURAL_VALIDATED ve FULLY_READY ayrı kanıt ister.

## İzlenebilirlik

[Ana plan](../roadmaps/MASTER_ROADMAP.md) · [H2 ek şartname](../roadmaps/H2_REVISION_SPEC.md)
· [Kaynak eşlemesi](../roadmaps/SOURCE_MAPPING_r3.md)
· [C1 denetimi](../../experiments/PLAN-20261011-ROADMAP/C1_READINESS_AUDIT.md).
Önceki planların aynı baytlı kopyaları `docs/roadmaps/archive/pre-r3-20261011/`
altında; SHA listesi `experiments/PLAN-20261011-ROADMAP/pre-revision-manifest.json`.

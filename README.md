# NeuroKinematics

NeuroKinematics, doğrulanmış robot kinematiğini öğrenilmiş başlangıçlarla destekleyen,
ölçülebilir ve yeniden üretilebilir hibrit IK sistemi geliştirme projesidir.

**11 Ekim 2026: Foundations/G0 ve Core/G1 araştırma kabulü PASS; C1-06S geometrik
IK PASS. Hybrid uygulaması başlamadı; sıradaki görev H2-01. Yeni neural model
henüz eğitilmedi; doğrudan neural hedef NOT_MET, neural katkı H1 NOT_MEASURED.**

[Önce buradan başla](START_HERE.md) · [Tam devir rehberi](docs/records/PRE_HYBRID_HANDOFF.md)
· [Ana roadmap r3](docs/roadmaps/MASTER_ROADMAP.md) · [Güncel durum](docs/records/STATUS.md).

## Kabul edilen sonuçlar ve açık hedefler

| Katman | Ölçülen sonuç | Kabulün kapsamı |
|---|---|---|
| Foundations | Referans/bağımsız FK 10.000 q: max 4,751e-16 m / 9,160e-16 dönüş matrisi farkı; Jacobian max normalize fark 1,768e-10 | G0 PASS; exact KR6 URDF matematik sözleşmesi |
| Core neural araştırma | C1-06: 21 checkpoint × 12.000 sorgu, A/B 0; kontrollü C1-06R tanıları genelleşmeyi çözmedi | G1 araştırma PASS; doğrudan neural ürün hedefi karşılanmadı |
| C1-06S geometrik kurtarma | Yeni final 12.000/12.000 A/B, 60.000 ölçümde yanlış kabul 0; 315 test PASS | Robot-özel geometrik IK PASS; neural model değildir |
| Hybrid ve yeni neural eğitim | H2-01 ve sonraki 7 görev planlandı | NOT_STARTED; H1, G2, G-N henüz kabul edilmedi |

Geometrik bileşen exact KUKA KR 6 R900 sixx için cebirsel aday dalları üretir,
eklem limitlerini ve bağımsız FK ile pose hatasını denetler. Öğrenilmiş ağırlığı
yoktur. Sonlu final iş yükünün başarısı tüm erişilebilir uzay garantisi değildir;
base-axis araması sonlu grid içerir. P95 CPU batch-1 0,5029 ms, max 103,1543 ms
ölçülmüştür; sert gerçek-zaman garantisi yoktur. Collision ve fiziksel güvenlik
denetlenmedi. Eski neural modeller yalnız hashli araştırma karşılaştırmalarıdır.

[G0 kanıtları](experiments/F0-06/FOUNDATIONS_EVIDENCE_INDEX.md) ·
[G1 kararı](experiments/C1-07/G1_DECISION.md) ·
[Geometrik kabul](experiments/C1-06S/acceptance.json) ·
[C1 hazırlık denetimi](experiments/PLAN-20261011-ROADMAP/C1_READINESS_AUDIT.md).

## Güncel uygulama sırası

Foundations → Core araştırma + geometrik kurtarma → Hybrid → MultiRobot →
Studio onboarding/GUI → Trajectory → MotionPlanning → bütünleşik Studio.
Fiziksel yürütme, tanımlı donanım/kalibrasyon/HIL/interlock kapılarına bağlıdır.

H2 sırası: **H2-01 → H2-02 → H2-N1 → H2-N2 → H2-03 → H2-04 → H2-05 → H2-06**.
H2-N1/N2 yeni veri/model protokolünü ve gerçekten yeni üç seed eğitimini içerir.
H2-03 eşli neural faydayı ölçer. G2 servis kabulü ve G-N neural katkı kabulü ayrıdır;
başarısız bir neural sonucu geometrik başarıyla başarılıya çevirmek yasaktır.

39 ileri görevin bağımlılıkları, gereksinimleri ve kabul testleri
[kanonik roadmap](docs/roadmaps/MASTER_ROADMAP.md),
[makine tarafından okunabilir plan](docs/roadmaps/ROADMAP_TASKS_r3.json),
[H2 ek şartname](docs/roadmaps/H2_REVISION_SPEC.md) ve
[izlenebilirlik](docs/TRACEABILITY.md) içindedir.
Yazılım sürümü 0.1.0; roadmap sürüm hedefleri yayımlanmış release değildir.

## Kurulum, kanıt ve kurtarma

Kanonik platform Windows 11 x64, Pixi 0.81.0, kilitli Python 3.12.14,
NumPy 2.5.3 ve Pinocchio 4.1.0'dır. Neural Core replay ayrı hashli
Torch 2.10.0+cu128 overlay kullanır. [SETUP](docs/SETUP.md) ve
[devir rehberi](docs/records/PRE_HYBRID_HANDOFF.md) gerçek komutları verir.
Yeni eğitim veya final çalıştırmadan önce ilgili görevin protokolü dondurulur.

Kod/belge checkpoint'i ve Git geçmişi bundle'da; büyük veri/model ağırlıkları
SHA-256 manifestli ayrı artifact arşivindedir. Git clone tek başına yeterli değildir.
Yerel arşiv ve kullanıcının Documents/NK_Arşiv kopyası aynı C: diskindedir;
disk arızasına dayanıklı bağımsız yedek NOT_CONFIRMED.
[Checkpoint kanıt kayıtları](experiments/PRE-HYBRID-20261011/) doğrulama kapsamını gösterir.

Eski C1-07 teslimi ve negatif araştırma sonucu korunur;
[tarihsel giriş eşlemesi](experiments/PRE-HYBRID-20261011/historical-snapshots.json)
hash zincirini açıkça taşır. Güncel çalışma bu README ve START_HERE'dan başlar.

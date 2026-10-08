# Core faz roadmap

Hedef v1.0.0 · Belge r12 · Durum AKTİF; C1-01 COMPLETE / T-C00 PASS, C1-02 COMPLETE / T-C07 PASS, C1-03 COMPLETE / T-C01 ve T-C02 PASS / ACCEPTED; C1-04 COMPLETE / T-C03 PASS / doğrudan IK NO-GO; C1-05 STAGE_1_COMPLETE / T-C04 NOT_RUN; Aşama 2 onayı bekliyor · Etkin emek 100–160 saat tahmini

## Amaç

Geçerli araştırma, ablation ve H2 kararı; olumlu sonuç zorunlu değil. Ayrıntılı tasarım: [faz raporu](../raporlar/02_Core_v1_0_r1.md).

## Giriş şartı

G0 geçti; robot/veri/baseline manifestleri. Kapsam ve sayısal eşikler deneye başlamadan dondurulur.

## Uygulama sırası

| Sıra | Görev | Bağımlılık | Emek |
|---|---|---|---|
| 1 | [C1-01 Harici baseline entegrasyonu](../tasks/C1-01.md) | F0-06 | 14–20 saat |
| 2 | [C1-02 Durumla şartlandırılmış veri çiftleri](../tasks/C1-02.md) | F0-06 | 14–24 saat |
| 3 | [C1-03 Diferansiyellenebilir FK](../tasks/C1-03.md) | F0-06 | 10–16 saat |
| 4 | [C1-04 Neural baseline modelleri](../tasks/C1-04.md) | C1-02, C1-03 | 14–24 saat |
| 5 | [C1-05 Physics-aware model ve varyantlar](../tasks/C1-05.md) | C1-04 | 22–34 saat |
| 6 | [C1-06 Ablasyon ve bağımsız nihai değerlendirme](../tasks/C1-06.md) | C1-01, C1-05 | 18–28 saat |
| 7 | [C1-07 Model kartı ve Core kapanışı](../tasks/C1-07.md) | C1-06 | 8–14 saat |

## Görev bazında kabul ve değerlendirme

### C1-01 Harici baseline entegrasyonu

**Gereksinim:** REQ-C01. **Test:** T-C00.

**Çıktı:** KDL, TRAC-IK, pick_ik adaptörleri ve sonuçları.

**Kabul:** Aynı sorgu, kaynak tavanı, tolerans ve bütçede tüm zorunlu baseline kayıtları var; iç eşik farkları belgeli.

**Başarısızlıkta:** 20 saatte engel sürerse DLS ile araştırmaya devam et; eksik kıyasla tam v1 veya üstünlük iddiası yayımlama.

**Kanıt:** [T-C00 kabul raporu](../../experiments/C1-01/RUN-20260928-T-C00-acceptance.md), `udp-v2` gate/lock/özetler. Beş yöntem × 120.000 = 600.000 kayıt, offline verify PASS; raw LOCAL_ONLY, uzak arşiv NOT_CONFIRMED. Etkin insan emeği NOT_MEASURED.

### C1-02 Durumla şartlandırılmış veri çiftleri

**Gereksinim:** REQ-C03. **Test:** T-C07.

**Çıktı:** q_current, pair_mode ve teacher metadata içeren veri sürümü.

**Kabul:** Yerel/geniş başlangıç dağılımları tanımlı; q_target doğrudan girdiye sızmıyor; kök çift grupları splitler arasında ayrık.

**Başarısızlıkta:** Öğretmen yalnız kolay örnekleri seçiyorsa seçim yanlılığını kaydet; test sorgularını öğretmen başarısına göre filtreleme.

**Kanıt:** [Aşama 1 sözleşmesi](../../experiments/C1-02/STAGE1_REVIEW.md), [Aşama 2 kabul raporu](../../experiments/C1-02/RUN-20260929-T-C07-acceptance.md) ve [acceptance.json](../../experiments/C1-02/acceptance.json). 24.000 çift, 34 shard, T-C07 ve tekrar üretim PASS; 2.281 wide öğretmen etiketi eksikliği korundu. REQ-C03 / T-C07 PASS / ACCEPTED.

### C1-03 Diferansiyellenebilir FK

**Gereksinim:** REQ-C02. **Test:** T-C01, T-C02.

**Çıktı:** Torch FK; referans ve gradyan sonuçları.

**Kabul:** Float64 eşliği G0 eşiklerinde; float32 ≤1e−5 m ve matris farkı ≤1e−5; 20 q için gradcheck atol 1e−5/rtol1e−3.

**Başarısızlıkta:** Autograd kopmasını veya frame farkını düzelt; yanlış FK ile neural deney başlatma.

**Kanıt:** [Nihai kabul raporu](../../experiments/C1-03/stage2/RUN_REPORT.md), [karar](../../experiments/C1-03/stage2/acceptance.json), [SHA](../../experiments/C1-03/stage2/SHA256SUMS). COMPLETE / T-C01 ve T-C02 PASS / ACCEPTED. İki koşuda 1086 q/dtype, 32 gradient q, 110 test ve 277 regresyon PASS. C1-04 girdileri hazır; NOT_STARTED.

### C1-04 Neural baseline modelleri

**Gereksinim:** REQ-C03. **Test:** T-C03.

**Çıktı:** Pose-only ve conditioned MLP checkpoint/config dosyaları.

**Kabul:** Küçük kontrollü veri öğrenme kontrolü geçer; aynı split ve bütçede modeller koşar; gerçek poz hatası raporlanır.

**Başarısızlıkta:** Loss düşüp FK düzelmiyorsa veri/ölçek/gradyanı tanıla; mimariyi rastgele büyütme.

**Kanıt:** [Aşama 1 tasarım incelemesi](../../experiments/C1-04/STAGE1_REVIEW.md), [Aşama 2 çalışma kaydı](../../experiments/C1-04/stage2/RUN_REPORT.md), [kapanış kararı](../../experiments/C1-04/stage2/acceptance.json). T-C03 PASS ve E-C01 iki model × üç eşli seed tamam; bağımsız FK ölçümünde her koşu Profil A 0/3.600. Görev deneysel baseline olarak COMPLETE; doğrudan IK **NO-GO**. Sıradaki C1-05 E-C03, sonuç iyileşmesini henüz kanıtlamadı.

### C1-05 Physics-aware model ve varyantlar

**Gereksinim:** REQ-C03, REQ-C04. **Test:** T-C04.

**Çıktı:** E-C03–08 seçili deney konfigürasyonları ve kayıtları.

**Kabul:** FK/limit etkisi ayrı; q/delta ve quaternion/6D karışmıyor; her ana karşılaştırma 3 seed; arama bütçesi kayıtlı.

**Başarısızlıkta:** Tekillik veya curriculum zarar verirse çıkar ve olumsuz etkiyi koru; testle hyperparameter seçme.

**Kanıt:** [Aşama 1 incelemesi](../../experiments/C1-05/STAGE1_REVIEW.md), [çalışma kaydı](../../experiments/C1-05/RUN_REPORT.md). STAGE_1_COMPLETE; T-C04 NOT_RUN. Yeni eğitim için açık onay bekleniyor.

### C1-06 Ablasyon ve bağımsız nihai değerlendirme

**Gereksinim:** REQ-C04, REQ-C05. **Test:** T-C05.

**Çıktı:** H2 sonucu; paired bootstrap; failure analysis; raw results.

**Kabul:** 10.000 ana sorgu ve zor alt kümeler raporlu; olumlu/olumsuz/belirsiz H2 kararı gerekçeli; test sızıntısı yok.

**Başarısızlıkta:** Sonuç zayıfsa hipotezi reddet veya belirsiz de; yeni deneye yeni config ve gerekirse yeni final test ata.

**Kanıt:** `experiments/C1-06/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### C1-07 Model kartı ve Core kapanışı

**Gereksinim:** REQ-C06. **Test:** T-C06.

**Çıktı:** MODEL_CARD; G1 kararı; Hybrid devir manifesti.

**Kabul:** Robot/TCP, normalizasyon, ağırlık, eğitim dağılımı ve kısıtlar tam; temiz ortamda çıkarım ve değerlendirme örneği çalışır.

**Başarısızlıkta:** Eksik hash veya veri kimliği varsa model taşınmaz; yalnız .pt dosyasını teslimat sayma.

**Kanıt:** `experiments/C1-07/`. Mevcut sonuç: ÖLÇÜLMEDİ.

## G1 kararı

Geçerli araştırma, ablation ve H2 kararı; olumlu sonuç zorunlu değil. Teknik kusurlar ile araştırma hipotezinin reddi farklı kararlardır. Faz raporundaki zorunlu testler ve destek profili karşılanmadan sürüm etiketi verilmez.

- [ ] Gereksinimler ve testler kapatıldı.
- [ ] Ham kayıtlar ve tekrar üretim tarifi mevcut.
- [ ] Hipotezler desteklendi, reddedildi veya belirsiz olarak yazıldı.
- [ ] Açık kısıtlar ve sonraki faz girdileri listelendi.

## Her çalışma oturumunda

Görev durumunu güncelle; ne yaptığını, hangi testin neden çalıştırıldığını ve ne öğrendiğini yaz. Bir sonraki oturumun tek ana işini belirle. [Faz devir şablonu](../templates/PHASE_HANDOFF.md) ve [deney şablonu](../templates/RUN_REPORT.md) kullanılacaktır.

## 27 Eylül 2026 · C1-01 güncel uygulama kaydı

C1-01 Aşama 2 sürüyor. Kullanıcı ana test hazırlığını onayladı; UDPv4 ortak koşu ve mutlak deadline düzeltmesi sonrası Linux doğrulaması bekleniyor. T-C00 henüz kabul edilmedi. Sonraki Core görevleri bu çalışma kapsamında başlatılmadı. [Kanıt ve sıra](../../experiments/C1-01/RUN-20260927-main-preparation.md).

## 28 Eylül 2026 · C1-01 kabulü

C1-01 / REQ-C01 / T-C00 **PASS / ACCEPTED**. `udp-v2` prepare/smoke/pilot ve 600.000 tam ölçüm ile offline verify PASS; beş zorunlu varyant aynı dondurulmuş koşullarda. Eski eksik ölçümler birleştirilmedi. [Kabul raporu](../../experiments/C1-01/RUN-20260928-T-C00-acceptance.md), [gate](../../experiments/C1-01/udp-v2/verify/gate.json). Sonraki iş bağımlılık sırasıyla C1-02; C1-03 ve diğer Core görevleri henüz başlamadı. G1/Core faz kabulü ayrı ve açık.

## 28 Eylül 2026 · C1-02 Aşama 1

F0-04 köklerinden 50/50 local/wide çift sözleşmesi, öğretmen pilot bütçesi, sızıntı matrisi ve 43 dosyalı SHA manifesti donduruldu. [Çalışma kaydı](../../experiments/C1-02/RUN_REPORT.md). C1-02 **IN_PROGRESS / STAGE_1_COMPLETE**; üretim ve T-C07 **NOT_RUN**. Aşama 2 açık kullanıcı onayı bekler. C1-04 için C1-02 yanında C1-03 kabulü de gerekir.

## 29 Eylül 2026 · C1-02 kabulü

Kullanıcı onayı sonrası 90 wide pilot, iki temiz 24.000 satırlık üretim, T-C07 ve tekrar üretim denetimi tamamlandı. Canonical veri hash'i ve 34/34 shard eş; leakage/benchmark overlap ve train-only normalizasyon PASS. 2.281 wide etiketi başarısızlığı ölçüldü ve satırlar korundu. C1-02 **COMPLETE / T-C07 PASS / ACCEPTED**; [kabul raporu](../../experiments/C1-02/RUN-20260929-T-C07-acceptance.md). Sıradaki iş C1-03, **NOT_STARTED**; C1-04 için C1-03 kabulü de gerekir. G1 kararı açık.

## 29 Eylül 2026 · C1-03 Aşama 1

G0 hashleri, robot/frame/autograd sözleşmesi, Torch 2.10.0+cpu overlay ve sonuçlardan bağımsız test örnekleri donduruldu. [İnceleme](../../experiments/C1-03/STAGE1_REVIEW.md), [çalışma kaydı](../../experiments/C1-03/RUN_REPORT.md). **IN_PROGRESS / STAGE_1_COMPLETE**, T-C01/T-C02 **NOT_RUN**. Aşama 2 yalnız açık onaydan sonra; C1-03 kabulü ve C1-02 girdisi olmadan C1-04 açılmaz. Bu görevde C1-04 başlatılmaz.

## 29 Eylül 2026 · C1-03 Aşama 2 ara kaydı

Açık kullanıcı onayı ve Stage1 hash denetimi sonrası Torch FK uygulandı. Yerel T-C01/T-C02, 110 unit/negatif/arayüz testi ve 277 Foundations regresyonu geçti. ADR-012 ve protokol r2 ortam/harness düzeltmelerini kaydeder; eşikler ve örnekler değişmedi. **IN_PROGRESS / CLEAN_REPRODUCTION_PENDING**; ikinci temiz ortam ve nihai kabul audit bekleniyor. [Çalışma kaydı](../../experiments/C1-03/stage2/RUN_REPORT.md). C1-04 başlamadı.

## 29 Eylül 2026 · C1-03 nihai kabul

**REQ-C02 / T-C01 / T-C02 PASS / ACCEPTED; C1-03 COMPLETE.**
Standart Torch fixed/revolute kernel, dondurulmuş KUKA robot/TCP/frame ve
autograd sözleşmesini iki gerçek koşuda geçti. Her ortamda 1086 q/dtype,
32 iç konfigürasyon/2880 türev, 32 gradcheck/Jacobian, sensitivity/batch/edge,
110 test (21 C1-02 arayüz dahil), 24 öldürülen gerçek source mutant ve
277 Foundations regresyonu PASS; skip0. Yeni checkout/ortamda 15/15 komut
PASS; 2317 satır/2695396 bayt raw sonuçlar iki koşuda bayt düzeyinde aynı.
Eşikler, örnekler, Foundations ve C1-02 girdileri değiştirilmedi.

[Nihai çalışma kaydı](../../experiments/C1-03/stage2/RUN_REPORT.md),
[kabul kararı](../../experiments/C1-03/stage2/acceptance.json),
[komutlar](../../experiments/C1-03/stage2/COMMANDS.md) ve
[kanıt manifesti](../../experiments/C1-03/stage2/evidence-manifest.json).
Uygulama commitleri `7d9e282` ve `4022e2359306a780422c94f25252bd2eaa90ed8f`.
Kapanış commit kimliği Git geçmişinden okunur. Önceki ara durum kayıtları
tarihseldir; güncel karar bu kabul kaydıdır. Linux/CUDA/fiziksel robot NOT_RUN;
performans/etkin emek NOT_MEASURED. C1-04 girdileri hazır, **NOT_STARTED**;
neural eğitim, G1 kararı ve v1.0.0 etiketi bu kapsamda oluşturulmadı.

## 2 Ekim 2026 · C1-04 Aşama 1

C1-02/C1-03 kabulü ve yerel C1-02 shard hashleri üzerinden E-C01, T-C03, giriş/çıkış, seed/bütçe ve validation protokolü donduruldu. [İnceleme](../../experiments/C1-04/STAGE1_REVIEW.md) ve [RUN_REPORT](../../experiments/C1-04/RUN_REPORT.md). C1-04 **IN_PROGRESS / STAGE_1_COMPLETE**; T-C03/E-C01 **NOT_RUN**. Aşama 2 için açık onay beklenir; C1-05 ve G1 başlamadı.

## 3 Ekim 2026 · C1-04 Aşama 2 ve düşük başarı kapısı

Onay sonrası T-C03 PASS, E-C01 üç eşli seed ve 21.600 bağımsız FK validation sonucu tamamlandı. Altı model koşusunda Profil A ayrı ayrı 0/3.600 olduğundan [kapanış kararı](../../experiments/C1-04/stage2/acceptance.json) doğrudan IK için **NO-GO**'dur. Conditioned FK medyanı daha düşük ama operasyonel eşik altında değildir. Temiz checkout/taze ortamda altı checkpointten 60 sabit çıkarım ve FK farkı sıfırdır; ağırlıklar `LOCAL_ONLY`. [C1-05 devri](../../experiments/C1-04/stage2/NEXT_MODEL_DECISION.md) E-C03 supervised + FK kaybını kontrollü yeni deney olarak seçer, E-C04 limit etkisini ayrı tutar. C1-05 henüz **NOT_STARTED**; düşük başarı eşik değiştirilerek kabul edilmez. C1-06 nihai test ve G1 açık.

## 8 Ekim 2026 · C1-05 Aşama 1

C1-05 **IN_PROGRESS / STAGE_1_COMPLETE; T-C04 NOT_RUN**. Girdi/checkpoint SHA
ve erişim doğrulandı; 129 regresyon ve tam T-C01/T-C02 PASS. Aynı conditioned
modelde FK, ayrı limit ve koşullu tanh karşılaştırmaları donduruldu; yeni opt-in
FK domain/pilot/eğitim NOT_RUN. [RUN_REPORT](../../experiments/C1-05/RUN_REPORT.md).
Aşama 2 açık onay bekler. C1-06 test ve 10000 benchmark mühürlü, G1 açık kalır.

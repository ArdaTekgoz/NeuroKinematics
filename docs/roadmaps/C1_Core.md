# Core faz roadmap

Hedef v1.0.0 · Belge r2 · Durum AKTİF; C1-01 IN_PROGRESS / STAGE_1_COMPLETE · Etkin emek 100–160 saat

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

**Kanıt:** `experiments/C1-01/`. Aşama 1 config ve platform kararı donduruldu; solver uygulaması ve T-C00 NOT_RUN, performans NOT_MEASURED.

### C1-02 Durumla şartlandırılmış veri çiftleri

**Gereksinim:** REQ-C03. **Test:** T-C07.

**Çıktı:** q_current, pair_mode ve teacher metadata içeren veri sürümü.

**Kabul:** Yerel/geniş başlangıç dağılımları tanımlı; q_target doğrudan girdiye sızmıyor; kök çift grupları splitler arasında ayrık.

**Başarısızlıkta:** Öğretmen yalnız kolay örnekleri seçiyorsa seçim yanlılığını kaydet; test sorgularını öğretmen başarısına göre filtreleme.

**Kanıt:** `experiments/C1-02/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### C1-03 Diferansiyellenebilir FK

**Gereksinim:** REQ-C02. **Test:** T-C01, T-C02.

**Çıktı:** Torch FK; referans ve gradyan sonuçları.

**Kabul:** Float64 eşliği G0 eşiklerinde; float32 ≤1e−5 m ve matris farkı ≤1e−5; 20 q için gradcheck atol 1e−5/rtol1e−3.

**Başarısızlıkta:** Autograd kopmasını veya frame farkını düzelt; yanlış FK ile neural deney başlatma.

**Kanıt:** `experiments/C1-03/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### C1-04 Neural baseline modelleri

**Gereksinim:** REQ-C03. **Test:** T-C03.

**Çıktı:** Pose-only ve conditioned MLP checkpoint/config dosyaları.

**Kabul:** Küçük kontrollü veri öğrenme kontrolü geçer; aynı split ve bütçede modeller koşar; gerçek poz hatası raporlanır.

**Başarısızlıkta:** Loss düşüp FK düzelmiyorsa veri/ölçek/gradyanı tanıla; mimariyi rastgele büyütme.

**Kanıt:** `experiments/C1-04/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### C1-05 Physics-aware model ve varyantlar

**Gereksinim:** REQ-C03, REQ-C04. **Test:** T-C04.

**Çıktı:** E-C03–08 seçili deney konfigürasyonları ve kayıtları.

**Kabul:** FK/limit etkisi ayrı; q/delta ve quaternion/6D karışmıyor; her ana karşılaştırma 3 seed; arama bütçesi kayıtlı.

**Başarısızlıkta:** Tekillik veya curriculum zarar verirse çıkar ve olumsuz etkiyi koru; testle hyperparameter seçme.

**Kanıt:** `experiments/C1-05/`. Mevcut sonuç: ÖLÇÜLMEDİ.

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

# Studio — generic robot, yörünge ve planlama deneyimi

Roadmap r3 · 11 Ekim 2026 · Durum PLANLANDI / NOT_STARTED · Tahmin: 80–130 etkin saat.
Yazılım hedef sürümü belge revizyonundan ayrıdır; henüz release yapılmadı.

Hedef yazılım v3.0.0. [Özgün Studio raporu](../raporlar/04_Studio_v3_0_r1.md)
korunur. Giriş G2 + G-MR. S3-01 H3 değerlendirmesi MR altyapı kanıtını kullanır;
ikinci bir onboarding sistemi yazılmaz.

S3-02 sonrası **TR-01–04 ve MP-01–05** tamamlanır; S3-03 sonra açılır.
Bu sayfadaki 80–130 saat yalnız özgün Studio görevleridir. TR/MP ayrıca
52–82 ve 66–112 saat: genişletilmiş Studio toplamı **198–324 etkin saat**.
Fiziksel kontrol bu sürümde yoktur; STUDIO-EXT ileride aynı GUIyi genişletir.

## Uygulama sırası

| Sıra | Görev | Ön koşul | Saat |
|---|---|---|---|
| 1 | [S3-01 — İkinci robot H3 değerlendirmesi](../tasks/S3-01.md) | MR-06 | 18–28 |
| 2 | [S3-02 — GUI kabuğu ve generic servis bağlantısı](../tasks/S3-02.md) | S3-01 | 18–28 |
| 3 | [S3-03 — Analiz, birleşik playback ve uzun görevler](../tasks/S3-03.md) | S3-02, MP-05 | 20–34 |
| 4 | [S3-04 — Proje paketi ve temiz dağıtım](../tasks/S3-04.md) | S3-03 | 16–28 |
| 5 | [S3-05 — Kullanıcı testi ve genişletilmiş G3](../tasks/S3-05.md) | S3-04 | 8–12 |

## Görev bazında kabul

### S3-01 — İkinci robot H3 değerlendirmesi

Özgün H3 taşınabilirlik deneyini MR kanıtlarını kullanarak değerlendirmek.

**Gereksinim:** REQ-S01. **Test:** T-S01.

**Çıktı:** MR robot B manifest/data/model sonuçlarını aynı commitli KUKA sonuçlarıyla bağla; onboarding altyapısını tekrar geliştirme. Config/varlık değişiklikleri ve generic çekirdek düzeltmelerini say; robot başına eğitim ile zero-shot kavramını ayır. Özgün Studio raporunun H3 protokolünü ve açık sınırlamaları değerlendir; kullanıcıya destek profili devret.

**Kabul:** T-S01: aynı pipeline yeni varlık/config ile çalışır; solver-specific kod değişiklikleri sayılır; iki robot aynı committe testli. MR-PASS, H3 sonuç tablosunu otomatik doldurmaz; ölçülmüş kapsamla hipotez kararı mevcut.

**Başarısızlıkta:** Generic düzeltme gerekiyorsa MR regression tekrar edilir; yeni robotu destekli diye etiketleme.

**Kanıt:** `experiments/S3-01/`; PLANLANDI / NOT_RUN.

### S3-02 — GUI kabuğu ve generic servis bağlantısı

Robot seçimi, hedef ve 3B FK görünümünü ortak servise bağlamak.

**Gereksinim:** REQ-S02. **Test:** T-S02.

**Çıktı:** RobotModel/Registry/Onboarding/SolverResult API üzerinden ekran; GUI içinde ikinci FK/IK veya KUKA sabiti yok. Birim/frame/robot-model uyumsuzluğunu görünür hata yap; neural/geometric/numeric motor kimliğini göster. Windows CPU prototipinde 3B robot ve hazırlık durumu; trajectory/planning servisleri için arayüz noktaları.

**Kabul:** T-S02: yanlış kimlik/birim/frame doğru yakalanır; ortak kabul sonucu aynen görünür; hedef platform prototipi çalışır.

**Başarısızlıkta:** Tam hazır olmayan robot FULLY_READY görünmez; paketleme riskini destek profiline kaydet.

**Kanıt:** `experiments/S3-02/`; PLANLANDI / NOT_RUN.

### S3-03 — Analiz, birleşik playback ve uzun görevler

Generic onboarding, trajectory ve planner sonuçlarını yanıt veren arayüzde birleştirmek.

**Gereksinim:** REQ-S03, REQ-S04, REQ-SX01. **Test:** T-S03, T-S05, T-SX01.

**Çıktı:** Heatmap: verisiz voxel/başarısız çözüm ayrımı; robot/motor/scene/seed/validation kapsamı. Raw/simplified path, obstacle ve TCP trace, q/dq/ddq timeline; TR player API ile play/pause/seek. Uzun onboarding/training/planning worker; iptal≤2s görünür; geç cevap yeni robot/proje üstüne yazılmaz. Ekranda numerical-ready ile neural katkı/onboarding-ready ayrımı; bütün işler ortak servis üzerinden.

**Kabul:** T-S03/T-S05: özgün heatmap ve iptal koşulları PASS. T-SX01: robot/scene değişiminde eski path invalid; TR/MP örnekleri doğru kapsamla gösteriliyor.

**Başarısızlıkta:** UI görünümü validation otoritesi değil; ağır işi worker sürecine taşı, iptal yarışını test et.

**Kanıt:** `experiments/S3-03/`; PLANLANDI / NOT_RUN.

### S3-04 — Proje paketi ve temiz dağıtım

Çoklu robot, scene ve trajectory varlıklarını tekrar açılabilir pakete almak.

**Gereksinim:** REQ-S05, REQ-S06, REQ-SX02. **Test:** T-S04, T-S06, T-SX02.

**Çıktı:** Şemalı project/model/robot/scene/trajectory/seed/limit profili ve hashler; kaydet-aç eşliği. Windows CPU dev Python olmadan demo, dependency kilitleri/SBOM; unsupported hardware/runtime açık. Bozuk hash/revision/paket reddi; büyük model/data dış artifact konumu ve restore tarifi.

**Kabul:** T-S04/T-S06: temiz hedef kurulum ve roundtrip PASS. T-SX02: robot/scene/trajectory ilişkisi korunur; bozuk veya stale path sessiz valid olmaz.

**Başarısızlıkta:** Hedef OS denenmediyse dağıtım tamamlandı değil; kapsam değişimi ADR gerektirir.

**Kanıt:** `experiments/S3-04/`; PLANLANDI / NOT_RUN.

### S3-05 — Kullanıcı testi ve genişletilmiş G3

Kullanıcıların model→IK→plan→trajectory→simülasyon işini doğru yorumlayabildiğini doğrulamak.

**Gereksinim:** REQ-S06, REQ-SX03. **Test:** T-S07, T-SX03.

**Çıktı:** En az üç hedef kullanıcı; özgün ≥%80 görev tamamlama hedefi; hazır olma ve hata/UNKNOWN yorumları dahil. Generic robot seçimi, hedef çözme, engelli path, timeline, export ve iptal senaryoları. Kritik yanıltıcı readiness/safety etiketlerini gider; G3 devri G-MR/G-TR/G-MP kanıtlarını bağlasın.

**Kabul:** T-S07: özgün kullanıcı hedefi ve sorun değerlendirmesi mevcut, kritik yanıltıcı davranış yok. T-SX03: birleşik workflow testli; G3 yalnız ön kapılar ve temiz dağıtım ile kapanır; fiziksel execution dahil değil.

**Başarısızlıkta:** Küçük örneklemden pazar sonucu çıkarma; hedef altında workflow düzelt ve yeniden değerlendir.

**Kanıt:** `experiments/S3-05/`; PLANLANDI / NOT_RUN.

## Faz kapısı

**G3:** S3-01–05, G-MR/G-TR/G-MP, temiz Windows CPU dağıtımı, roundtrip,
yanıltıcı durumların düzeltilmesi ve ≥3 kullanıcı/≥%80 görev hedefi değerlendirilmiş.
Desteklenen kapsam yayımlanır; hazır olmayan motor veya robot gizlenerek başarı
verilmez. Fiziksel execution G3 ürünü sayılmaz.

Tüm işler için gereksinim → değişiklik → test → ham kanıt bağı tutulur.
Uygulamada protokol/seed/kimlikler sonuç görülmeden dondurulur; başarısız kayıtlar
korunur. Yukarıdaki süreler tahmindir; GPU bekleme, donanım temini ve kullanıcı
testi takvimini kapsamaz. İlk iki görevden sonra yalnız tahminler güncellenir.

[Ana roadmap](MASTER_ROADMAP.md) · [Kaynak eşlemesi](SOURCE_MAPPING_r3.md)
· [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).

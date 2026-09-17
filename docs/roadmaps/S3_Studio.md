# Studio faz roadmap

Hedef v3.0.0 · Belge r1 · Durum PLANLANDI · Etkin emek 80–130 saat

## Amaç

İkinci robot, kullanışlı arayüz ve temiz sistemde test edilmiş paket. Ayrıntılı tasarım: [faz raporu](../raporlar/04_Studio_v3_0_r1.md).

## Giriş şartı

G2 kapandı; SolverRequest/Result sözleşmesi. Kapsam ve sayısal eşikler deneye başlamadan dondurulur.

## Uygulama sırası

| Sıra | Görev | Bağımlılık | Emek |
|---|---|---|---|
| 1 | [S3-01 İkinci robot ve H3 deneyi](../tasks/S3-01.md) | H2-06 | 18–28 saat |
| 2 | [S3-02 GUI kabuğu ve servis bağlantısı](../tasks/S3-02.md) | S3-01 | 18–28 saat |
| 3 | [S3-03 Analiz ve uzun görev yönetimi](../tasks/S3-03.md) | S3-02 | 20–34 saat |
| 4 | [S3-04 Dışa aktarım ve dağıtım paketi](../tasks/S3-04.md) | S3-03 | 16–28 saat |
| 5 | [S3-05 Kullanıcı testi ve Studio kapanışı](../tasks/S3-05.md) | S3-04 | 8–12 saat |

## Görev bazında kabul ve değerlendirme

### S3-01 İkinci robot ve H3 deneyi

**Gereksinim:** REQ-S01. **Test:** T-S01.

**Çıktı:** Robot B manifesti, dataset, model ve H3 karşılaştırması.

**Kabul:** Aynı pipeline yeni varlık/config ile çalışır; solver-specific değişiklikler sayılır; iki robot aynı committe testli.

**Başarısızlıkta:** Genel düzeltme gerekirse iki robot regresyonu; per-robot eğitim sonucunu zero-shot olarak adlandırma.

**Kanıt:** `experiments/S3-01/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### S3-02 GUI kabuğu ve servis bağlantısı

**Gereksinim:** REQ-S02. **Test:** T-S02.

**Çıktı:** Robot/hedef/sonuç ekranları; 3B FK görünümü.

**Kabul:** Birim, frame, model uyumsuzluğu doğru yakalanır; GUI ortak servisi kullanır; hedef platform prototipi çalışır.

**Başarısızlıkta:** Paketleme riski varsa tek robot demo ile tanıla; alternatif platformu açık destek profili olarak kaydet.

**Kanıt:** `experiments/S3-02/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### S3-03 Analiz ve uzun görev yönetimi

**Gereksinim:** REQ-S03, REQ-S04. **Test:** T-S03, T-S05.

**Çıktı:** Heatmap, yörünge, worker ve iptal akışı.

**Kabul:** Verisiz voxel ile başarısız çözüm ayrılır; iş sürerken UI yanıt verir; iptal durumu ≤2s içinde görünür.

**Başarısızlıkta:** Ağır render/veri işini worker sürecine ayır; iptal sonucu yeni projenin üzerine yazılmaz.

**Kanıt:** `experiments/S3-03/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### S3-04 Dışa aktarım ve dağıtım paketi

**Gereksinim:** REQ-S05, REQ-S06. **Test:** T-S04, T-S06.

**Çıktı:** Şemalı proje/model paketi; Windows CPU aday dağıtımı; SBOM.

**Kabul:** Aç-kaydet-aç eşliği; bozuk pakette açık hata; temiz hedef sistemde Python dev kurulumu olmadan demo.

**Başarısızlıkta:** Hedef OS çalışmıyorsa tamamlandı deme; desteklenen platformu ADR ile sınırla ve açık engeli yaz.

**Kanıt:** `experiments/S3-04/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### S3-05 Kullanıcı testi ve Studio kapanışı

**Gereksinim:** REQ-S06. **Test:** T-S07.

**Çıktı:** Kullanıcı bulguları, demo, kılavuz ve G3 devir raporu.

**Kabul:** En az3 hedef kullanıcı; kritik görevlerde toplam yüzde 80 tamamlama hedefi değerlendirilmiş; kritik yanıltıcı davranışlar giderilmiş.

**Başarısızlıkta:** Hedef tutmazsa iş akışını sadeleştir ve yeniden değerlendir; küçük örneklemden pazar sonucu çıkarma.

**Kanıt:** `experiments/S3-05/`. Mevcut sonuç: ÖLÇÜLMEDİ.

## G3 kararı

İkinci robot, kullanışlı arayüz ve temiz sistemde test edilmiş paket. Teknik kusurlar ile araştırma hipotezinin reddi farklı kararlardır. Faz raporundaki zorunlu testler ve destek profili karşılanmadan sürüm etiketi verilmez.

- [ ] Gereksinimler ve testler kapatıldı.
- [ ] Ham kayıtlar ve tekrar üretim tarifi mevcut.
- [ ] Hipotezler desteklendi, reddedildi veya belirsiz olarak yazıldı.
- [ ] Açık kısıtlar ve sonraki faz girdileri listelendi.

## Her çalışma oturumunda

Görev durumunu güncelle; ne yaptığını, hangi testin neden çalıştırıldığını ve ne öğrendiğini yaz. Bir sonraki oturumun tek ana işini belirle. [Faz devir şablonu](../templates/PHASE_HANDOFF.md) ve [deney şablonu](../templates/RUN_REPORT.md) kullanılacaktır.

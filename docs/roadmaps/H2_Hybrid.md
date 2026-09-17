# Hybrid faz roadmap

Hedef v2.0.0 · Belge r1 · Durum PLANLANDI · Etkin emek 90–140 saat

## Amaç

Doğru kabul/hata davranışı, H1 kararı, yörünge ve ONNX eşliği. Ayrıntılı tasarım: [faz raporu](../raporlar/03_Hybrid_v2_0_r1.md).

## Giriş şartı

G1 kapandı; model kartı ve referans servisler. Kapsam ve sayısal eşikler deneye başlamadan dondurulur.

## Uygulama sırası

| Sıra | Görev | Bağımlılık | Emek |
|---|---|---|---|
| 1 | [H2-01 Kabul ve hata politikası](../tasks/H2-01.md) | C1-07 | 10–16 saat |
| 2 | [H2-02 Bütçeli hibrit çözücü](../tasks/H2-02.md) | H2-01 | 20–30 saat |
| 3 | [H2-03 Öğrenilmiş başlangıç H1 deneyi](../tasks/H2-03.md) | H2-02 | 16–24 saat |
| 4 | [H2-04 Zamanlı yörünge ve geometri kontrolü](../tasks/H2-04.md) | H2-02 | 24–38 saat |
| 5 | [H2-05 ONNX aktarımı ve runtime eşliği](../tasks/H2-05.md) | H2-02 | 12–20 saat |
| 6 | [H2-06 Hybrid kapanışı ve servis paketi](../tasks/H2-06.md) | H2-03, H2-04, H2-05 | 8–12 saat |

## Görev bazında kabul ve değerlendirme

### H2-01 Kabul ve hata politikası

**Gereksinim:** REQ-H01. **Test:** T-H01.

**Çıktı:** SolverRequest/Result şeması; failure taxonomy.

**Kabul:** Bozuk giriş, yanlış model, limit, collision NOT_CHECKED ve geç sonuç için doğru statü; yanlış başarı yok.

**Başarısızlıkta:** Belirsiz durumda UNKNOWN veya hata döndür; çözüm başarısızlığını erişilemezlik sayma.

**Kanıt:** `experiments/H2-01/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### H2-02 Bütçeli hibrit çözücü

**Gereksinim:** REQ-H02. **Test:** T-H02.

**Çıktı:** Neural seed, refinement ve bir yedek restart servisi.

**Kabul:** Ön işlem ve doğrulama toplam bütçeye dahil; her deneme kalan bütçeyi kullanır; her aday son denetimden geçer.

**Başarısızlıkta:** NaN neural çıkışta q_current sayısal seed; bütçe biterse açık hata; clamping sonrası yeniden doğrula.

**Kanıt:** `experiments/H2-02/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### H2-03 Öğrenilmiş başlangıç H1 deneyi

**Gereksinim:** REQ-H03. **Test:** T-H03.

**Çıktı:** B-H01–04 sonuçları; latency/iteration ve CI raporu.

**Kabul:** Aynı algoritma ve bütçede adil kıyas; H1 yüzde 20 P95 ve en fazla 1 yüzde puanı başarı kaybı hedefi açık değerlendirildi.

**Başarısızlıkta:** Kazanç yoksa sayısal motor varsayılan; iki hedefli düzeltme turundan sonra araştırma birikimine taşı.

**Kanıt:** `experiments/H2-03/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### H2-04 Zamanlı yörünge ve geometri kontrolü

**Gereksinim:** REQ-H04. **Test:** T-H04, T-H05.

**Çıktı:** 30 yörünge; türev raporu; collision backend ve örnekleri.

**Kabul:** Her yörüngede ≥200 zamanlı örnek; rollout önceki kabulü kullanır; ≥100 etiketli geometri testi; örneklenmiş yol kapsamı açık.

**Başarısızlıkta:** Eksik limit deneysel etiketlenir; bilinmeyen mesh/geometri UNKNOWN; sonlu örnekten sürekli garanti çıkarma.

**Kanıt:** `experiments/H2-04/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### H2-05 ONNX aktarımı ve runtime eşliği

**Gereksinim:** REQ-H05. **Test:** T-H06.

**Çıktı:** FP32 ONNX model paketi ve CPU ölçümleri.

**Kabul:** ≥1.000 girdide eklem farkı ≤1e−5 rad, FK farkı ≤0.1 mm/0.01 derece hedefi; her çıktıda bağımsız final denetim.

**Başarısızlıkta:** Fark varsa export/opset/ön işlem tanıla; desteklenmeyen runtime olarak bırak, eşikleri sonucu saklamak için değiştirme.

**Kanıt:** `experiments/H2-05/`. Mevcut sonuç: ÖLÇÜLMEDİ.

### H2-06 Hybrid kapanışı ve servis paketi

**Gereksinim:** REQ-H06. **Test:** T-H07.

**Çıktı:** G2 kararı; desteklenen motor ve Studio devir paketi.

**Kabul:** Kritik servis/kurulum testleri geçer; H1 sonucu varsayılan motor seçimine yansır; ONNX ve doğrulama kapsamı kayıtlı.

**Başarısızlıkta:** ONNX engelliyse aday sürüm; matematiksel hata varsa G2 açık kalır; negatif H1 bilimsel sonuçtur.

**Kanıt:** `experiments/H2-06/`. Mevcut sonuç: ÖLÇÜLMEDİ.

## G2 kararı

Doğru kabul/hata davranışı, H1 kararı, yörünge ve ONNX eşliği. Teknik kusurlar ile araştırma hipotezinin reddi farklı kararlardır. Faz raporundaki zorunlu testler ve destek profili karşılanmadan sürüm etiketi verilmez.

- [ ] Gereksinimler ve testler kapatıldı.
- [ ] Ham kayıtlar ve tekrar üretim tarifi mevcut.
- [ ] Hipotezler desteklendi, reddedildi veya belirsiz olarak yazıldı.
- [ ] Açık kısıtlar ve sonraki faz girdileri listelendi.

## Her çalışma oturumunda

Görev durumunu güncelle; ne yaptığını, hangi testin neden çalıştırıldığını ve ne öğrendiğini yaz. Bir sonraki oturumun tek ana işini belirle. [Faz devir şablonu](../templates/PHASE_HANDOFF.md) ve [deney şablonu](../templates/RUN_REPORT.md) kullanılacaktır.

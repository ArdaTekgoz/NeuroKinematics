# PE-05B — Sim-to-real doğruluk ve tekrarlanabilirlik

Durum: PLANLANDI / NOT_STARTED · Test: NOT_RUN · Belge r1 · Roadmap r3 · 11 Ekim 2026.
Sorumlu: Arda Tekgöz. Başlangıç/bitiş ve gerçek süre: KAYDEDİLMEDİ.
Etkin emek: PE-07A platform/ekip/bench envanteri sonrası tahminlenecek; takvim sözü yok.

## Amaç ve gereksinim

Nominal/kalibre modelin fiziksel trajectory ile farkını ölçmek.

Gereksinimler: REQ-PE05B. Kaynak: PE PDF s.30–33.
[Kaynak eşlemesi](../roadmaps/SOURCE_MAPPING_r3.md) · [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).

## Girdiler ve bağımlılıklar

Ön koşullar: [PE-04](PE-04.md). Önceki kabul raporu, hashli robot/model/scene/ortam girdileri
ve açık sınırlamalar okunmadan uygulama açılmaz. Negatif bilimsel sonuç ile
teknik kusur ayrılır; özel giriş kapıları ana roadmapte tanımlıdır.

## Yapılacak iş ve çıktılar

1. Aynı hareketin simülasyon/fiziksel joint/pose kayıtları; bağımsız pose ölçümü ve ölçüm belirsizliği.
2. Joint/position/orientation error ve repeatability; farklı workspace/yük/tool profilleri yalnız tanımlı kapsamda.
3. Calibration güncellemesi gerekirse yeni revision; dataset/model/twin/trajectory bağımlılıklarını yeniden doğrula.
4. Final veri ile kalibrasyon ayarlama yok; calibration-fit ve held-out ölçüm setlerini ayır.

## Test planı ve kabul

Test kimlikleri: T-PE05B.

- T-PE05B: ön kayıtlı fiziksel tolerans ve tekrar protokolü; sim-real farklar kapsam/belirsizlikle raporlu.
- Yeni kalibrasyonun regression ve stale-artifact politikası PASS; eski frozen KUKA kaynakları korunur.

Performans hedefi henüz sayısal olarak tanımlanmamışsa görev başında
`protocol.json` içinde iş yükü, eşik, gerekçe, seed, örnek sayısı ve stop kuralı
dondurulur. Eksik şartname kabul değildir; `BLOCKED_SPEC` olarak kaydedilir.
Eski F0/C1/H2 eşikleri bu yolla değiştirilmez. Komutlar uygulamada gerçek CLI
yollarıyla kaydedilecek; bu plan çalıştırılmış test kanıtı değildir.

İlgili önceki katmanların birim, kimlik ve uçtan uca testleri yeniden çalıştırılır;
değişmeyen eğitim kampanyası sırf doküman revizyonu için tekrar edilmez.

## Fonksiyon dışı şartlar ve kanıt

Deterministik config/seed, açık hata, finite/shape/unit/frame doğrulaması,
robot ve model uyumluluğu, zaman/bellek maliyeti ve bağımlılık kilidi kaydedilir.
Başarısız denemeler ve ham kayıtlar korunur. Büyük veri/model ağırlıkları normal
Git takibine alınmaz; manifest ve restore tarifi tutulur.

Planlanan kanıt: `experiments/PE-05B/`. Henüz mevcut/dolu olduğu iddia edilmez.
Teslimatta `RUN_REPORT.md`, `protocol.json`, girdi/çıktı SHA manifesti,
gerçek komut/ortam/seed/başlangıç-bitiş, ham log/JUnit/ölçüm ve
`acceptance.json` gerekir. Ürün testi için demo/benchmark artifactları yukarıdaki
kapsamdadır. [Çalışma şablonu](../templates/RUN_REPORT.md).

## Başarısızlık ve kapsam sınırı

Tolerans dışıysa physical support profili açık kalır; sadece simülasyon hatasını raporlayarak fiziksel fark saklanmaz.

## Kapanış

- [ ] Gereksinim → değişiklik → test → ham kanıt bağı tamamlandı.
- [ ] Zorunlu kabul kriterleri ölçüldü; başarısız/çalıştırılmayanlar açık.
- [ ] Regresyon, tekrar üretim, açık sınırlamalar ve hashli devir hazır.
- [ ] STATUS ve TRACEABILITY tarihli kayıtla güncellendi.

Sonraki doğrudan işler: [STUDIO-EXT](STUDIO-EXT.md). Karar: VERİLMEDİ. Uygulama: BAŞLAMADI.
Kapanışta plan silinmez; tarihli gerçekleşen sonuç aşağıya eklenir.

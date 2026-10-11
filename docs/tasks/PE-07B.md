# PE-07B — Yürütme kilitleri ve fiziksel kabul

Durum: PLANLANDI / NOT_STARTED · Test: NOT_RUN · Belge r1 · Roadmap r3 · 11 Ekim 2026.
Sorumlu: Arda Tekgöz. Başlangıç/bitiş ve gerçek süre: KAYDEDİLMEDİ.
Etkin emek: PE-07A platform/ekip/bench envanteri sonrası tahminlenecek; takvim sözü yok.

## Amaç ve gereksinim

Tam trajectory yürütme öncesi donanım ve yazılım ön koşullarını kilitlemek.

Gereksinimler: REQ-PE07B. Kaynak: PE PDF s.37–39; ADR-027.
[Kaynak eşlemesi](../roadmaps/SOURCE_MAPPING_r3.md) · [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).

## Girdiler ve bağımlılıklar

Ön koşullar: [PE-06](PE-06.md). Önceki kabul raporu, hashli robot/model/scene/ortam girdileri
ve açık sınırlamalar okunmadan uygulama açılmaz. Negatif bilimsel sonuç ile
teknik kusur ayrılır; özel giriş kapıları ana roadmapte tanımlıdır.

## Yapılacak iş ve çıktılar

1. Physical E-stop, actuator disable, hardware/software limits, driver faults, watchdog/comm timeout, controlled reset/startup/shutdown kanıtları.
2. Readiness authorization: model/calibration/scene/trajectory/firmware identity, fresh state, twin valid, HIL accepted ve operatör yetkisi.
3. UI acil durdurması ayrı kolaylık; fiziksel E-stop yerine geçmez. PC kapanması ve bağlantı kaybında yerel durdurmayı ölç.
4. Kilit atlama/unknown-required-check/stale-authorization negatif testleri.

## Test planı ve kabul

Test kimlikleri: T-PE07B.

- T-PE07B: ön kayıtlı durdurma ve interlock testlerinin tümü PASS; eksik zorunlu bilgiyle execution 0.
- Bu mühendislik kabulü sertifikasyon iddiası değildir; test kapsamı ve sorumlular açık.

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

Planlanan kanıt: `experiments/PE-07B/`. Henüz mevcut/dolu olduğu iddia edilmez.
Teslimatta `RUN_REPORT.md`, `protocol.json`, girdi/çıktı SHA manifesti,
gerçek komut/ortam/seed/başlangıç-bitiş, ham log/JUnit/ölçüm ve
`acceptance.json` gerekir. Ürün testi için demo/benchmark artifactları yukarıdaki
kapsamdadır. [Çalışma şablonu](../templates/RUN_REPORT.md).

## Başarısızlık ve kapsam sınırı

Critical interlock başarısızsa full movement disabled; gerçek donanım değerlendirmesi olmayan sonucu uygunluk belgesine dönüştürme.

## Kapanış

- [ ] Gereksinim → değişiklik → test → ham kanıt bağı tamamlandı.
- [ ] Zorunlu kabul kriterleri ölçüldü; başarısız/çalıştırılmayanlar açık.
- [ ] Regresyon, tekrar üretim, açık sınırlamalar ve hashli devir hazır.
- [ ] STATUS ve TRACEABILITY tarihli kayıtla güncellendi.

Sonraki doğrudan işler: [PE-04](PE-04.md). Karar: VERİLMEDİ. Uygulama: BAŞLAMADI.
Kapanışta plan silinmez; tarihli gerçekleşen sonuç aşağıya eklenir.

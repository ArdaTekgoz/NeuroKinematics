# PE-05A — Temel kalibrasyon ve robot kimliği

Durum: PLANLANDI / NOT_STARTED · Test: NOT_RUN · Belge r1 · Roadmap r3 · 11 Ekim 2026.
Sorumlu: Arda Tekgöz. Başlangıç/bitiş ve gerçek süre: KAYDEDİLMEDİ.
Etkin emek: PE-07A platform/ekip/bench envanteri sonrası tahminlenecek; takvim sözü yok.

## Amaç ve gereksinim

Kapalı çevrim trajectory yürütmeden önce gerçek joint ve model eşleşmesini doğrulamak.

Gereksinimler: REQ-PE05A. Kaynak: PE PDF s.30–33; ADR-027.
[Kaynak eşlemesi](../roadmaps/SOURCE_MAPPING_r3.md) · [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).

## Girdiler ve bağımlılıklar

Ön koşullar: [PE-03](PE-03.md). Önceki kabul raporu, hashli robot/model/scene/ortam girdileri
ve açık sınırlamalar okunmadan uygulama açılmaz. Negatif bilimsel sonuç ile
teknik kusur ayrılır; özel giriş kapıları ana roadmapte tanımlıdır.

## Yapılacak iş ve çıktılar

1. Encoder zero/scale/direction, gear ratio/homing offset, joint order/sign ve physical model identity.
2. Base/TCP/tool, nominal link/axis ölçüleri ve başlangıç calibration; bağımsız ölçüm yöntemi/belirsizlik/cihaz bilgisi.
3. Robot_revision/URDF_hash/firmware/calibration_version ile kalibrasyon paketi; değişiklikte eski validation ve model stale.
4. Sınırlı commissioning hareketleri PE-07A planına bağlı; tam çalışma zarfı sim-to-real değerlendirmesi PE-05B.

## Test planı ve kabul

Test kimlikleri: T-PE05A.

- T-PE05A: işaret/sıra/ölçek/offset ve başlangıç pose eşliği ön kayıtlı mühendislik toleransında.
- Sadece aynı URDF ile iki FK sonucu karşılaştırması fiziksel kalibrasyon sayılmaz; bağımsız ölçüm gerekir.

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

Planlanan kanıt: `experiments/PE-05A/`. Henüz mevcut/dolu olduğu iddia edilmez.
Teslimatta `RUN_REPORT.md`, `protocol.json`, girdi/çıktı SHA manifesti,
gerçek komut/ortam/seed/başlangıç-bitiş, ham log/JUnit/ölçüm ve
`acceptance.json` gerekir. Ürün testi için demo/benchmark artifactları yukarıdaki
kapsamdadır. [Çalışma şablonu](../templates/RUN_REPORT.md).

## Başarısızlık ve kapsam sınırı

Ölçüm veya tolerans kaynağı eksikse full execution yok; nominal URDF fiziksel doğruluk kanıtı değildir.

## Kapanış

- [ ] Gereksinim → değişiklik → test → ham kanıt bağı tamamlandı.
- [ ] Zorunlu kabul kriterleri ölçüldü; başarısız/çalıştırılmayanlar açık.
- [ ] Regresyon, tekrar üretim, açık sınırlamalar ve hashli devir hazır.
- [ ] STATUS ve TRACEABILITY tarihli kayıtla güncellendi.

Sonraki doğrudan işler: [PE-06](PE-06.md). Karar: VERİLMEDİ. Uygulama: BAŞLAMADI.
Kapanışta plan silinmez; tarihli gerçekleşen sonuç aşağıya eklenir.

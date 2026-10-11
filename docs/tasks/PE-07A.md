# PE-07A — Fiziksel kapsam ve güvenlik gereksinimleri

Durum: PLANLANDI / NOT_STARTED · Test: NOT_RUN · Belge r1 · Roadmap r3 · 11 Ekim 2026.
Sorumlu: Arda Tekgöz. Başlangıç/bitiş ve gerçek süre: KAYDEDİLMEDİ.
Etkin emek: PE-07A platform/ekip/bench envanteri sonrası tahminlenecek; takvim sözü yok.

## Amaç ve gereksinim

Gerçek hareketten önce referans platformu, risk sınırlarını ve donanım sorumluluğunu belirlemek.

Gereksinimler: REQ-PE07A. Kaynak: PE PDF s.37–39, 52–54, 58–62; ADR-027 sıra düzeltmesi.
[Kaynak eşlemesi](../roadmaps/SOURCE_MAPPING_r3.md) · [ADR-027](../adr/ADR-027-roadmap-revision-and-neural-role.md).

## Girdiler ve bağımlılıklar

Ön koşullar: [MP-05](MP-05.md). Önceki kabul raporu, hashli robot/model/scene/ortam girdileri
ve açık sınırlamalar okunmadan uygulama açılmaz. Negatif bilimsel sonuç ile
teknik kusur ayrılır; özel giriş kapıları ana roadmapte tanımlıdır.

## Yapılacak iş ve çıktılar

1. TK56 adayının gerçek actuator/encoder/driver/controller, joint/velocity/acceleration limitleri, güç/IO, acil durdurma ve mekanik sınır envanteri; mevcut olmayan özelliği mevcut sayma.
2. Uygulama, ROS, embedded ve driver sorumluluk matrisi; E-stop/disable/limits/watchdog/fault/startup/shutdown/reset gereksinimleri.
3. İlk enerjilendirme ve sınırlı commissioning test planı; fiziksel test alanı, operatör yetkisi, durdurma ölçütleri ve bağımsız stop yolu.
4. Controller rate, feedback age, watchdog timeout, tracking threshold, stop-time/distance ve calibration hedeflerini donanım kaynaklarıyla ön kaydet. Sayısal değerler henüz yoksa BLOCKED_SPEC.
5. PE-02/03 teknik platform ve transport seçimi için implementation specification; STM32/CAN öneridir, zorunlu evrensel NK bağımlılığı değildir.

## Test planı ve kabul

Test kimlikleri: T-PE07A.

- T-PE07A: gereksinim→sorumlu katman→test eşlemesi; kritik limit/zaman/stop değerleri kaynaktan doğrulanmış, açık eksikler listelenmiş.
- Bu task fiziksel uygunluk/safety sertifikası üretmez; PE-07B yürütme interlock kabulü olmadan PE-04 çalışmaz.

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

Planlanan kanıt: `experiments/PE-07A/`. Henüz mevcut/dolu olduğu iddia edilmez.
Teslimatta `RUN_REPORT.md`, `protocol.json`, girdi/çıktı SHA manifesti,
gerçek komut/ortam/seed/başlangıç-bitiş, ham log/JUnit/ölçüm ve
`acceptance.json` gerekir. Ürün testi için demo/benchmark artifactları yukarıdaki
kapsamdadır. [Çalışma şablonu](../templates/RUN_REPORT.md).

## Başarısızlık ve kapsam sınırı

Donanım veya ölçüm imkânı yoksa ilgili fiziksel task WAITING_HARDWARE; simülasyon sonucu yerine konmaz.

## Kapanış

- [ ] Gereksinim → değişiklik → test → ham kanıt bağı tamamlandı.
- [ ] Zorunlu kabul kriterleri ölçüldü; başarısız/çalıştırılmayanlar açık.
- [ ] Regresyon, tekrar üretim, açık sınırlamalar ve hashli devir hazır.
- [ ] STATUS ve TRACEABILITY tarihli kayıtla güncellendi.

Sonraki doğrudan işler: [PE-01](PE-01.md). Karar: VERİLMEDİ. Uygulama: BAŞLAMADI.
Kapanışta plan silinmez; tarihli gerçekleşen sonuç aşağıya eklenir.

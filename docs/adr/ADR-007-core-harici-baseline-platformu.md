# ADR 007 — Core harici baseline ortak yürütme ortamı

Durum: Kabul edildi; yürütme Stage 2 onayına bağlı
Tarih: 24 Eylül 2026
Yazılım hedefi: v1.0.0 · Belge: r1 · Görev: C1-01

## Bağlam

Foundations DLS kanıtı native Windows x64 üzerinde üretildi. KDL, TRAC-IK ve pick_ik MoveIt 2/ROS 2 pluginleridir. Farklı host ve işletim sisteminde ölçülen süreler adil hız sıralamasına sokulamaz. F0-05 robot, sorgu, schema ve hashleri değişmezdir.

## Alternatifler

Native Windows ortak plugin kurulumu için bu makinede doğrulanmış ortam yok. Linux'ta dış servis çağrısı ek transport ve scheduling maliyeti getirir. Ayrı sistemlerde yalnız fonksiyonel doğrulama yapılabilir, fakat T-C00 süre kıyası kapanmaz.

## Karar ve gerekçe

Hedef Ubuntu 24.04 LTS x86_64 / ROS 2 Jazzy / MoveIt 2.15.2 üzerinde tek hostta beş yöntemdir. DLS aynı hostta tekrar koşar. Harici MoveIt C++ pluginleri ve DLS aynı yerel, kalıcı worker request/reply sınırından ölçülür. Ana süre IPC, dönüşüm ve bağımsız doğrulamayı içerir; solver iç süresi ayrıca saklanır. Plugin kaynakları exact commit ile pinlenir, ROS Debian revizyonları ve transitif bağımlılık lock'u Stage 2 smoke öncesi oluşturulur. Windows F0-05 sonuçları tarihsel kanıt olarak kalır.

## Sonuçlar

REQ-C01/T-C00 yeni C1-01 schema ve registry gerektirir. F0-05 schema/runner değiştirilmez; eski sonuç doğrulanabilir kalır. Seçilen Linux ortamı bu makinede henüz kurulmadı; entegrasyon ve süre kıyası `NOT_RUN`. Linux ortak host sağlanamazsa fonksiyonel kanıt ayrı raporlanır ve gecikme kıyası `NOT_COMPARABLE`; zorunlu solver eksikse görev tamamlanmaz. Karar değişikliği ADR revizyonu ve yeni config/hash gerektirir.

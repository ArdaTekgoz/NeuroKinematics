# ADR-010 · C1-01 sabit container bellek tavanı

Durum: UYGULANDI / LINUX PREPARE DOĞRULANDI / ANA ÖLÇÜMLER BEKLENİYOR, 27 Eylül 2026.

## Kanıt ve sorun

`udp-v1` prepare PASS (258 test, beş worker expiry probe). İki smoke girişinde runtime drift kontrolü ölçümleri başlamadan engelledi. Smoke başlangıç tanısı tek farkı yakaladı: `/proc/meminfo` MemTotal `11886228 kB` → `11886236 kB`, 8 kB. Kanıt: `experiments/C1-01/udp-v1-runtime-diagnostic-82fcfe4606a24edfa6a3ebf79c1cba69.log`. Farkın işletim sistemi içindeki nedeni ayrıca kanıtlanmadı. Host toplam RAM gözlemi container'ın uygulanan bellek tavanı değildir. İlk capture-only MATCH tanısı aralıklı farkı yakalamadı; tarihsel kayıtlar korunur.

## Karar

Tüm yeni aşamalar ve beş yöntem için `--memory 8g --memory-swap 8g`: 8 GiB container bellek üst sınırı, swap yok. Docker bu iki değer eşit olduğunda swapı kapatır; [resmî kaynak](https://docs.docker.com/engine/containers/resource_constraints/). 8 GiB, mevcut yaklaşık 11,3 GiB host RAM'den küçüktür; bu görev için sabit kaynak politikasıdır, gerçek kullanım ihtiyacı henüz Linux prepare/pilot ile doğrulanacak.

Runtime kimliği `/sys/fs/cgroup/memory.max=8589934592` ve `memory.swap.max=0` değerlerini kontrol eder. Eksik/unlimited/farklı limit veya swap kontrolü FAIL; host toplam RAM'in 8 GiB altında olması FAIL. Host MemTotal ve MemAvailable aşama başında/sonunda gate içinde gözlem olarak korunur; birebir runtime kimliği karşılaştırmasına girmez. Solver sonuç toleransı, süre bütçesi ve kabul ölçütleri değişmez. Ortam farklarını genel olarak görmezden gelen bir karşılaştırma uygulanmaz; diğer runtime alanları birebir bağlı kalır.

## Geçiş ve kabul

Yeni kaynak/test/script snapshotları için `udp-v2` session: prepare → smoke → pilot → full → verify. Mevcut immutable image/native worker değişmez; Docker veya C++ rebuild gerekmiyor. `udp-v1` lock/gate/snapshot/başarısız deneme logları düzenlenmez ve yeni session kapısı olarak kullanılmaz. Önceki benchmarklar da bu politikayla birleştirilmez.

Yerel bellek politikası negatif testleri ve 8 kB değişimin kayıt altında kalması dahil 29 session testi PASS (`memory-policy-session-tests.xml`). Gerçek PowerShell launcher/mock Docker argüman akışı dört senaryo PASS (`session-launcher-flow-check.json`), gerçek Docker çalıştırılmadı. Frozen-contract kontrolü 52 PASS. Yeni Linux koşuları NOT_RUN; T-C00 kabulü bekliyor. OOM veya kaynak kontrolü hatası olursa sonuç kabul edilmez; yeni politika ve session ayrıca değerlendirilir.

Linux kullanıcı doğrulaması: `udp-v2` prepare 264 PASS / 1 ADR-008 deselected ve beş gerçek worker expiry probe PASS. Runtime lock gerçek cgroup v2 8 GiB / swap0 değerlerini içeriyor; gate host RAM başlangıç/bitiş gözlemlerini koruyor. Kaydedilmiş kanıt bağları salt okunur PASS (`udp-v2-prepare-evidence-check.json`). Yeni smoke/pilot/full/verify henüz NOT_RUN; T-C00 kabulü bekliyor.

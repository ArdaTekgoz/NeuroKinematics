# C1-01 T-C00 kabul çalışma kaydı

Kimlik: RUN-20260928-C101-TC00<br>
Durum: **PASS / C1-01 ACCEPTED**<br>
Görev ve gereksinim: C1-01 / REQ-C01 / T-C00<br>
Tarih ve sorumlu: 28 Eylül 2026, Arda Tekgöz (Docker koşusu) ve Codex (kanıt/kabul incelemesi)

## Soru ve değişiklik

Beş zorunlu baseline varyantı aynı dondurulmuş 12.000 sorgu, iki deadline profili (10/50 ms), beş ölçüm geçişi, robot/TCP, toleranslar ve kaynak tavanıyla çalışıp doğrulanmış kayıt üretiyor mu? Kabul ölçütü tüm 600.000 denemenin eksiksiz, sıra/şema/bağımsız FK/limit/deadline doğrulamasından geçmesi ve ölümcül altyapı hatasının sıfır olmasıdır; her sorgunun çözülmesi şartı değildir.

Stage 1 dondurulmuş girdiler değişmedi. Stage 2 MoveIt KDL, TRAC-IK speed, pick_ik local/global ve aynı IPC/ölçüm sınırındaki DLS varyantını entegre etti. Aynı Ubuntu/ROS ortamı, CPU 0/1, UDPv4, 8 GiB cgroup tavanı ve sıfır swap uygulandı. IPC restart sorunları [ADR-009](../../docs/adr/ADR-009-dds-restart-transport-deneyi.md), tarihsel FK platform ayrımı [ADR-008](../../docs/adr/ADR-008-tarihsel-fk-platform-kapsami.md), bellek politikası [ADR-010](../../docs/adr/ADR-010-c101-bellek-tavani.md) ile kayıtlıdır. Eski eksik ölçümler bu sonuçla birleştirilmedi.

## Tekrar üretim

Depo başlangıcı `main` / `33d18e54d6c9480627985d58e6a82557835f7fc0`; Stage 2 uygulama ve kabul commit'i `207bf734de6536e2b590e922930df1547fdc29f1`, `origin/main` push PASS. Bu son teslim kaydı ayrıca küçük bir belge commit'iyle tamamlanır. Kullanıcının iki ilgisiz Word dosyası commit dışıdır. Runtime [lock](udp-v2/runtime-lock.json) SHA `ad1bd5b23b360f9cfef990b85711256c110f671fc60aab9fb319fb9e6e2ecc24`; image `sha256:00a76905d283882ca2fab1d3093c6ebea635a48e50bf3a1d5bffa52d7d1b74d7`. Temel ortam lock, dpkg closure, Pixi lock, native worker binary, kaynak/test/script snapshot SHA'ları runtime lock içinde. Ubuntu 24.04.5 x86_64, ROS 2 Jazzy, WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`, AMD Ryzen 7 250; CPU affinity `[0,1]`, OMP/OPENBLAS/MKL/NUMEXPR 1, RAM cap 8 GiB, swap0, GPU `NOT_USED`. Host RAM gözlemleri aşama gate'lerinde; bunlar sabit kaynak tavanı değildir. Robot/TCP/config/query hashleri [frozen manifest](frozen-hashes.json), [baseline config](baseline-config.json) ve runtime lock ile bağlı. Öğrenme modeli/checkpoint ve eğitim seed'i bu görevde uygulanmaz; solver rastgelelik ve onun gözlenebilirlik sınırı config/özetlerde, kaynak derleme bilgileri lock içinde.

Kullanıcının depo kökünde çalıştırdığı gerçek sıra:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage prepare -SessionName udp-v2
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage smoke -SessionName udp-v2
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage pilot -SessionName udp-v2
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage full -SessionName udp-v2
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage verify -SessionName udp-v2
```

Full 27 Eylül 20:23:32.771861 UTC'de başlayıp 28 Eylül 14:51:58.223924 UTC'de bitti: **18,4737 saat duvar süresi**. Bu süre etkin insan emeği değildir; etkin emek `NOT_MEASURED`, 20 saat tahminine uyum iddiası yok. Kaynak/kod eşitliği aşamalar arasında denetlendi. Geçişler tek container'da, yöntemler sırayla çalıştı.

## Test ve ham kanıt

| Kapı | Ölçüm / test | Durum ve kanıt |
|---|---:|---|
| Prepare | 264 PASS, 1 ADR-008 deselected; beş gerçek worker expired-request probe | PASS; [gate](udp-v2/prepare/gate.json), [JUnit](udp-v2/prepare/regression.xml) |
| Smoke | Beş yöntem × sekiz = 40 SUCCESS | PASS; [gate](udp-v2/smoke/gate.json) |
| Pilot | Beş yöntem × 24 = 120 kayıt, fatal altyapı hatası 0 | PASS; [gate](udp-v2/pilot/gate.json) |
| Full | Beş yöntem × 120.000 = 600.000, 12.000 farklı sorgu/yöntem | Ölçüm `MEASURED_UNVERIFIED`; [gate](udp-v2/full/gate.json) |
| Offline verify | 600.000 satır, beş yöntem PASS, fatal altyapı hatası 0 | PASS; [gate](udp-v2/verify/gate.json), beş [özet](udp-v2/verify/) |

Verify gate SHA `eda4bf6f815790aaebba4146c5369740e3a1d70d1fc8429cfd6f1084269c8d0d`, full gate SHA `d7b9126583429f93c8629708b11633b99ffaf87ca664799c4bb01665924fb83d`. Gate, full raw ve beş doğrulama özeti SHA bağları yerelde salt okunur denetlendi: [kanıt kontrolü](udp-v2-verify-evidence-check.json). Gerçek Docker ölçümü ve Linux offline verify kullanıcı tarafından yapıldı. Codex ayrıca yeni Docker ölçümü veya bağımsız FK koşusu çalıştırmadı.

Ham veri `udp-v2/full/*-benchmark.jsonl`, yaklaşık 1,1 GB, **LOCAL_ONLY**; beş raw SHA, boyut ve 120.000'er kayıt full gate/özetlerde. Global stderr yaklaşık 35 MB ve büyük full raw dosyalar Git dışındadır; yerel kanıtlar silinmedi. Repo commit'i bu ham dosyaları taşımayacak. Başka makinede tam satır doğrulaması için bu dosyalar ayrıca kopyalanmalıdır; kalıcı harici arşiv `NOT_CONFIRMED`. Deney raporları, gate'ler, lock ve hash manifestleri Git'e alınacak.

## Sonuç ve yorum

| Yöntem | SUCCESS / 120.000 | TIMEOUT | Diğer ortak durum | Profile B deadline başarısı | Tüm denemeler P50 / P95 / P99 ms |
|---|---:|---:|---|---:|---:|
| DLS/default | 75.509 (62,924%) | 33.713 | 10.778 UNRESOLVED | 62,509% | 5,518 / 50,731 / 51,024 |
| KDL/default | 115.348 (96,123%) | 4.652 | 0 | 96,007% | 1,100 / 11,320 / 28,447 |
| TRAC-IK/speed | 119.997 (99,998%) | 2 | 1 JOINT_LIMIT_FAILURE | 99,997% | 1,162 / 1,848 / 2,600 |
| pick_ik/local | 75.150 (62,625%) | 44.849 | 1 UNRESOLVED | 62,625% | 1,282 / 50,901 / 51,190 |
| pick_ik/global | 54.091 (45,076%) | 65.909 | 0 | 44,718% | 39,127 / 50,747 / 68,748 |

Tam konum/yönelim hata dağılımları, alt kümeler, deadline profilleri, tüm/başarılı deneme süreleri, native status ve eksik değer adetleri beş verify özetinde bulunur. Bu tablo yalnız betimsel sonuçtur; eşleştirilmiş bootstrap güven aralığı veya yöntem üstünlüğü testi **NOT_RUN**. pick_ik/global 60.225 restart ve 1.204.540 warmup ile toplam duvar süresini büyüttü; warmup ana deneme süresine dahil değildir. TIMEOUT ve UNRESOLVED erişilemezlik kanıtı değildir. `collision=NOT_CHECKED`; fiziksel güvenlik veya çarpışmasızlık iddiası yok. TRAC-IK'deki bir JOINT_LIMIT_FAILURE ölçüm sonucu olarak korunur, başarılı gösterilmez.

**Kabul kararı:** REQ-C01 / T-C00 **PASS / ACCEPTED**. Beş zorunlu baseline varyantı aynı dondurulmuş sorgu, platform, bellek/CPU/thread politikası, tolerans ve 10/50 ms bütçede eksiksiz ölçülmüş ve bağımsız doğrulamadan geçmiştir. İç solver parametre ve rastgelelik farkları baseline config, dependency kaydı ve runtime kanıtında belgeli. Eşik düşürülmedi; eski başarısız/eksik girişimler ayrı tutuldu.

## Sonraki adım

Stage 2 kaynakları, bu rapor, [görev](../../docs/tasks/C1-01.md), [STATUS](../../docs/records/STATUS.md), [TRACEABILITY](../../docs/TRACEABILITY.md) ve [roadmap](../../docs/roadmaps/C1_Core.md) uygulama commit'inde push edildi. C1-01 çıkışı C1-06 için aynı query_id ve özet/ham SHA bağlarıdır; C1-02/03/04/05 işleri başlamadı. Raw/large stderr yerel kanıtlarını koruyun; başka makinede tekrar doğrulama istenirse 1,1 GB ham dosyalar ayrıca taşınmalıdır. Core fazı C1-01 kabulüyle tamamlanmaz.

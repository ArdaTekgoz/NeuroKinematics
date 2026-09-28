# C1-01 ana koşu hazırlığı

Kimlik: RUN-20260927-C101-MAIN-PREP<br>
Durum: IN_PROGRESS / STAGE_2_IMPLEMENTING<br>
Görev ve gereksinim: C1-01 / REQ-C01 / T-C00<br>
Tarih ve sorumlu: 27 Eylül 2026, kullanıcı ve Codex

## Soru ve değişiklik

Kullanıcının “Onaylıyorum” ve “işleme kaldığın yerden devam et” mesajları ana test kodunu hazırlama onayını verir. Önceki ajan değişimi/onay bekleme kaydı tarihsel durumdur.

UDPv4 hedefli stres deneyi 500/500 kayıt ve 256 yeniden başlatma ile PASS oldu. Yeni ana koşu bütün beş yöntemi aynı UDPv4, CPU 0/1, thread ve ölçüm koşullarında yeniden ölçecek. Önceki dört yöntemin 480.000 doğrulanmış kaydı tarihsel kanıt olarak kalır; yeni koşuyla birleştirilmez.

İncelemede dondurulmuş sözleşmenin kalan süre aktarımıyla uygulama arasında fark bulundu: worker çözücüye tam göreli 10/50 ms veriyordu. Yeni protokol mutlak monoton bitiş zamanını taşır; worker IPC ve dönüşümde geçen süreyi düşer. Giriş hazırlığı ve bağımsız doğrulama dış toplam süreye dahildir. Solver ayarları, query listesi, toleranslar ve dondurulmuş dosyalar değişmez. Native worker değiştiği için yalnız bu worker mevcut kilitli image üzerinde yeniden derlenecek.

Diğer düzeltmeler: son ölçümden sonra kullanılmayacak worker başlatılmaz; bağımsız FK altyapı hatası açık VALIDATION_ERROR olur; bozuk JSON satırının metni ve baytları logda korunur. İlerleme her 1.000 kayıtta ve son kayıtta, ölçüm aralığı dışında yazılır.

`c101_session.py` ve `run_c101_session.ps1`, prepare → smoke → pilot → full → verify sırasını uygular. Kaynak ve testlerin kopyaları salt okunur bağlanır. Image, bağımlılıklar, native binary, kaynaklar, testler, komut scriptleri, CPU ve runtime SHA ile bağlanır; her aşama önceki kapı ve kanıt hashlerini doğrular. Eski veya değişmiş kanıt sonraki aşamayı açamaz. Aynı dizinin üzerine koşu yapılmaz.

## Tekrar üretim

- Başlangıç commit: `33d18e54d6c9480627985d58e6a82557835f7fc0`; Aşama 2 çalışma ağacı kirli, henüz commit/push yok.
- Türetilecek temel image: `sha256:8fa7616bb86884e8ce27c0ff86527e1592e519d2a44ab0bd7eb473fbc032b5ff`.
- Eski ortam lock SHA: `12471f7a5619f3efee1dfd737e57e66615b0930209e50b96fe10ec880e43b5c1`.
- Query list SHA: `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`.
- Yeni image/bağımlılık denetimi: `runtime-build-v1/`; yeni ortak koşu: `udp-v1/`. Henüz üretilmediler.
- Ortam: Ubuntu 24.04 x86_64, ROS 2 Jazzy, Docker Desktop / WSL 2; CPU 0/1; OMP/OpenBLAS/MKL/NUMEXPR 1; TRAC-IK ve global solver iç thread sınırları dondurulmuş configte.
- GPU kullanılmaz. Governor önceki WSL ortamında NOT_AVAILABLE. Yeni RAM/kernel/CPU gerçek prepare sırasında kaydedilecek.
- Gerçek komut sırası: [COMMANDS.md](COMMANDS.md). Docker komutlarını kullanıcı PowerShell'de çalıştıracak.
- Etkin emek toplamı: NOT_MEASURED. Tarihsel oturum süreleri eksik; 20 saat sınırına uyulduğu iddia edilmez. Benchmark bekleme süresi etkin emekten ayrı tutulur.

## Test ve ham kanıt

Yerel Windows regresyonu: **259 PASS / 0 FAIL / 0 skipped**, 19,74 saniye. C1-01 ile F0-05/F0-06 birlikte çalıştırıldı; Windows'ta tarihsel test de kapsamdadır. Kanıt: [JUnit](stage2-main-preparation-tests.xml), [log](stage2-main-preparation-tests.log). Yeni Linux prepare aynı kodla 258 test ve ADR-008 gereği 1 deselected bekler; gerçek kimlik listesi ayrıca denetlenir.

İlk yerel koşuda 258 PASS / 1 FAIL kaydı [XML](stage2-main-preparation-failed-clock.xml) ve [log](stage2-main-preparation-failed-clock.log) olarak korundu. `test_late_finite_candidate_is_validated_but_timeout` gerçek 60 ms beklemeye dayanıyordu. Bu Windows Python 3.12 ortamında monotonic saat `GetTickCount64`, çözünürlük 15,625 ms; sınır testi tek başına tekrarlandığında PASS oldu. Test 60 ms çağrı + 1 ms doğrulama veren kontrollü saat girdisine geçirildi. 50 ms kabul eşiği veya production timeout mantığı gevşetilmedi. Süresi geçmiş gerçek DLS subprocess testi de 259 test içinde PASS.

İki yeni PowerShell scriptinin parser kontrolü 0 hata: [kanıt](stage2-main-powershell-parser.json). Python session/build-audit/probe syntax kontrolü PASS. Native C++ derleme ve beş gerçek worker protokol kontrolü dahil Docker build, yeni Linux prepare/smoke/pilot/full/verify: **NOT_RUN**.

Yerel komut (aynı kapsam; geçici dizin GUID ile ayrıldı):

```powershell
$env:PATH = (Join-Path (Get-Location).Path '.pixi\envs\default\Library\bin') + ';' + $env:PATH
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& .\.pixi\envs\default\python.exe -m pytest -q -p no:cacheprovider tests/c1_01 tests/f0_05 tests/f0_06 --basetemp ('tmp/c101-final-' + [guid]::NewGuid().ToString('N')) -o junit_family=legacy --junitxml=experiments/C1-01/stage2-main-preparation-tests.xml
```

İlk alt incelemede etkinleştirilmemiş Windows DLL yolu NumPy native hata verdi; kullanılan Pixi ortamının `Library/bin` yolu ve thread değişkenleriyle yukarıdaki yerel regresyon tamamlandı. Bu yerel hazırlık, Linux Docker ölçümü olarak değerlendirilmez.

Önceki Linux kritik test sonucu 211 PASS / 1 deselected ve UDPv4 stres sonucu bu revizyonun ana test kabulü değildir. ADR-008 kapsamındaki tarihsel Windows residual testi korunur; yeni Linux prepare topladığı gerçek test kimliklerini JUnit sonuçlarıyla birebir eşler.

## Sonuç ve yorum

T-C00 önceki denemesi INCOMPLETE; görev kapanmadı. Yeni tam koşu 12.000 sorgu × 2 bütçe × 5 geçiş × 5 yöntem = 600.000 ölçüm gerektirir. Full bitişi önce MEASURED_UNVERIFIED olur; bağımsız doğrulama, altyapı hata incelemesi ve görev kabulü ayrıdır. TIMEOUT veya sayısal çözümsüzlük kayıtları atılmaz.

Tam ham JSONL dosyaları normal Git'e eklenmez. Koşu özetleri/gate dosyalarında SHA-256, bayt, satır adedi ve yerel saklama durumu bulunur. Commit öncesi bu dosyaların kalıcı saklama durumu ayrıca kontrol edilecek.

## Sonraki adım

Kullanıcı yeni worker image build/lock çıktısını iletir. Ardından aynı yeni ortamda kritik regresyon → beş solver smoke → küçük pilot → tam T-C00 → offline doğrulama/özet → kabul belgeleri → Aşama 2 commit/push. Sonraki Core görevi başlatılmadı.

Eski 500 kayıt stresindeki duvar süresini global 120.000 kayda doğrusal ölçeklemek yaklaşık 18 saat verir; bu yeni koşu için güvenilir süre tahmini değildir. Yeniden başlatma/ısınma yoğunluğu süreyi belirler; yeni pilot sonrasında kullanıcıya güncel tahmin verilecek.

## 27 Eylül 2026 · Docker komut yolu düzeltmesi

Kullanıcının `build_c101_runtime.ps1` çağrısı Docker komut çözümlemesinde başarısız oldu. `Get-Command docker -CommandType Application` bu Windows ortamında hem `docker.exe` hem uzantısız `docker` için iki sonuç veriyor. `.Source` bir dizi oldu; call operator iki yolu tek program adı gibi değerlendirdi. Docker build başlamadı; `runtime-build-v1/` dizini oluşmadı.

Build ve session launcher açıkça `docker.exe` arayıp ilk Application sonucunun tek `.Path` değerini kullanacak şekilde düzeltildi. Windows PowerShell 5.1 üzerinde her iki gerçek scriptin AST içindeki çözümleme ifadesi çalıştırılarak tek, mevcut `.exe` dosya yolu ve 0 parser hatası doğrulandı. Docker çağrılmadı. Kanıt: [docker-command-resolution-check.json](docker-command-resolution-check.json); geçici kontrol komutu `powershell -NoProfile -ExecutionPolicy Bypass -File .\tmp\check_c101_docker_resolution.ps1`. Önceki yerel 259 test kaydı Python değişiklikleri için geçerlidir; bu düzeltme yalnız PowerShell program seçimini etkiler.

Kullanıcıya açıklanan kapsam: ana eksik ölçüm hâlâ tam T-C00'dır. Ek build/regresyon/smoke/pilot, incelemede bulunan kalan bütçe kusuru ve ortak UDPv4 runtime değişikliği sonrası doğrulamadır. C++ worker KDL/TRAC-IK/pick_ik pluginlerini çağıran yerel yardımcı programdır; yeni kaynağın container içindeki executable'a dönüşmesi build gerektirir. Örneğin 50 ms bütçede IPC/dönüşüm 2 ms tükettiyse plugin yaklaşık 48 ms alır; dış toplam ölçüm bağımsız doğrulamayla birlikte yine 50 ms kabul sınırına tabidir. Eşikler değiştirilmedi. Sonraki kullanıcı adımı aynı build komutunu yeniden çalıştırmaktır.

## 27 Eylül 2026 · Docker Linux motoruna erişim bekleniyor

Komut yolu düzeltmesi sonrası kullanıcı çağrısı `dockerDesktopLinuxEngine` named pipe bulunamadığı için image inspection sırasında exit 1 verdi. Docker CLI seçimi çalıştı; Linux engine API erişilemiyor. Desktop kapalı, henüz başlamamış veya farklı engine seçilmiş olabilir; bu çıktıyla tek alt neden kesinleşmez. Build henüz başlamadı ve `runtime-build-v1/` yerel kontrolde oluşmamıştı. Yeni image/lock/ana testler NOT_RUN.

Sıradaki kullanıcı adımı Docker Desktop'ı açıp motorun hazır olmasını beklemek, `docker.exe version` içinde Server bölümünü ve `docker.exe info --format '{{.OSType}}/{{.Architecture}}'` ile Linux x86_64/amd64 durumunu kontrol etmek; ardından aynı build komutunu çalıştırmaktır. C++ kaynak değişikliği depoda uygulanmıştır; kullanıcıdan elle C++ düzenlemesi istenmez. Bu incelemede Docker başlatılmadı veya çalıştırılmadı.

## 27 Eylül 2026 · Worker build PASS, ortam kilidi kurtarma adımı

Kullanıcı çıktısı ve yerel kaydedilmiş log/image inspection: C++ worker 22,9 saniyede başarıyla derlendi. Image ID `sha256:00a76905d283882ca2fab1d3093c6ebea635a48e50bf3a1d5bffa52d7d1b74d7`; worker binary SHA `f4cc995293ffb2f6ea7fc40e9ac9f2a3896338c88acec3d835b33a781ae19b77`; input manifest SHA `7446120ee88c9ec84bf22f401fe1a95d338f987299f9deeed30a386b299f2bf0`. Build içi dpkg closure/Pixi lock/kaynak/input denetimi PASS: 691 paket ve önceki lock SHA değerleri korunmuş.

Hata ortam lock çağrısında oluştu: inherited entrypoint `pixi run --locked` kaynak değişikliği sonrası editable `neurokinematics` paketini yeniden kurmayı denedi, build backend `hatchling==1.27.0` için PyPI erişimi istedi. Container ağının kapalı olması bu isteği engelledi. C++ derlemesi ve build audit başarısız değildir; final environment-lock henüz yayımlanmadı. PowerShell NativeCommandError biçimindeki başlangıç logları stderr gösterimidir; gerçek başarısız işlem son Pixi kurulumu/lock çağrısıdır. Kanıt `runtime-build-v1/docker-build.log`, `docker-image-inspect.json`, `docker-environment-lock.log`; orijinalleri korundu.

Build lock ve session çağrılarına `PIXI_NO_INSTALL=true` eklendi. EntryPoint'in `--locked` kontrolü devam eder; aynı kurulu Pixi ortamı aktive edilir, package reinstall yapılmaz. Yerel Pixi 0.81.0 `run --help` bu env seçeneğini doğruladı; [resmî run dokümanı](https://pixi.prefix.dev/latest/reference/cli/pixi/run/) davranışı açıklar. Gerçek yerel `PIXI_NO_INSTALL=true; pixi run --locked python ...` NumPy 2.5.3 ve Pinocchio 4.1.0 ile başarılı; session gate testleri **23 PASS**, `runtime-noinstall-session-tests.xml`.

`build_c101_runtime.ps1 -ResumeLock` kayıtlı immutable image ve temel lock bağını kontrol ederek yalnız lock/audit çağrılarını sürdürür. Mevcut build veya hata logunu yeniden yazmaz; her lock denemesi benzersiz log/aday dosya adı alır. Son audit başarısız olursa final lock yayımlanmaz, var olan lock'un üstüne yazılmaz. Windows PowerShell 5.1 üzerinde mock Docker çağrılarıyla iki anlamlı kontrol PASS: başarılı audit tek lock yayımlar; başarısız audit yayınlamayı engeller. Kanıt `runtime-lock-resume-check.json`; gerçek Docker çağrısı yapılmadı. İki launcher parser ve komut yolu kontrolü de PASS.

Sonraki kullanıcı komutu [COMMANDS.md](COMMANDS.md) içindeki `-ResumeLock` çağrısıdır. Linux lock kurtarma, prepare/protokol/smoke/pilot/full/verify henüz NOT_RUN. C1-01 IN_PROGRESS; eski full PARTIAL_VERIFIED kalır, commit/push yapılmadı.

## 27 Eylül 2026 · Runtime build ve ortam kilidi tamamlandı

Kullanıcı `-ResumeLock` çıktısı: RUNTIME BUILD COMPLETE; 691 paket/Pixi lock/source/input denetimi PASS; CPU 0/1; image `sha256:00a76905d283882ca2fab1d3093c6ebea635a48e50bf3a1d5bffa52d7d1b74d7`. Yeni `runtime-build-v1/environment-lock.json` yayımlandı, SHA `9369f45ba52acd6962244bf56ced425c0d895e00771bb5c672895196c6e66990`. Kayıtlı image inspection, audit log, temel lock, host Pixi lock ve input manifest bağları yerelde salt okunur kontrol edildi: PASS. Kanıt [lock-binding-verification.json](runtime-build-v1/lock-binding-verification.json), `docker-environment-lock-20260927-165133-068-2ce18e32f57b488e9e1a90fdd8d65d4f.log`, aynı denemenin dependency audit logu. Bu kontrol Docker çağrısı içermez.

Sıradaki kullanıcı komutu `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage prepare -SessionName udp-v1`. Aynı yeni Linux ortamında kritik regresyon (258 test / ADR-008 gereği 1 deselected bekleniyor) ve beş gerçek worker için süresi geçmiş istek negatif kontrolü yapılacak. Sonuç henüz NOT_RUN; prepare geçerse beş solver smoke, pilot ve tam T-C00 sırası sürer. C1-01 IN_PROGRESS; tam benchmark kabulü ve commit/push bekliyor.

## 27 Eylül 2026 · Prepare launcher değişken adı düzeltmesi

Kullanıcı prepare çağrısı image ID boş olduğu için Docker çağrısından önce durdu. Ortam lock dosyası geçerli ve `udp-v1/` henüz oluşmamıştı. PowerShell değişken adlarında büyük/küçük harf ayrımı yapmaz: `[string]$EnvironmentLock` parametresi ile parsed JSON için kullanılan `$environmentLock` aynı değişkendi. String tip kısıtı JSON nesnesini string'e çevirdi; `.image_id` boş kaldı. Parsed kayıt `$environmentRecord` olarak yeniden adlandırıldı; lock veya native image değiştirilmedi.

Önceki yalnız parser/komut yolu kontrolleri bu akış kusurunu yakalamamıştı. Yeni kalıcı Windows launcher testi gerçek scripti ayrı geçici repo ve mock native `docker.exe` ile baştan sona çalıştırır. Windows PowerShell 5.1 üzerinde dört kontrol PASS: relative lock yolu, absolute lock yolu, sonraki aşamada session lock kopyası, Docker nonzero exit aktarımı. Kanıt `session-launcher-flow-check.json`; gerçek Docker veya Linux test çalıştırılmadı. Tekrar üretim: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\c1_01\check_session_launcher.ps1`.

Sonraki kullanıcı adımı aynı prepare komutunu yeniden çalıştırmaktır; yeni build gerekmiyor. Linux prepare/protokol/smoke/pilot/full hâlâ NOT_RUN; C1-01 IN_PROGRESS.

## 27 Eylül 2026 · Linux prepare PASS

Kullanıcı `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage prepare -SessionName udp-v1` komutunu çalıştırdı: **258 passed, 1 deselected, 22,61 saniye**. ADR-008 kapsamındaki tek tarihsel test dışarıda; yeni bir eşik veya dışlama eklenmedi. Beş gerçek worker için süresi geçmiş istek kontrolü PASS. Runtime SHA `4ececbf3577e381c3ded2b45c3749fc3e71852dcc33dff12d67d9879062a46c4`; prepare gate SHA `83439e24503df4c8f73efd5fa960d99f43f0cd926f345d48d216dc933e69c528`.

Kaydedilmiş gate'in 10 kanıt dosyası, JUnit test kimlikleri, beş worker probe sonucu ve kaynak/test/script snapshotlarının runtime lock ile eşleşmesi yerelde salt okunur doğrulandı: PASS. Kanıtlar: [prepare gate](udp-v1/prepare/gate.json), [JUnit](udp-v1/prepare/regression.xml), [protokol kontrolü](udp-v1/prepare/protocol/expired-request-probe.json), [kanıt denetimi](udp-v1-prepare-evidence-check.json). Docker ve Linux testlerini kullanıcı çalıştırdı; asistanın denetimi yeni ölçüm başlatmadı.

Sıradaki adım aynı session içinde `-Stage smoke`: beş yöntem × sekiz sorgu, toplam 40 ölçüm. Ardından pilot → tam T-C00 → offline doğrulama/özet → görev kabulü → Aşama 2 commit/push. Yeni smoke/pilot/full NOT_RUN; C1-01 IN_PROGRESS. Önceki eksik koşu kanıtları korunuyor.

## 27 Eylül 2026 · Smoke ölçüm öncesi ortam uyuşmazlığı

Kullanıcının `-Stage smoke -SessionName udp-v1` koşusu `runtime/code/test/environment drift since preparation` ile ölçüm öncesinde FAIL. Kanıt: `udp-v1-smoke-20260927-182923-561-d9fcead7f70442a2ba1f468350f6e94e.log`, `udp-v1/smoke-failure-c83e6682ffeb4ec8a1a74a17843bd` önekli hata kaydı. Smoke sonuç klasörü ve ölçümler oluşmadı; prepare PASS kanıtı korunuyor. Kaydedilmiş kaynak/test/script snapshot hashleri yerelde tekrar eşleşti. Diğer runtime alanlarının hangisinin farklı olduğu mevcut hata metninden belirlenemiyor; RAM/CPU/kernel veya başka alan için henüz neden iddiası yok.

`scripts/diagnose_c101_runtime.py` ve `.ps1` hazırlandı: aynı image/policy/snapshotlarla runtime yakalar, kilit ile farklı alanların eski/yeni değerlerini gösterir. Evidence ve snapshot mountları salt okunur; session kilidi/gate değişmez, solver çağrısı ve ölçüm yok; benzersiz host logu oluşturulur. Python/PowerShell syntax, nested/missing-key fark tespiti ve mevcut snapshot hash kontrolü yerelde PASS. Gerçek Docker tanısı NOT_RUN; kullanıcı komutu COMMANDS.md başında. Tanı sonucuna göre düzeltme belirlenecek; kabul kontrolü gevşetilmedi. C1-01 IN_PROGRESS; pilot/full/verify ve commit/push bekliyor.

## 27 Eylül 2026 · Runtime tanısı MATCH, smoke tekrar denemesi

Kullanıcı `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose_c101_runtime.ps1 -SessionName udp-v1` komutunu çalıştırdı. Kaydedilmiş `udp-v1-runtime-diagnostic-cd126d7c4b774b23b204a085887944d9.log` incelendi: MATCH, `differences: []`, runtime SHA `4ececbf3577e381c3ded2b45c3749fc3e71852dcc33dff12d67d9879062a46c4`. Bu yalnız tanı anındaki eşleşmeyi kanıtlar; önceki smoke uyuşmazlığının hangi alandan kaynaklandığı bilinmiyor. Kilit veya test kodu değiştirilmedi; önceki başarısız deneme korunuyor.

Yerel session kontrolü: prepare gate PASS; smoke klasörü yok, `.c101-active` yok. Aynı session ve değişmeyen katı runtime kontrolüyle smoke yeniden denenebilir; yeniden build/prepare gerekmez. Kullanıcıya aynı smoke komutu verildi. Yeni smoke ölçümleri, pilot/full/verify NOT_RUN; görev IN_PROGRESS. Hata tekrarlanırsa uyuşmazlık anında alan farkını kaydetmek için tanı geliştirmesi değerlendirilecek; kontrol atlanmayacak.

## 27 Eylül 2026 · İkinci drift hatası, başlangıç tanısı

İkinci kullanıcı smoke çağrısı da ölçüm öncesi aynı drift hatasıyla FAIL; log `udp-v1-smoke-20260927-185326-386-3264546913ff47d696f513244bf5ac23.log`. Aradaki capture-only MATCH çıktısı hatanın giderildiğini kanıtlamadı; neden hâlâ bilinmiyor.

Tanı launcherına gerçek launcher ile aynı container adı ve `C101_SESSION_SCRIPT` env değeri eklendi. `--smoke-startup` tanısı mevcut snapshot modülünün `main()` başlangıcını çalıştırır; `capture_runtime` dönüşünü bellekte yakalayıp özel BaseException ile ölçüm dispatchinden önce durur. Snapshot dosyaları, runtime lock ve gate'ler değiştirilmez. Gerçek main'in geçici `.c101-active` dosyası oluşturulup finally ile kaldırıldığından evidence mountu yazılabilir; host tanı logu benzersizdir. Başlangıç tanısı ölçüm başlatmaz, önceki failure kayıtlarını korur.

Yerel sentetik kontrol PASS: runtime yakalama sonucu iletiliyor, ölçüm dispatchine erişilmiyor, mutex temizleniyor, capture fonksiyonu ve argv geri yükleniyor. Python ve PowerShell syntax PASS. Docker başlangıç tanısı NOT_RUN; kullanıcıya güncellenmiş tanı komutu verildi. Yeni main smoke/benchmark kodu yazılmadı; prepare kanıtı korunuyor. Sonuç incelenene kadar yeni ölçüm çağrısı bekletiliyor.

## 27 Eylül 2026 · 8 kB RAM farkı ve sabit kaynak tavanı

Kullanıcı smoke başlangıç tanısı DRIFT: tek fark `mem_total`, `11886228 kB` → `11886236 kB`. Kanıt [tanı logu](udp-v1-runtime-diagnostic-82fcfe4606a24edfa6a3ebf79c1cba69.log). Bu, host RAM gözleminin 8 kB değiştiğini kanıtlar; WSL/kernel iç nedeninin ayrıca kanıtı yok. Hiçbir yeni solver ölçümü başlamadı.

[ADR-010](../../docs/adr/ADR-010-c101-bellek-tavani.md) ile container kaynak tavanı açık tanımlandı: tüm yeni aşamalarda 8 GiB / sıfır swap. Native image ve worker aynı kalır. Runtime schema 1.1.0, cgroup v2 memory.max/swap.max değerlerini birebir doğrular; eksik/unlimited/farklı tavan FAIL. Host RAM 8 GiB altında FAIL. Değişken MemTotal/MemAvailable sonuç gate'inde başlangıç/bitiş gözlemleri olarak korunur. Diğer runtime kimlikleri ve solver kabul eşikleri değişmez; drift hata mesajı artık değişen üst alan adlarını da gösterir.

Yerel `tests/c1_01/test_session_gates.py`: **29 PASS, 1,40 saniye**, [JUnit](memory-policy-session-tests.xml). Altı yeni örnek sabit tavan, unlimited/farklı limit/swap, eksik controller dosyası ve 8 kB değişimin gate'e kaydı/runtime kimliğinin korunmasını denetler. `check_session_launcher.ps1`: dört mock senaryo PASS; 8g/8g argümanları doğrulandı ([kanıt](session-launcher-flow-check.json)). `verify_c101_stage1.py`: 52 PASS / 0 FAIL; `git diff --check`: exit 0. Gerçek Docker asistan tarafından çalıştırılmadı.

Kaynak/test/script snapshot kimliği değiştiği için `udp-v2` session gerekir. `udp-v1` dosyaları yeniden yazılmaz; prepare PASS ve iki başarısız smoke korunur. Yeni build gerekmez. Sonraki kullanıcı komutu `run_c101_session.ps1 -Stage prepare -SessionName udp-v2`; beklenen 264 PASS / 1 deselected + beş probe PASS. Yeni Linux politika doğrulaması, smoke/pilot/full/verify NOT_RUN. C1-01 IN_PROGRESS; kabul ve commit/push bekliyor.

## 27 Eylül 2026 · udp-v2 Linux prepare PASS

Kullanıcı `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage prepare -SessionName udp-v2` koşusu: **264 passed, 1 deselected, 15,24 saniye**; beş gerçek worker expired-request probe PASS. Runtime SHA `ad1bd5b23b360f9cfef990b85711256c110f671fc60aab9fb319fb9e6e2ecc24`; gate SHA `87403d7c3edd860f237c75b1c2c4ba1aae02fdc18952eb974d1c05fafee0f67e`. Gerçek cgroup v2 tavanı 8589934592 byte / swap0; host MemTotal başlangıç/bitiş `11886236 kB`, MemAvailable `11009964` → `10973528 kB` olarak gözlem kaydında.

Kaydedilmiş runtime/gate bağı, 10 kanıt dosyası hashleri, 264 testin JUnit/collection kimlikleri, kaynak/test/script snapshot hashleri ve protokol genel PASS kaydı yerelde salt okunur doğrulandı. [Kanıt denetimi](udp-v2-prepare-evidence-check.json), [prepare gate](udp-v2/prepare/gate.json), [JUnit](udp-v2/prepare/regression.xml). Gerçek Linux çalıştırması kullanıcıya aittir; asistan Docker çalıştırmadı. Smoke klasörü ve aktif mutex yok.

Sırada aynı session `-Stage smoke -SessionName udp-v2`: beş yöntem × sekiz sorgu = 40 ölçüm. Çıktısı incelendikten sonra pilot → full → verify → kabul/commit/push. Yeni ana ölçümler NOT_RUN; C1-01 IN_PROGRESS. ADR-010 kaynak tavanı Linux prepare'da doğrulandı; uzun ölçüm doğrulaması bekliyor.

## 27 Eylül 2026 · udp-v2 smoke PASS

Kullanıcı `run_c101_session.ps1 -Stage smoke -SessionName udp-v2` koşusu PASS. Başlangıç/bitiş 16:54:55.768118 → 16:54:58.973274 UTC. Beş yöntemin her birinde sekiz SUCCESS, toplam 40 kayıt; her yöntem bir launch, 20 warmup, sıfır restart ve worker_start_error null. Linux gate ham kayıtların bağımsız FK/sıra/kimlik kontrolünden sonra PASS yayımladı.

Runtime SHA `ad1bd5b23b360f9cfef990b85711256c110f671fc60aab9fb319fb9e6e2ecc24`; prepare gate SHA `87403d7c3edd860f237c75b1c2c4ba1aae02fdc18952eb974d1c05fafee0f67e`; smoke gate SHA `b9c66d373401fb5d16758387e3317eabaa00103dfcf8c4ec477600920a6bfd52`. Host MemTotal başlangıç/bitiş `11886236 kB`. Ham dosyaların hash/bayt/satır sayıları, gate kanıtları, runtime/prepare bağı ve özetlerdeki status/warmup bilgileri yerelde salt okunur doğrulandı: PASS. Bu son kontrol bağımsız FK'yi yeniden çalıştırmadı; gerçek Linux ölçümleri kullanıcı tarafından çalıştırıldı.

Kanıt: [smoke gate](udp-v2/smoke/gate.json), [kanıt denetimi](udp-v2-smoke-evidence-check.json), `udp-v2-smoke-20260927-195453-460-6cb8b1e58fd740bf98f14b3623292bbd.log`, beş ham JSONL ve summary/stderr dosyası. Pilot klasörü ve aktif mutex yok. Sıradaki komut aynı session `-Stage pilot`: 12 sorgu × 10/50 ms × beş yöntem = 120 ölçüm. Pilot/full/verify NOT_RUN; C1-01 IN_PROGRESS, kabul ve commit/push bekliyor. Kolay smoke başarısı tam benchmark performansı olarak yorumlanmaz.

## 27 Eylül 2026 · udp-v2 pilot PASS, full sırada

Kullanıcı pilot koşusu 20:20:15.566742 → 20:20:32.910232 UTC: beş yöntemde 24'er kayıt, toplam 120, gate PASS. DLS 14 SUCCESS / 8 TIMEOUT / 2 UNRESOLVED; KDL ve TRAC-IK 24'er SUCCESS; local 14 SUCCESS / 10 TIMEOUT; global 10 SUCCESS / 14 TIMEOUT. Fatal altyapı hatası ve worker_start_error yok. PASS ölçüm bütünlüğü ve entegrasyon kapısıdır; bütün sorguların çözülmesi anlamına gelmez.

Global 13 launch / 260 warmup / 11 restart; 13,581873 saniye duvar süresinin 11,301380023 saniyesi warmup. 24 kayıt → 120.000 kaba doğrusal ölçeklemesi yalnız global için 18,86 saat; küçük alt küme ve timeout/restart oranı nedeniyle tam koşu ETA'sı değildir. Diğer yöntemler ikişer launch / 40'ar warmup / sıfır restart. Pilot performans sıralaması veya tam benchmark sonucu olarak kullanılmaz.

Yerel salt okunur hash/ham bayt/satır/sıra/status/warmup ve runtime/prepare/smoke bağı PASS; Linux gate'in yaptığı bağımsız FK burada tekrar çalıştırılmadı. Pilot gate SHA `ddff9f7bb310935d3036ca9506ab678c0b456e50e40e1ca6a2aafa3ebda5415c`; runtime SHA `ad1bd5b2...`. Host MemTotal `11886224 kB`, önceki aşamadan farklı gözlem; sabit cgroup policy aynı ve gate PASS. Kanıt [pilot gate](udp-v2/pilot/gate.json), [kanıt kontrolü](udp-v2-pilot-evidence-check.json), `udp-v2-pilot-20260927-232005-072-741a81a2330440d794a254532f289b88.log` ve beş raw/summary/stderr dosyası.

Full klasörü ve aktif mutex yok. Sıradaki kullanıcı komutu `run_c101_session.ps1 -Stage full -SessionName udp-v2`: 12.000 × iki bütçe × beş geçiş × beş yöntem = 600.000 ölçüm. Full/verify NOT_RUN. Bitiş MEASURED_UNVERIFIED; ardından verify/özet/kabul ve Aşama 2 commit/push. C1-01 IN_PROGRESS. Uzun koşu sırasında prize bağlı/uyku kapalı/aynı Docker kaynak ayarları ve ağır başka iş olmaması gereği kullanıcıya belirtildi.

## 28 Eylül 2026 · udp-v2 full ölçüm tamamlandı, offline verify bekliyor

Kullanıcı tam T-C00 ölçümünü çalıştırdı: 27 Eylül 20:23:32.771861 → 28 Eylül 14:51:58.223924 UTC, **18,4737 saat**, beş yöntemin her birinde 120.000; toplam **600.000** ham kayıt. Gate durumu `MEASURED_UNVERIFIED`. Runtime SHA `ad1bd5b2...`, full gate SHA `d7b9126583429f93c8629708b11633b99ffaf87ca664799c4bb01665924fb83d`. Her yöntem worker_start_error null; özetlerde toplam kayıt ve sıcak başlatma sayıları tutarlı.

Ölçülen ortak durum adetleri: DLS 75.509 SUCCESS / 33.713 TIMEOUT / 10.778 UNRESOLVED; KDL 115.348 SUCCESS / 4.652 TIMEOUT; TRAC-IK 119.997 SUCCESS / 2 TIMEOUT / 1 JOINT_LIMIT_FAILURE; local 75.150 SUCCESS / 44.849 TIMEOUT / 1 UNRESOLVED; global 54.091 SUCCESS / 65.909 TIMEOUT. Global 60.227 worker launch / 60.225 restart / 1.204.540 warmup; tam koşu süresinin baskın nedeni bu yeniden başlatma politikası. Bu sayılar henüz bağımsız offline doğrulama ve son karşılaştırma yerine geçmez.

Kaydedilmiş 15 full dosyasının SHA-256'sı, ham JSONL bayt boyutu/satır sayısı/özet status toplamları, runtime/prepare/pilot gate bağı yerelde salt okunur PASS; [kanıt denetimi](udp-v2-full-evidence-check.json), [full gate](udp-v2/full/gate.json). Bu denetimde 600.000 satırın bağımsız FK/şema/sıra kontrolü **NOT_RUN**; Docker asistan tarafından çalıştırılmadı. `verify` klasörü ve aktif mutex yok. Sıradaki kullanıcı komutu `run_c101_session.ps1 -Stage verify -SessionName udp-v2`. Verify PASS ve kayıt/saklama/kabul incelemesi sonrasında Aşama 2 commit/push. C1-01 IN_PROGRESS.

## 28 Eylül 2026 · Offline verify ve kabul

Kullanıcı `run_c101_session.ps1 -Stage verify -SessionName udp-v2` koşusunu tamamladı: beş yöntem PASS, yöntem başına 120.000 kayıt, fatal altyapı hatası 0. Verify gate SHA `eda4bf6f815790aaebba4146c5369740e3a1d70d1fc8429cfd6f1084269c8d0d`; full gate SHA bağı `d7b9126583429f93c8629708b11633b99ffaf87ca664799c4bb01665924fb83d`. Beş verify özeti hashleri ve 12.000 ayrı sorgu/yöntem, 600.000 deneme toplamı yerelde salt okunur PASS: [kanıt kontrolü](udp-v2-verify-evidence-check.json). Kullanıcının Linux satır/FK doğrulaması tekrar çalıştırılmadı.

Gereksinim ve kayıt incelemesi [ayrı T-C00 kabul raporunda](RUN-20260928-T-C00-acceptance.md): REQ-C01/T-C00 PASS / ACCEPTED, görev COMPLETE. Etkin emek NOT_MEASURED; full wall 18,47 saat. Büyük ham veri ve global full stderr LOCAL_ONLY, uzak arşiv NOT_CONFIRMED; Git manifest/gate/özet taşır. Core fazı henüz bitmedi, sıradaki görev C1-02. Kapanış belgeleri commit/push kapsamına hazırlanıyor; bu tarihli geçmiş NOT_RUN/IN_PROGRESS kayıtları kendi zamanlarındaki durumu gösterir.

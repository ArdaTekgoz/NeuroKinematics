# Deney veya uygulama kaydı

**Güncel C1-01 kapanışı:** [28 Eylül 2026 T-C00 kabul kaydı](RUN-20260928-T-C00-acceptance.md). Aşağıdaki Stage 1/Stage 2 devam notları tarihsel koşu durumlarını korur.

Kimlik: RUN-20260924-C101-S1
Durum: IN_PROGRESS / STAGE_1_COMPLETE; T-C00 NOT_RUN
Görev ve gereksinim: C1-01 · REQ-C01
Tarih ve sorumlu: 24 Eylül 2026 · Codex; proje sahibi Arda Tekgöz

## Soru ve değişiklik

Soru: F0-06/G0 girdileri korunurken dört harici IK varyantı mevcut DLS ile karşılaştırılabilir ortak sözleşmede nasıl entegre edilecek? Bu oturumun tek ana farkı kaynak/platform/adapter/benchmark sözleşmesinin sonuç görmeden dondurulmasıdır. Beklenen Stage 1 kabulü hash, config, solver ayrımı, negatif kontroller ve F0 regresyonlarıdır. Solver matematiksel başarı eşiği Stage 1'de ölçülmez.

## Tekrar üretim

Başlangıç branch/HEAD: `main` / `e1971bf8b70154d6f4d882546c20de0eaab2d83d`. Başlangıç `origin/main` aynı SHA. İki ilgisiz, izlenmeyen Word dosyası vardı ve korunuyor. Kapanış commit'i bu kayıtla aynı commit olduğundan içine SHA yazılarak döngü oluşturulmadı; `git log -1 --format=%H -- experiments/C1-01/RUN_REPORT.md` ile bulunur. Yazılım halen `0.1.0`; hedef `v1.0.0`, belge revizyonları ayrı güncellendi.

Gerçek yürütme Microsoft Windows 11 Pro 10.0.26200 x64, AMD Ryzen 7 250 (8 çekirdek/16 mantıksal), 23.31 GiB görünür RAM; Python 3.11.9 sistem, Pixi 0.81.0 ve mevcut kilitli `.pixi` ortamı. GPU kullanımı ve etkin insan emeği `NOT_MEASURED`; yeni ROS/C++ runtime, compiler, CPU seti ve Linux `NOT_AVAILABLE`. Stage 2 hedefi Ubuntu 24.04 LTS x86_64 / ROS 2 Jazzy / MoveIt 2.15.2; henüz kurulmadı. Robot/TCP hashleri ve 15 immutable dosya [frozen-hashes.json](frozen-hashes.json); query-list SHA `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`. Config ve beş exact solver pini [baseline-config.json](baseline-config.json); source/license [DEPENDENCIES.md](DEPENDENCIES.md). Seedler F0-05 query configinden devralınır; pick_ik global determinism henüz `NOT_AVAILABLE`. Gerçek komutlar [COMMANDS.md](COMMANDS.md). Başlangıç/bitiş duvar saati ve download beklemesi kaydedilmedi; insan emeği tahmin edilmez.

## Test ve ham kanıt

| Test | Girdi/çıktı | Gerçek sonuç | Durum |
|---|---|---|---|
| G0/handoff/hash kapısı | F0-06 karar, handoff ve 15 immutable dosya | 15/15 SHA; G0 PASS / ACCEPTED | PASS |
| C1-01 Stage 1 config ve mutasyon | `baseline-config.json`, F0-05 config/query; `stage1-verification.json` | 52/52 PASS; 15/15 negatif mutasyon yakalandı | PASS |
| İlk toplu F0 pytest çağrısı | `stage1-f0-regression.log`, `.xml` | 5 collection error; exit 2; test execution başlamadı | FAIL; ham kanıt korundu |
| F0-00–F0-06 ayrı regresyon | 7 ayrı `.log` / `.xml` | 6/16/102/159/39/175/26 = 523 PASS, 0 FAIL | PASS |
| T-C00 solver ve tam benchmark | `NOT_AVAILABLE` | `NOT_RUN`; 0 harici solver çalıştırması; 0 yeni benchmark satırı | NOT_RUN |

Stage 1 ham kanıt dosyalarının SHA-256 değerleri [evidence-hashes.json](evidence-hashes.json) içindedir. Testler matematiksel harici baseline başarısını veya Linux yürütmesini kanıtlamaz. F0-05’in tarihsel 120.000 satırlı Windows sonucu değiştirilmedi.

## Sonuç ve yorum

Sözleşme donduruldu; C1-01 **IN_PROGRESS / STAGE_1_COMPLETE**, **PASS/COMPLETE değil**. Beş solver kimliği kayıtlıdır ama yalnız DLS'in F0-05 tarihsel sonucu vardır. KDL, TRAC-IK ve pick_ik local/global entegrasyonu `NOT_RUN`; yeni başarı/gecikme `NOT_MEASURED`. Aynı Linux hostta DLS tekrar koşmadan Windows/Linux hız kıyası yapılamaz. Collision `NOT_CHECKED`; fiziksel robot/kalibrasyon/güvenlik test edilmedi. pick_ik deprecated, yalnız temel bakımdadır. ROS paket revizyonları ve runtime lock'u Linux ortamı olmadığı için açık Stage 2 kapısıdır. DLS depo genel lisansı `NOT_DECLARED`.

## Sonraki adım

Açık kullanıcı onayı sonrası tek ana iş: kilitli Linux/Jazzy ortamını kurup ortak adapter ve dört harici solver varyantını kolay smoke ile çalıştırmak. Smoke geçmeden tam T-C00 benchmark yapılmaz. 20 saat etkin emek sınırı gerçek çalışma aralıklarıyla izlenir. C1-02/03/06 başlatılmaz.

## RUN-20260925-C101-S2 · Aşama 2 devam kaydı

Durum: `IN_PROGRESS / STAGE_2_IMPLEMENTING`; T-C00 `NOT_RUN`. Tarih: 25 Eylül 2026; sorumlu: Codex, proje sahibi Arda Tekgöz. Açık onay, branch/remote SHA ve 17 dondurulmuş hash [stage2-approval.json](stage2-approval.json) ve Stage 1 kayıtlarında. Etkin emek başlangıcı 25 Eylül 2026 11:50 UTC; otomatik bekleme, araç indirmesi ve kullanıcının Docker kurulumu etkin emek sayılmayacak, kapanışta gerçek aralıklar ayrıca toplanacak.

### Soru ve değişiklik

REQ-C01 için DLS ve dört harici solver varyantını tek IPC/sonuç doğrulama yüzeyinde nasıl çalıştırabiliriz? Ortak wire contract, 46 alanlı yeni C1-01 sonuç şeması, bağımsız Pinocchio doğrulayıcı, süre/hata sınıfları, Linux-only beş solver runner ve tam benchmark önüne doğrulanmış smoke kapısı eklendi. MoveIt plugin C++ worker kaynak kodu ve ROS paketi hazırlandı; **henüz derlenmedi**. F0-05 dosyaları değişmedi. Platform kullanıcının tercihiyle Windows Docker Desktop + WSL 2 üzerinde Ubuntu 24.04 x86_64 / ROS 2 Jazzy container olacak; ayrı VM kurulmayacak.
`scripts/record_c101_linux_lock.py` gerçek container çalıştığında exact source commitleri ve deb bağımlılık kapanışını kaydetmek üzere hazırlandı; henüz çalıştırılmadı.

### Tekrar üretim

Stage 2 kodu şu an commitlenmemiş çalışma ağacındadır; iki ilgisiz Word dosyası hâlâ kapsam dışı. Yerel Python kontrolleri mevcut `pixi.lock` ile Windows'ta çalıştı. Docker image digest, exact Ubuntu/ROS deb revizyonları, C++ compiler/runtime, Linux CPU seti ve RAM **NOT_AVAILABLE**; kullanıcı PowerShell doğrulama çıktıları bekleniyor. Robot/TCP, query listesi ve config SHA Stage 1 manifestinden doğrulanır; yeni veri/seed/ölçüm sonucu üretilmedi. Gerçek komutlar [COMMANDS.md](COMMANDS.md) içinde. C++ source ve paket sürümleri Stage 1 exact commit pinlerinden tüketilecek, binary paket sürümleri temiz container'da kilitlenecek.

### Test ve ham kanıt

| Test | Gerçek sonuç | Durum |
|---|---|---|
| Yeni Python modülleri `py_compile` | çıkış 0 | PASS |
| `tests/c1_01/test_adapter.py` | 8/8; wire allowlist, q_target sızıntısı, joint/NaN mutasyonu, quaternion round trip, yanlış worker yanıtı, process çökmesi/timeout, smoke gate, bağımsız FK tahrifi ve geç sonlu adayın TIMEOUT olarak korunması; `stage2-adapter-tests.xml` | PASS |
| Stage 1 hash/config yeniden kontrolü | `python scripts/verify_c101_stage1.py`: 52/52 | PASS |
| F0-05 kritik regresyon | `tests/f0_05`: 175/175; `stage2-f0_05-regression.xml` | PASS; 32 JUnit uyumluluk uyarısı |
| Windows DLS ilk IPC denemesi | ready; 50 ms çağrı `TIMEOUT`; Pinocchio B geometri true; stderr geçici kayıtta | FUNCTIONAL CHECK ONLY |
| C++ MoveIt worker derleme / Docker build / dependency lock | yürütülmedi | NOT_RUN |
| KDL, TRAC-IK, pick_ik local/global kolay ve zor smoke | yürütülmedi | NOT_RUN |
| Beş solver küçük uçtan uca smoke ve tam T-C00 | yürütülmedi; 0 yeni Linux denemesi | NOT_RUN |

### Sonuç ve sonraki adım

Bu kayıt harici solver desteği veya kıyaslama başarısı iddiası değildir. C1-01 açık; smoke başarısı olmadan tam benchmark CLI tarafından reddedilir. Kullanıcıdan WSL/Docker doğrulama çıktıları geldikten sonra container'ı gerçek image digest ve deb bağımlılık kilidiyle kur, pinned kaynakları derle, test sırasını uygula, beş kolay smoke sonucunu ve başarısızlıkları sakla. Ancak beşi PASS ise tam T-C00 benchmarka geç. C1-02/03/06 başlamadı.

## RUN-20260926-C101-S2 · Docker ortam doğrulaması ve build hazırlığı

Durum `IN_PROGRESS / STAGE_2_IMPLEMENTING`; T-C00 NOT_RUN. Soru: kullanıcının Docker Desktop/WSL 2 ortamı ortak Ubuntu x86_64 yürütmesini başlatabiliyor mu? Kullanıcının verdiği ekran görüntüleri kalıcı PNG kanıtı olarak saklandı. Docker Desktop 4.92.0 / Engine 29.8.0 Linux amd64 server; WSL 2.7.14.0, `docker-desktop` WSL 2 Running; hello-world başarılı; Ubuntu 24.04.5 LTS / x86_64 container başarılı. Önceki PATH hatası yeni terminalde çözüldü. Komutlar kullanıcı tarafından çalıştırıldı; uzaktan erişim varsayılmadı.

### Soru ve değişiklik

`Dockerfile`, sınırlı build context, exact apt transaction kayıt/kurulum scripti, pinned source fetch/colcon build scripti ve PowerShell build/kanıt/lock scripti hazırlandı. Frozen F0 ve Stage 1 dosyaları değiştirilmedi. ROS base image gerçek RepoDigest ile build argümanına verilecek. Source pinleri ve solver varyantları aynı. `kuka_agilus_support` mesh namespace'i varlıkları değiştirmeden ament kaynak indeksine kaydedilecek. Windows CRLF bash dosyası yalnız container kopyasında LF'ye çevrilecek. İlgisiz kullanıcı Word dosyaları kapsam dışı.

### Test, kanıt ve yorum

PowerShell syntax PASS ve iki hazırlık Python scripti py_compile PASS. [COMMANDS.md](COMMANDS.md) içinde sonraki gerçek build komutu `NOT_RUN` etiketiyle hazır. C++/Bash/Dockerfile gerçek build, Jazzy source uyumluluğu, dependency lock audit ve plugin çalıştırmaları **NOT_RUN**. Kullanıcı Docker/WSL doğrulaması PASS, C1-01/T-C00 kabulü değildir. Exact build compiler/runtime ve deb closure gerçek build çıktısından alınacak; tahmin edilmedi. Etkin emek bu devam kaydında henüz toplanmadı; 20 saat tavanı kapanışta çalışma aralıklarıyla değerlendirilecek.

### Sonraki tek adım

Kullanıcı `scripts/build_c101_docker.ps1` komutunu çalıştırır ve `docker-build.log`/son çıktıyı iletir. Başarılı build + ortam lock audit sonrası Linux adapter testleri ve beş solver kolay smoke komutu verilir. Build hata verirse ilk hata sınıfı ve ham log üzerinden düzelt; smoke ve tam benchmark başlatma.

### 2026-09-26 · İlk Docker build hatası ve düzeltme

Kullanıcı tarafından çalıştırılan build FAIL: `docker-build.log` içinde `given path 'buildtool' does not exist`. Ham log `docker-build-failed-rosdep-20260926.log` olarak korundu. Kaynak checkout ve Pixi locked install başarılı; C++ derleme başlamadı. Hazırlık scriptindeki rosdep seçenek hatası düzeltildi: `-t build -t buildtool -t build_export -t buildtool_export -t exec`. Resmi rosdep parser her tür için tekrarlanan seçenek ister: https://github.com/ros-infrastructure/rosdep/blob/master/src/rosdep2/main.py . PowerShell log helper önceki logları benzersiz `.previous` kopyalarında korur; bu kopyalar Docker context dışında tutulur.

Yerel doğrulama: PowerShell Parser PASS; `git diff --check` çıkış 0 (CRLF uyarıları). Düzeltilmiş gerçek Linux build NOT_RUN; dependency lock, C++ derleme, beş solver smoke ve tam benchmark NOT_RUN. `InvalidDefaultArgInFrom` uyarısı ayrı; build scripti gerçek digest argümanını verir, bu hatanın sebebi değildir.

Sonraki komut (kullanıcının PowerShell terminalinde; yeniden deneme NOT_RUN):

```powershell
Set-Location -LiteralPath 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c101_docker.ps1
```

Build ve ortam kilidi başarıyla tamamlanmadan smoke başlatılmayacak; beş solver kolay smoke PASS olmadan tam benchmark başlatılmayacak.

### 2026-09-26 · Kullanıcı Docker build ve ortam lock sonucu

REQ-C01 / T-C00 hazırlığı: düzeltilmiş build PASS. `docker-build.log`: 10 paket derlendi; c101_moveit_worker, pick_ik, TRAC-IK plugin ve moveit_kinematics tamamlandı; image export başarılı. Eski .h header ve Docker ARG uyarıları başarısızlık oluşturmamış. `docker-build-evidence/` içinde exact apt planları, rosdep çözümlemeleri, dpkg closure, compiler/libc ve deb hashleri mevcut.

`environment-lock.json` ve `docker-environment-lock.log`: Ubuntu 24.04.5 x86_64 / Jazzy, CPU affinity [0,1], Pixi 0.81.0, 691 dpkg paket, frozen kaynak commitleri doğrulandı. Image ID `sha256:407db332e8cdfe00317a53d25b107a906408fd6fc69d5ab3198ad2da7826ab1e`. Yerel read-only kontrol: image inspection ID ile lock ID, gerçek pixi.lock SHA ile lock SHA ve dpkg kayıt adedi eşleşti / PASS. Build/lock kullanıcı tarafından yürütüldü; uzaktan Docker çalıştırılmadı.

Sonraki adım: COMMANDS içindeki Linux adapter pytest komutu; çıkış 0 ve tüm testler PASS sonrasında beş solver smoke. Linux adapter testleri, solver yükleme/çalıştırma, smoke ve tam T-C00 henüz NOT_RUN. Build PASS solver smoke PASS anlamına gelmez; tam benchmark kapısı kapalıdır.

### 2026-09-26 · Linux adapter pytest başlangıç hatası

Kullanıcı denemesi test collection başlamadan `ModuleNotFoundError: No module named 'lark'` ile durdu. Pytest entry-point autoload, ROS `launch_testing` eklentisini Pixi Python içine yükledi. Bu adapter testleri ROS launch eklentisini kullanmıyor; testler çalışmadı, PASS/FAIL test sonucu yok. Ham çıktı `linux-adapter-failed-plugin-autoload-20260926.log` olarak korundu.

Düzeltme: yalnız adapter pytest çağrısına Docker `-e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` eklendi (https://docs.pytest.org/en/stable/how-to/plugins.html). Pytest built-in fixture/assert/JUnit desteği ve sekiz adapter testi aynı kalır. Bağımlılık kilidi/image değiştirilmez; rebuild gerekmez. Düzeltilmiş Linux çağrı NOT_RUN. Smoke ve benchmark NOT_RUN; sıradaki adım adapter testini aynı image ile tekrar çalıştırmak.

### 2026-09-26 · Linux adapter testi PASS

Kullanıcı aynı pinned Docker image ile `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` kullanarak adapter testini yürüttü. Terminal: 8 passed in 0.66s. Kalıcı `linux-adapter-tests.xml` bağımsız okundu: tests=8, failures=0, errors=0, skipped=0, time=0.655s. REQ-C01 adapter sözleşmesi ve hata yolları Linux PASS. Solver plugin smoke / T-C00 henüz NOT_RUN.

Sonraki kullanıcı komutu (beş solver 50 ms entegrasyon smoke; NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" neurokinematics-c101:stage2 python -m neurokinematics.core.cli smoke --output /evidence/linux-smoke --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Ham JSONL, stderr ve yöntem özetleri linux-smoke/ altında; toplu karar linux-smoke/smoke-gate.json. Beş yöntem PASS sonrası diğer kabul kontrolleri ve küçük uçtan uca smoke; ardından tam benchmark. Bu çağrı tam benchmark başlatmaz.

### 2026-09-26 · Beş solver entegrasyon smoke ve ek kontrol

Kullanıcı Linux smoke: dls/default, kdl/default, trac_ik/speed, pick_ik/local, pick_ik/global PASS; yöntem başına 8, toplam 40 ölçülmüş kayıt. linux-smoke/smoke-gate.json mevcut; beş ham SHA-256 yerelde eşleşti. Solver smoke kapısı aynı Linux yürütmesinde PASS.

Ek Windows offline smoke_gate kontrolü FAIL: independent orientation_error_deg mismatch; bu platformlar arası yeniden FK hesabıdır, Linux smoke sonucu iptal edilmiş veya eşik değiştirilmiş değildir. Kaynak henüz belirlenmedi. Kabul/T-C00 tamamlanmadı. Önce aynı immutable Linux image içinde beş raw verify-results çalıştırılacak; ardından negatif/F0 ve küçük uçtan uca kontroller tamamlanacak. Tam benchmark başlatılmadı.

Sonraki kullanıcı komutu (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
foreach ($solver in @('dls/default', 'kdl/default', 'trac_ik/speed', 'pick_ik/local', 'pick_ik/global')) {
    $safeId = $solver.Replace('/', '-')
    docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" neurokinematics-c101:stage2 python -m neurokinematics.core.cli verify-results --results "/evidence/linux-smoke/$safeId-smoke.jsonl" --solver $solver --mode smoke
    if ($LASTEXITCODE -ne 0) { throw "Verification failed: $solver" }
}
```

### 2026-09-26 · Aynı Linux image offline doğrulaması PASS

Kullanıcı verify-results çıktısı: beş yöntem ayrı ayrı PASS, her biri 8 kayıt, SHA değerleri smoke-gate ile aynı. Windows yeniden FK farkının kök nedeni açık; aynı Linux runtime tekrar kontrolü PASS olduğundan Windows sonucu Linux benchmark ölçümü olarak kullanılmayacak. Eşikler değiştirilmedi.

Yerel kritik regresyon: `pixi run --locked python -m pytest -q tests/f0_05 tests/f0_06 --junitxml=experiments/C1-01/stage2-f0_05-f0_06-regression.xml`: 201 PASS, 56 JUnit record_property uyarısı, 17.98s, çıkış 0. Bu Windows/Pixi regresyonudur. Aynı Linux image regresyon komutu sırada / NOT_RUN; frozen kaynaklar değişmeden yalnız test dizini read-only mount edilir:

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Tests = (Resolve-Path -LiteralPath '.\tests').Path
docker run --rm --cpuset-cpus 0,1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Tests,target=/work/tests,readonly" neurokinematics-c101:stage2 python -m pytest -q tests/c1_01/test_adapter.py tests/f0_05 tests/f0_06 -o junit_family=legacy --junitxml=/evidence/linux-critical-regression.xml
```

C1-01 küçük uçtan uca 10/50ms kontrol ve tam T-C00 henüz NOT_RUN. Foundations source/acceptance girdileri değiştirilmedi.

### 2026-09-26 · Linux kritik regresyon FAIL ve sayısal tanı

Kullanıcı Linux kritik regresyon: 208 PASS / 1 FAIL, 12.34s. F0-06 test_real_benchmark_schema_mutation, Windows tarihsel sonuç ilk satırında Linux FK orientation_error_deg yeniden hesabı uyuşmuyor; mutation silme adımına ulaşmadan duruyor. Ham log linux-critical-regression-failed-20260926.log, XML linux-critical-regression.xml. Bu gate PASS değildir; tam benchmark ve küçük uçtan uca çalışma bekletildi.

Read-only scripts/diagnose_c101_cross_platform.py eklendi. Yerel Windows/Pixi tanı çıkış 0: position=0.0009426571163315197m, orientation=0.0010457694117205901rad / 0.05991817363546874deg eski kayıtla birebir aynı; A/B geometry true ve limit PASS. Linux farkı henüz sayısal olarak ölçülmedi, kök neden açık. F0 girdileri/kodu ve kabul eşikleri değiştirilmedi. Sonraki Linux tanı (NOT_RUN):

```powershell
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" neurokinematics-c101:stage2 python /diagnostics/diagnose_c101_cross_platform.py | Tee-Object -FilePath "$c101Evidence\linux-cross-platform-diagnostic.log"
```

### 2026-09-26 · Platform kapsamı ADR-008

Linux tanı: archived 0.05991817363546874deg, native 0.05991817364763415deg; fark 1.216541e-11deg. Konum ve A/B/limit kararları aynı. Kesin alt runtime sebebi açık. ADR-008 tarihi Windows testini korur; orijinal 208 PASS / 1 FAIL sonucu değiştirilmez. Ek C1-01 native residual schema/mutation testi Windows 1 PASS; Linux NOT_RUN. Eski kayıttaki numeric alanlar yalnız geçici bellek kopyasında native FK ile türetilir; eski kanıt veya tolerans değiştirilmez. Küçük smoke/tam benchmark NOT_RUN.

Sonraki Linux kapsamı açık regresyon (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Tests = (Resolve-Path -LiteralPath '.\tests').Path
docker run --rm --cpuset-cpus 0,1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Tests,target=/work/tests,readonly" neurokinematics-c101:stage2 python -m pytest -q tests/c1_01 tests/f0_05 tests/f0_06 --deselect=tests/f0_06/test_gate.py::test_real_benchmark_schema_mutation -o junit_family=legacy --junitxml=/evidence/linux-portable-critical-regression.xml
```

Deselect edilen test Windows'ta PASS, orijinal Linux'ta FAIL; bu yeni koşu orijinal Linux 209/209 PASS anlamına gelmez. Ek native test eksik alan ve residual tahrifini aynı strict validator ile kontrol eder.

### 2026-09-26 · Taşınabilir Linux kritik regresyon PASS ve pilot hazırlığı

Kullanıcı sonucu 209 passed / 1 deselected, 12.14s. XML bağımsız okundu: 209 test, failures/errors/skipped=0, 12.135s. Ayrı tarihsel Windows residual testi ADR-008 kapsamıyla korunur; eski Linux 208 PASS / 1 FAIL kaydı değişmedi. Beş solver Linux smoke + offline PASS. C1-01 IN_PROGRESS / STAGE_2_IMPLEMENTING, T-C00 NOT_RUN.

`scripts/run_c101_pilot.py` hazır: aynı image, CPU0/1, Jazzy/thread policy, 209 testlik scoped regression ve beş smoke raw doğrulama kapısından sonra frozen query listinden her main/boundary/singularity × local/wide grubunun ilk ikisini seçer (12 sorgu). İki bütçe10/50ms × tek ölçüm geçişi × beş solver =120 kayıt. Her bütçe ilk20 frozen query warmup; mevcut ortak worker/timing/validator kullanılır. Pilot planı yalnız bellekte bir geçiştir, baseline-config/full benchmark değiştirilmez. Ham dosyalar linux-pilot/ içindeki benchmark isimli helper çıktılarıdır; yalnız subset pilot kanıtıdır, tam T-C00 değildir. Native timeout/matematiksel başarısızlıklar korunur; process/install/adapter/invalid/validation hatası veya eksik sıra/sayı pilot FAIL. Kolay solver geometri başarısı önceki beş smoke kapısında zorunlu.

Yerel select_rows doğrulama PASS (12, altı grubun her birinde2); py_compile PASS, diff-check PASS. Pilot gerçek Linux yürütmesi NOT_RUN. Rerun önceki linux-pilot dizini üzerine yazmaz.

Sonraki kullanıcı komutu:

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" neurokinematics-c101:stage2 python /diagnostics/run_c101_pilot.py --evidence /evidence --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Kalan sıra: pilot PASS → runner/code ve tekrar üretim gözden geçirme → tam T-C00 (12000×2×5×5=600000 ölçüm; warmup ayrı) → aynı Linux runtime'da her raw offline doğrulama + subgroup özetleri → gereklilik/kabul ve 20saat etkin emek kaydı → STATUS/TRACE/task/roadmap/RUN_REPORT kapanışı → yalnız C1-01 kapsamı commit + remote push doğrulama. Core fazı bu görevle tamamlanmaz; sonraki görevler roadmap bağımlılıklarına göre. Kullanıcının Word dosyaları commit dışı. Aşama2 commit/push henüz yapılmadı; tam ölçümler ve kabul kanıtı henüz yok. Pilotun geçen sonucu matematiksel tam benchmark üstünlüğü veya robot güvenliği iddiası değildir.

### 2026-09-26 · Pilot PASS; full öncesi warm restart düzeltmesi

Kullanıcı pilot PASS, 12 sorgu ×2bütçe ×5solver =120 kayıt; her solver24. Beş raw SHA kontrolü PASS. DLS14SUCCESS/8TIMEOUT/2UNRESOLVED; KDL24SUCCESS; TRAC24SUCCESS; picklocal14SUCCESS/10TIMEOUT; pickglobal10SUCCESS/14TIMEOUT. Bunlar küçük pilot dağılımları, tam üstünlük sonucu değildir.

Runner incelemesi somut kusur buldu: pickglobal pilot worker_launch_elapsed_ns listesi35 eleman; timeout sonrası replacement worker ısıtılmadan ölçülüyordu. Eski pilot kanıtı korunur; tam T-C00 için yeterli değil. runner.start_warm_worker her yeni worker üzerinde aynı ilk20 frozen warmup sorguyu yürütür; warmup gecikmiş yanıtını dış süreye katmadan en çok3s toplar, process ölürse koşu INCOMPLETE olur. Ölçülmüş çağrıların late window20ms, deadline ve toplam timing sözleşmesi değişmedi. Warmup çağrıları ve restart sayıları ayrıca kaydedilir.

Yeni regression test replacement warmup sırasını kontrol eder. Yerel tests/c1_01 10/10 PASS,1.67s, stage2-warmup-fix-tests.xml. PowerShell Parser PASS. Pilot gate yeni scoped test sayısı210 ister. Build script önceki ortam/image/smoke/pilot/regression kanıtını attempts/<timestamp>/ kopyasına korur; bu kopyalar Docker context dışıdır.

Sonraki kullanıcı adımı düzeltilmiş image build+lock (NOT_RUN):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c101_docker.ps1
```

Sonrası: yeni Linux critical210test → yeni beş smoke → yeni pilot (önceki linux-pilot korunarak yeni çıktılar) → tam T-C00. Tam benchmark NOT_RUN. Commit/push kapanış sonrası; Core/C1-01 tamamlandı değildir.

### 2026-09-26 · Warm restart düzeltmesiyle image rebuild PASS

Kullanıcı build + lock PASS: 10 ROS paket derlendi; image ID sha256:8393248c1e48b539611f209fbb9db7678cc879ff2099a33e794ff9167484278c. Yerel image inspection / environment lock ID ve Pixi lock SHA eşleşmesi PASS. 691 dpkg closure aynı d7e5e17a68b5b5e202be392de1d1fc23a16daac85b65e04798fb2227924ce76d; CPU0/1, Jazzy. Önceki kanıt attempts/20260926-135117-390/ içinde korunmuş. Yeni image smoke/pilot/benchmark henüz NOT_RUN.

Sonraki kullanıcı komutu (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Tests = (Resolve-Path -LiteralPath '.\tests').Path
docker run --rm --cpuset-cpus 0,1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Tests,target=/work/tests,readonly" neurokinematics-c101:stage2 python -m pytest -q tests/c1_01 tests/f0_05 tests/f0_06 --deselect=tests/f0_06/test_gate.py::test_real_benchmark_schema_mutation -o junit_family=legacy --junitxml=/evidence/linux-portable-critical-regression.xml
```

Yeni restart-warmup regression nedeniyle scoped test beklentisi210. Deselect tarihi Windows residual testidir (ADR-008); eski FAIL kaydı korunur. Tam T-C00 başlamadı. Sonraki sıra yeni Linux critical → beş smoke →120kayıt pilot → full600000kayıt.

### 2026-09-26 · Yeni image kritik test PASS ve smoke yenileme

Kullanıcı yeni image scoped critical:210PASS/1deselected,12.59s. JUnit bağımsız okuma210,failures/errors/skipped0,time12.585s. ADR-008 ayrılan test değişmedi. Eski linux-smoke ve linux-pilot depo içinde linux-smoke-before-warmup-fix ve linux-pilot-before-warmup-fix adlarıyla korundu; önceki image kanıtıyla attempts/20260926-135117-390/ içinde de mevcut. Yeni output dizinleri boş, kanıt üzerine yazılmayacak.

Sonraki kullanıcı komutu (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" neurokinematics-c101:stage2 python -m neurokinematics.core.cli smoke --output /evidence/linux-smoke --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Yeni smoke PASS sonrası pilot120koşu; tamT-C00henüzNOT_RUN. Yeni image8393248c1e48... üzerindeki warmup regression testPASS; harici warmup davranışı yenilenen smoke/pilot ile doğrulanacak.

### 2026-09-26 · Warm restart düzeltmeli image beş smoke PASS

Kullanıcı yeni image beş solver smoke PASS, 11:07:20.291505–11:07:23.173456 UTC; her solver8, toplam40ölçüm. Yerel beş hamSHA eşleşmesiPASS. Yeni summary: her solver warmup_calls20, launches1, restart_warmups0. Smoke veri alanları doğrulandı; aynı Linux FK offline denetimi pilot scriptinin smoke_gate kontrolünde çalışacak. Yeni image kritik210testPASS. Yeni pilot dizini henüz yok.

Sonraki kullanıcı komutu (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" neurokinematics-c101:stage2 python /diagnostics/run_c101_pilot.py --evidence /evidence --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Pilot120kayıt geçtikten ve restart warmup kayıtları incelendikten sonra fullT-C00 komutu verilecek. Tam benchmarkNOT_RUN, Aşama2henüz commit/push yapılmadı.

### 2026-09-26 · Yeni pilot PASS; tam T-C00 başlatma komutu

Kullanıcı yeni pilot120kayıtPASS,11:08:34.316850–11:08:51.813725UTC. Her yöntem24ölçüm. Beş rawSHA ve environment_lock_sha eşleşmesiPASS. DLS/KDL/TRAC/local:2launch,40warmup,0restart; global14launch,280warmup,12restart. Her worker20warmup sözleşmesi gerçek pilotta doğrulandı. TIMEOUT/UNRESOLVED sonuçlar korunur; matematiksel üstünlük sonucu çıkarılmadı.

scripts/run_c101_full.py ek read-only pilot kapısı: aynı audited image ID/lock, 12frozen selection, her solver24pilot kaydı canonical/order/FK yeniden doğrulama, source config ve launch/warmup sayısı. Ardından mevcut runner full planını değişmeden kullanır:12000query×2deadline×5pass×5solver=600000ölçüm,120000/solver. Çıktı linux-full/, mevcut dizin varsa üzerine yazılmaz. run-binding.json image/lock/config/code/pilot/smoke/regression hashlerini taşır. Yerel py_compilePASS; gerçek fullNOT_RUN. Bağlanan script için rebuild gerekmez.

Sonraki kullanıcı komutu (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
$c101Image = (Get-Content -LiteralPath "$c101Evidence\environment-lock.json" -Raw | ConvertFrom-Json).image_id
docker run --rm --name c101-full-benchmark --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" $c101Image python /diagnostics/run_c101_full.py --evidence /evidence --image-id $c101Image --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

İkinci PowerShell'de ilerleme kontrolü (ölçüm çalışmasını değiştirmez):

```powershell
Get-ChildItem -LiteralPath 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\C1-01\linux-full' -Filter '*-benchmark.jsonl' | Select-Object Name,Length,LastWriteTime
```

Başlangıç PILOT VERIFIED sonrası uzun sessiz çalışma normal olabilir; ham dosya boyutu ilerlemeyi gösterir, kesin test sayısı/başarı sonucu değildir. Bitiş MEASURED_UNVERIFIED veya INCOMPLETE; PASS değildir. Sonrası aynı Linux runtime verify-results/summarize, kapsam/negatif/handoff/tekrar üretim kapanışı ve20saat etkin emek kaydı, ardından yalnız ilgili Stage2commit/push. Koşu bekleme süresi etkin emek değildir. FullNOT_RUN olarak kalır, kullanıcı başladı/çıktı kanıtı gelince güncellenir.

### 2026-09-26 · Full T-C00 INCOMPLETE; IPC restart kusuru

Kullanıcı full11:11:52.181591–12:19:53.129788UTC. DLS/KDL/TRAC/local120000'er MEASURED_UNVERIFIED; global115/120000INCOMPLETE (115TIMEOUT). Toplam480115ölçülmüş kayıt. Beş hamSHA yerelde eşleşti. Global summary worker_start_error INVALID_OUTPUT: invalid ready line Expecting value line1column1; 116launch,2300tamamlananwarmup,114restart-warmup. Kesin bozuk ready bytes eski kodda saklanmamış; tek logdan alt olay kesin belirlenemiyor.

Kod incelemesi deterministik IPC lifecycle risklerini doğruladı: reader closure self.lines yeniden atanmasını izliyordu; eski reader yeni process kuyruğuna yazabiliyordu. stop.communicate stdout için reader thread ile yarışıyordu. Queue process-local yakalandı; stop yalnız wait + reader join +streamclose yapar. Invalid JSON satırını görmezden gelme/threshold değişikliği yok. Yeni eski-reader/queue isolation regression önceki davranışı yakalar. Yerel11/11PASS,1.75s,stage2-queue-isolation-tests.xml. Tam koşu eski code/image kanıtıyla korunur; yeni fixLinuxNOT_RUN. ROS init öncesi log warning ayrıca mevcut, duruş nedeni olarak kanıtlanmadı.

Önce aynı original immutable image'de tamamlanan4yöntem verify+summarize (NOT_RUN):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
$c101Image = (Get-Content -LiteralPath "$c101Evidence\environment-lock.json" -Raw | ConvertFrom-Json).image_id
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" $c101Image python /diagnostics/verify_c101_completed.py
```

Bu script raw değiştirmez; tam120000yöntemleri strictcanonical/FK/status/deadline/hash ile doğrular, verified-summary.json ve completed-methods-verification.json üretir. GlobalINCOMPLETEkalır, aggregatePARTIAL_VERIFIED/full_acceptancefalse. Ölçülen dört yöntemin kendi içinde geçerli kayıtları korunur; global yokken tam kıyas/üstünlük/CorePASS iddiası yok. Yeni image/fixrestartstress ve ölçüm koşullarını koruyan rerun planı sırada. Build archive artık linux-full de korur/context dışında. Pilot yeni scopedtest211ister. Commit/push/kapanış yapılmadı; C1-01IN_PROGRESS/T-C00INCOMPLETE.

### 2026-09-26 · Dört tamamlanmış yöntem bağımsız doğrulandı

Kullanıcı aynı original immutable image'de verify_c101_completed.py: aggregatePARTIAL_VERIFIED,full_acceptancefalse. DLS/KDL/TRAC/local her120000 kayıt PASS (canonical,sıra,FK,hash,status/deadline integrity). Global115INCOMPLETE. DörtverifiedrawSHAdeğişmedi. BuPASS başarı oranı100% değildir. DLS75315SUCCESS/34434TIMEOUT/10251UNRESOLVED; KDL115452SUCCESS/4548TIMEOUT; TRAC119993SUCCESS/5JOINT_LIMIT_FAILURE/2TIMEOUT; local75150SUCCESS/44850TIMEOUT. NativeSUCCESS veya commonSUCCESS tek başına herProfile deadline kabul oranı değildir; verified-summary grupları kullanılacak. Limithataları korunur; erişilemezlik/güvenlik/hızüstünlüğü çıkarımı yok.

IPCqueuefix yeniimageNOT_RUN. stress_c101_global_restart.py hazır: ilk250frozenquery,10/50ms,1geçiş,500ölçüm; enaz100restart gerçekleşmeli, herlaunch20warmup, fataltransporterror0 ve500sıralıFKdoğrulamakayıt. Bu gate performance/tamT-C00 değildir, 115teki restart sorunu için hedefli regresyondur. PythoncompilePASS; LinuxstressNOT_RUN. Yeniimage211kritiktest önce gerekli.

Sonraki kullanıcı adımı build/lock (NOT_RUN):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c101_docker.ps1
```

Build script mevcut linux-full (raw+verified özetler) ve eskiimage/lock/smoke/pilot kanıtını attempts/<timestamp>/ içine korur. Yeniimagehazırsonrası211Linuxcritical→globalrestartstress→yenismoke/pilot→ölçümkoşullarıveprotokolrevizyonlarıaçıklanarakrerun. Mevcut4yöntemi silme/yeni sonuçla üstüneyazma; finalkarşılaştırma aynıtiming/runtimekoşullarıyla yapılmalı. Kapanış/commit/pushyok; C1-01STAGE_2_IMPLEMENTING/T-C00INCOMPLETE.

### 2026-09-26 · IPC düzeltmeli rebuild PASS ve kullanıcı sıralama talimatı

Kullanıcı build/lockPASS:10paket, image sha256:8fa7616bb86884e8ce27c0ff86527e1592e519d2a44ab0bd7eb473fbc032b5ff. Image inspection ID/lock ve gerçekPixilockSHAeşleşmesiPASS. 691dpkgclosure aynı; önceki fullraw+verifiedözet archive attempts/20260926-155030-006/linux-full/ doğrulandı. YeniimageLinux211kritiktest ve globalrestartstressNOT_RUN; T-C00öncekikoşuINCOMPLETE.

Kullanıcı yeni talimat: sonraki ANA smoke veya benchmark testi kodu yazılmadan önce durup kullanıcı onayı istenecek; kullanıcı ajan değişikliği yaparak o kodu yazdırıp test edecek. Yeni ana smoke/benchmark kodu yazılmadı. Mevcut Linux unit/negative/regression adımıyla devam etmek açıkça yetkilendirildi. Agent devri/subagent otomatik başlatılmayacak. Sıra:211kritikregresyon→mevcut hedefli500kayıtrestartstress→ana smoke/pilot/benchmark kodu hazırlama sınırında DUR/kullanıcı onayı+ajan değişimi→kalan ölçüm/kapanış.

Sıradaki kullanıcı komutu (NOT_RUN, mevcut testler):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Tests = (Resolve-Path -LiteralPath '.\tests').Path
$c101Image = (Get-Content -LiteralPath "$c101Evidence\environment-lock.json" -Raw | ConvertFrom-Json).image_id
docker run --rm --cpuset-cpus 0,1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Tests,target=/work/tests,readonly" $c101Image python -m pytest -q tests/c1_01 tests/f0_05 tests/f0_06 --deselect=tests/f0_06/test_gate.py::test_real_benchmark_schema_mutation -o junit_family=legacy --junitxml=/evidence/linux-portable-critical-regression.xml
```

Beklenti211PASS/1deselected(ADR-008tarihselWindowsFKtestikapsamı). Eski210testçıktı arşivdekorunur. Ana smoke/benchmark yeni code/yürütmesi verilmedi, görevkapanmadı; commit/pushyok.

### 2026-09-26 · IPCfiximage kritik211PASS; mevcut restartstress sırası

Kullanıcı211passed/1deselected12.89s. JUnit bağımsız okuma211,failures/errors/skipped0,time12.880s. Aynı audited8fa7616b...image kullanıldı. ADR-008tarihselWindowsFKtesti ayrı tutuluyor. Yeniana smoke/benchmarkkoduyazılmadı; kullanıcıonay/ajandeğişimsınırı korunur.

Sıradaki MEVCUT hedefli pick_ik/global restartstressscripti (NOT_RUN):ilk250frozenquery×10/50ms×1pass=500ölçüm. Enaz100restart,herlaunch20warmup,500kayıtstrictFK/sıra/hash vefataltransporterror0kontrolü. YeniLinuxoutputdizinihenüzyok.

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
$c101Image = (Get-Content -LiteralPath "$c101Evidence\environment-lock.json" -Raw | ConvertFrom-Json).image_id
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" $c101Image python /diagnostics/stress_c101_global_restart.py --evidence /evidence --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Kanıtlinux-global-restart-stress/stress-gate.json vehelperraw/summary/stderr. BuhedefliIPCregresyonu,ana smoke/tamT-C00değil; gerçekmatematikselTIMEOUTsonuçlarıkorunur. Çıktıincelendiktensonraanasmoke/benchmarkkoduyazmaadımında kullanıcıonayıveajandeğişimibeklenecek. C1-01IN_PROGRESS; öncekiT-C00INCOMPLETE/4yöntemPARTIAL_VERIFIEDkalır.

### 2026-09-26 · IPCfix sonrasında restartstress tekrar INCOMPLETE

Kullanıcı globalrestartstress121/500INCOMPLETE:121TIMEOUT,120restartwarmup,2420warmupcall. worker_start_error INVALID_OUTPUT invalidreadyline JSONExpectingvalue. FailedrawSHA eşleşti(6ca5bf5a...). Bu sonuç öncekiqueuefixproblemitamçözdüiddiasınıdesteklemez. Queueisolationyereltestgeçerli; gerçekbozukreadybytehenüzsaptanmadı. Altsüreç/ROS/DDSbaşlangıçstdoutkirlenmesihipotezi araştırılacak, kanıtlanmadı. Ana smoke/benchmarkbekletilir.

Yenibuildistemeöncesi aynıimagehedeflitanı scripts/diagnose_c101_ready.py hazır:250frozenquery/10ms/1pass,aynıtransportdeadline/warmup,strictJSONparserValueErroranındarawsatırrepr/utf8hexve/dev/shmnameskaydedilir. Bozuksatıratlanmaz/başarılısayılmaz,source/threshold/image değişmez. Linux-ready-diagnostic/ ayrıçıktı,eski121kanıtkorunur. py_compilePASS; gerçekLinux tanıNOT_RUN. Kullanıcı yeniANA smoke/benchmarkkodundanönceonay/ajandeğişimi sınırı korunur; buhedeflihata tanısıdır.

Sonraki kullanıcıkomutu:

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
$c101Image = (Get-Content -LiteralPath "$c101Evidence\environment-lock.json" -Raw | ConvertFrom-Json).image_id
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" $c101Image python /diagnostics/diagnose_c101_ready.py
```

C1-01IN_PROGRESS,fullT-C00INCOMPLETE;4öncekitamamlanmışyöntem480000kayıtverifiedolarakkorunur. Commit/pushyapılmadı. Tanıçıkış0başarıgatesideğil; diagnostic-report.json gerçekerror/statusiçerir.

### 2026-09-26 · Bozuk ready satırı yakalandı; DDS UDPv4 kontrollü deneyi

Kullanıcıtanı121/250INCOMPLETE:stdoutFastDDSRTPS_TRANSPORT_SHM segmentcreateerror,251/dev/shmgiriş. diagnostic-report.json/invalid-protocol-lines.json gerçekbytesiçeriyor. BuJSONreadybozulmasınınkaynağı; SHMbirikimialtnedenhipotezi destekkazanmışancakamkanıtlanmışdeğil. ADR-009öneri/doğrulamabekliyor. Mevcutstressscriptine opsiyonel ayrıoutputve DDSenvironment+SHMbefore/afterkanıtı eklendi; py_compilePASS. YeniANA smoke/benchmarkkoduyazılmadı.

Aynıimage’de hedeflistress (NOT_RUN, rebuildgerekmez):

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
$c101Scripts = (Resolve-Path -LiteralPath '.\scripts').Path
$c101Image = (Get-Content -LiteralPath "$c101Evidence\environment-lock.json" -Raw | ConvertFrom-Json).image_id
docker run --rm --cpuset-cpus 0,1 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 --mount "type=bind,source=$c101Evidence,target=/evidence" --mount "type=bind,source=$c101Scripts,target=/diagnostics,readonly" $c101Image python /diagnostics/stress_c101_global_restart.py --evidence /evidence --output /evidence/linux-global-restart-stress-udp --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Source:https://fast-dds.docs.eprosima.com/en/v2.14.6/fastdds/env_vars/env_vars.html .500ölçüm≥100restartve20warmup/launchkapısıaynı. Eski121kanıta dokunulmaz. GeçerseANA testkoduyazmadanönceonay/ajandeğişimindeDUR; geçmezse yenierrorkanıtı üzerinden hedefli tanı. C1-01IN_PROGRESS/T-C00INCOMPLETE,commit/pushyok.

### 2026-09-26 · UDPv4 restartstress500PASS; kullanıcı talimatıyla DUR

Kullanıcı500/500PASS:256restart,258launch,5160warmup;226SUCCESS/274TIMEOUT,worker_start_errornull. /dev/shmönce0,sonra2lttnggiriş;fastrtpssegment0. YerelrawSHA/envlockSHA ve20warmup/launchbağlarıPASS. UDPv4 transport deneyi hedefliduruş sorununu500ölçümde engelledi; tamT-C00veuzunsüreüstünlük iddiası değil. ADR-009hedeflikanıt eklendi; ortakruntimekilidi/ana testkoduhenüzbekliyor.

Kullanıcı sınırına ulaşıldı: yeniANA smoke/benchmarkkoduyazılmadı,koşu verilmedi. experiments/C1-01/MAIN_TEST_HANDOFF.md gerçekdurum,kayıtyolları,kalanişlerveaynıölçümkoşulu gereğini içerir. Kullanıcınınajandeğişimiveaçıkonayıbekleniyor. C1-01IN_PROGRESS,öncekiT-C00INCOMPLETE/4yöntemPARTIAL_VERIFIED; Stage2commit/pushyok.

## 27 Eylül 2026 · Onay sonrası ana koşu hazırlığı

Güncel kayıt: [RUN-20260927-main-preparation.md](RUN-20260927-main-preparation.md). Kullanıcı ana test hazırlığına onay verdi; ortak UDPv4 runtime, kanıt zinciri ve kalan süre aktarımı hazırlanıyor. Önceki onay bekleme notları tarihsel kayıttır. Yeni Linux build/smoke/pilot/full/verify NOT_RUN; C1-01 IN_PROGRESS, commit/push yapılmadı.

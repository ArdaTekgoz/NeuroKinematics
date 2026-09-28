# C1-01 Aşama 1 komut kaydı

**Güncel sıra (28 Eylül 2026):** `udp-v2` prepare/smoke/pilot/full/verify tamamlandı; 600.000 kayıt, beş yöntemde offline verify PASS. C1-01/T-C00 ACCEPTED; [kabul raporu](RUN-20260928-T-C00-acceptance.md). Aşama 2 uygulama commit'i `207bf734de6536e2b590e922930df1547fdc29f1` `origin/main`'e push edildi; büyük raw ve full stderr LOCAL_ONLY. Sonraki Core görevi C1-02 henüz başlamadı.

24 Eylül 2026 · çalışma dizini depo kökü · yalnız inceleme, sözleşme ve F0 regresyonu. Komutlar aşağıda gerçek yürütme sırasına göre özetlendi. Harici solver kurulumu ve T-C00 benchmark komutu **çalıştırılmadı**.

| Sıra | Gerçek komut / çağrı | Sonuç ve kanıt |
|---|---|---|
| 1 | `git branch --show-current; git rev-parse HEAD; git remote -v; git status --porcelain=v1` | `main`, başlangıç `e1971bf8b70154d6f4d882546c20de0eaab2d83d`, iki ilgisiz Word dosyası izlenmiyor. |
| 2 | `rg --files -g AGENTS.md ...` ve görev, roadmap, Core raporu, F0-05/F0-06 dosyaları `Get-Content` | Alt kapsam AGENTS.md yok. Kaynak dosyalar [STAGE1_REVIEW.md](STAGE1_REVIEW.md) içinde bağlandı. |
| 3 | `python -` ile `handoff-inputs.json` içindeki yolların `hashlib.sha256` kontrolü | 15/15 eşleşme; ayrıca [stage1-verification.json](stage1-verification.json). |
| 4 | `Get-Command wsl,ros2,docker,podman,...; wsl --list --verbose; docker version ...; python --version; pixi --version` | WSL komutu var fakat dağıtım yok; Docker/ROS 2 komutu yok; Python 3.11.9, Pixi 0.81.0. Komutun Docker kısmı bulunamadı hatası verdi; ortam tespiti olarak korundu. |
| 5 | `git ls-remote` MoveIt 2, TRAC-IK, pick_ik exact tag/commit | Kimlikler [DEPENDENCIES.md](DEPENDENCIES.md) ve [baseline-config.json](baseline-config.json) içinde. |
| 6 | Resmî kaynaklardan pinned `kdl_kinematics_parameters.yaml`, `trac_ik_kinematics_parameters.yaml`, `pick_ik_parameters.yaml` ve plugin C++ dosyalarını salt okunur HTTP ile okuma | Epsilon, mod, timeout ve thread parametreleri kaynak kaydına işlendi; pluginler çalıştırılmadı. |
| 7 | `python -m py_compile scripts/verify_c101_stage1.py` | PASS. |
| 8 | `python -` ile 15 handoff girdisi + config + F0-05 query listesi için `frozen-hashes.json` üretimi | 17 dosya; config SHA [frozen-hashes.json](frozen-hashes.json) içinde. |
| 9 | `python scripts/verify_c101_stage1.py --output experiments/C1-01/stage1-verification.json` | 52 PASS / 0 FAIL; 15 negatif mutasyon içerir. |
| 10 | `pixi run --locked python -m pytest -q tests/f0_00 tests/f0_01 tests/f0_02 tests/f0_03 tests/f0_04 tests/f0_05 tests/f0_06 --junitxml=experiments/C1-01/stage1-f0-regression.xml` | **FAIL, exit 2:** beş pytest collection hatası; aynı adlı test modülleri/conftest çakıştı. [stdout/stderr](stage1-f0-regression.log) ve [JUnit](stage1-f0-regression.xml) korundu. Test yürütmesi başlamadı. |
| 11 | Her `f0_00`–`f0_06` için ayrı `pixi run --locked python -m pytest -q tests/<name> --junitxml=experiments/C1-01/stage1-<name>.xml` | 6+16+102+159+39+175+26 = **523 PASS / 0 FAIL**; her klasörün `.log` ve `.xml` dosyaları burada. Warnings JUnit `record_property` uyumluluğu hakkında; test hatası yok. |

Sözleşme tekrar kontrolü:

```powershell
python scripts/verify_c101_stage1.py --output experiments/C1-01/stage1-verification.json
```

Gelecek Stage 2 komutları ancak açık onay ve exact ROS ortam lock'u sonrası bu dosyaya gerçek çalıştırıldıkları anda eklenecektir. Henüz kurulum, plugin smoke veya benchmark komutu yoktur.

## 25 Eylül 2026 · Aşama 2 yerel hazırlık kaydı

Kullanıcının Aşama 2 onayı [stage2-approval.json](stage2-approval.json) içinde. Docker Desktop + WSL 2 henüz kullanıcı tarafından doğrulanmadı. Bu bölümdeki komutlar yalnız mevcut Windows/Pixi ortamında **gerçekten çalıştırılan** hazırlık kontrolleridir. Docker build, ROS bağımlılık kilidi, C++ derleme ve beş solver smoke **NOT_RUN**.

| Sıra | Gerçek komut | Sonuç |
|---|---|---|
| 1 | `pixi run --locked python -m py_compile src/neurokinematics/core/contract.py src/neurokinematics/core/worker.py src/neurokinematics/core/dls_worker.py src/neurokinematics/core/results.py src/neurokinematics/core/runner.py src/neurokinematics/core/cli.py` | PASS; çıkış 0. |
| 2 | `pixi run --locked python -m pytest -q tests/c1_01/test_adapter.py` | İlk koşu 6/6 PASS; çıktı konuşma/terminal kaydında. |
| 3 | `pixi run --locked python -m pytest -q tests/c1_01/test_adapter.py` | Genişletilen testlerle 7/7 PASS; çıktı konuşma/terminal kaydında. |
| 4 | `pixi run --locked python -m pytest -q tests/c1_01/test_adapter.py --junitxml=experiments/C1-01/stage2-adapter-tests.xml` | 7/7 PASS; [JUnit](stage2-adapter-tests.xml). |
| 5 | `python scripts/verify_c101_stage1.py` | 52/52 PASS; 17 Stage 1 hash girdisi korunuyor. |
| 6 | `git diff --check` | Exit 0; Git CRLF dönüşüm uyarısı verdi, whitespace hatası yok. |
| 7 | `pixi run --locked python -m pytest -q tests/f0_05 --junitxml=experiments/C1-01/stage2-f0_05-regression.xml` | 175/175 PASS; 32 eski `record_property`/JUnit uyumluluk uyarısı; [JUnit](stage2-f0_05-regression.xml). |
| 8 | `pixi run --locked python -m py_compile scripts/record_c101_linux_lock.py src/neurokinematics/core/runner.py src/neurokinematics/core/cli.py` | PASS; çıkış 0. |
| 9 | `pixi run --locked python -m pytest -q tests/c1_01/test_adapter.py --junitxml=experiments/C1-01/stage2-adapter-tests.xml` | Geç yanıt adayını saklama testi eklendikten sonra 8/8 PASS; JUnit aynı dosyada son koşuyla güncellendi. |

Elle yapılan ilk DLS IPC denemesi dondurulmuş ilk sorgu ve 50 ms profiliyle Windows/Pixi ortamında hazır yanıtı aldı; worker `TIMEOUT`, ortak durum `TIMEOUT`, bağımsız Profile B geometri `true` döndürdü. Bu yalnız adapter entegrasyon kontrolüdür; Linux/Docker smoke veya T-C00 ölçümü olarak kullanılmaz. Stderr `temp/c101/dls-stderr.log` (Git dışında, geçici); ilk deneme için kalıcı ham JSONL üretilmedi.

Kullanıcının PowerShell kurulum/doğrulama komutlarının çıktıları beklendikten sonra gerçek image digest, apt/ROS paket kapanışı, exact source commitleri, C++ build ve smoke komutları burada tarihli yürütme kaydıyla tutulacak. Başarısız komutlar ve loglar korunacak. Beş kolay smoke sonucu PASS olmadan `benchmark` CLI smoke kapısı tam koşuyu reddeder.

## 26 Eylül 2026 · Kullanıcı ortam doğrulaması ve sonraki build komutu

Kullanıcı tarafından çalıştırılan komutların ekran görüntüsü kanıtı [Docker version](user-docker-version-20260926.png) ve [WSL/container](user-environment-20260926.png) olarak korundu. Bunlar Codex tarafından uzaktan çalıştırılmadı.

| Kullanıcının gerçek komutu | Görülen sonuç |
|---|---|
| `Get-Command docker -ErrorAction SilentlyContinue`; `docker version` | CLI bulunuyor; Docker Desktop 4.92.0; client/server Engine 29.8.0; server `linux/amd64`, context `desktop-linux`. |
| `wsl --version`; `wsl --list --verbose` | WSL 2.7.14.0; `docker-desktop` Running, WSL version 2. |
| `docker run --rm hello-world` | `Hello from Docker!`; indirme ve container çalıştırması başarılı. |
| `docker run --rm --platform linux/amd64 ubuntu:24.04 sh -c 'cat /etc/os-release; uname -m'` | Ubuntu 24.04.5 LTS / `VERSION_ID=24.04`; `x86_64`. |

**Sonraki komut: hazırlandı, NOT_RUN.** PowerShell'de depo kökünden çalıştırılır:

```powershell
Set-Location -LiteralPath 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c101_docker.ps1
```

Bu komutun gerçek adımları:

1. Dondurulmuş query dosyasının SHA-256 kontrolü ve Linux x86_64 Docker server kontrolü.
2. `docker pull --platform linux/amd64 ros:jazzy-ros-base-noble`; `docker image inspect` ile alınan immutable `ros@sha256:...` değeri.
3. `docker build --platform linux/amd64 --progress plain --build-arg ROS_BASE_IMAGE=<gerçek-digest> -f experiments/C1-01/Dockerfile -t neurokinematics-c101:stage2 .`.
4. Build içinde apt simulation ile exact paket/dependency revizyonlarını önce kaydet, sonra `package=version` biçiminde kur; MoveIt/TRAC-IK/pick_ik kaynaklarını Stage 1 commitlerinde checkout et; `colcon build --packages-up-to c101_moveit_worker moveit_kinematics trac_ik_kinematics_plugin pick_ik --parallel-workers 2 --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF`.
5. Başarılı build sonrası image inspection ve build kanıtlarını dışarı çıkar; `docker run --rm --cpuset-cpus 0,1 --mount type=bind,source=<C1-01-kanıt-dizini>,target=/evidence <image> python scripts/record_c101_linux_lock.py --external-root /opt/c101/external --image-id <gerçek-image-id> --output /evidence/environment-lock.json`.

Kanıt hedefleri: `docker-base-pull.log`, `docker-base-inspect.json`, `docker-build.log`, `docker-image-inspect.json`, `docker-build-evidence/`, `docker-environment-lock.log`, `environment-lock.json`. Indirilen `.deb` dosyaları image'in `/var/cache/apt/archives` dizininde tutulur; byte hashleri build kanıtındadır. Source build ve dependency closure henüz doğrulanmadı; herhangi bir hata scripti durdurur, log korunur. Bu script **smoke veya tam benchmark başlatmaz**.

Hazırlık kontrolleri: PowerShell Parser `build_c101_docker.ps1` için çıkış 0 / syntax PASS; `pixi run --locked python -m py_compile scripts/c101_install_apt.py scripts/record_c101_linux_lock.py` çıkış 0. Bash/Dockerfile/C++ gerçek Linux build doğrulaması NOT_RUN.

### Build/lock sonrası Linux test ve solver smoke komutları · NOT_RUN

Aşağıdaki komutlar mevcut CLI yollarıyla hazırdır. **Şimdi çalıştırılmayacak:** önce başarılı build ve `environment-lock.json` çıktısı incelenecek, ardından Linux adapter testleri geçecek.

```powershell
$c101Evidence = (Resolve-Path -LiteralPath '.\experiments\C1-01').Path
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 neurokinematics-c101:stage2 python -m pytest -q tests/c1_01/test_adapter.py --junitxml=/evidence/linux-adapter-tests.xml
```

Adapter testi PASS sonrası:

```powershell
docker run --rm --cpuset-cpus 0,1 --mount "type=bind,source=$c101Evidence,target=/evidence" neurokinematics-c101:stage2 python -m neurokinematics.core.cli smoke --output /evidence/linux-smoke --external-worker /opt/c101/ws/install/c101_moveit_worker/lib/c101_moveit_worker/c101_moveit_worker
```

Bu smoke 50 ms profiliyle her solverda dondurulmuş ilk dört `main/local`, ilk iki `boundary/local`, ilk iki `singularity/local` sorguyu ölçer; her yöntem önce ilk 20 frozen sorguyla ısınır. Beş yöntem ayrı kimlikle çalışır; kolay main sorgulardan en az birinin bağımsız Profile B deadline başarısı gerekir. Sınır/tekillikte matematiksel çözümsüz dönüş saklanır, process/kurulum/adapter/geçersiz-output/validator hatası smoke kapısını düşürür. Geç sonlu yanıt 20 ms sınırlı toplama penceresinde alınabilirse TIMEOUT olarak saklanıp FK doğrulanır; bütün süreler toplamda kalır. Bu ilk entegrasyon smoke'u tam T-C00 veya küçük 10/50 ms uçtan uca kabulün yerine geçmez. Diğer negatif/F0 kontrolleri ve küçük uçtan uca smoke sonrasında tam benchmarka geçilir; beş kolay smoke PASS olmadan tam benchmark komutu verilmeyecek.

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

## 27 Eylül 2026 · Ortak UDPv4 ana koşu

Ana test hazırlığı kullanıcı tarafından onaylandı. Bu bölüm önceki onay bekleme notlarını günceller. Ayrıntılar: [çalışma kaydı](RUN-20260927-main-preparation.md). Komutları sırayla, her adımın çıktısı incelendikten sonra çalıştırın. Aşağıdaki Docker adımları henüz **NOT_RUN**.

### 1. Yerel worker build ve bağımlılık denetimi

Kalan bütçe protokolü C++ worker'da değişti. Mevcut exact image üzerinden yalnız `c101_moveit_worker` derlenir; ağ kapalıdır, apt veya Pixi paket güncellemesi yapılmaz. Eski image, lock ve sonuçlar korunur. Yeni kanıt `runtime-build-v1/` dizinine yazılır; mevcut çıktı dizini veya image etiketi yeniden kullanılmaz.

```powershell
Set-Location -LiteralPath 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c101_runtime.ps1
```

Başarıda `RUNTIME BUILD COMPLETE` ve yeni lock yolu yazılır. Derleme ve gerçek paket/kaynak denetimi başarılı olmadan sonraki adıma geçilmez. Bir deneme başarısız olursa kanıtı silmeden çıktıyı inceleyin; gerekirse yeni `-BuildName runtime-build-v2` ve prepare için buna karşılık gelen `-EnvironmentLock` kullanılır.

27 Eylül kullanıcı denemesi Docker komut yolu çözümlemesinde durdu; build başlamadı. İki eşleşen `docker` yolu yerine tek `docker.exe` seçimi düzeltildi ve Windows PowerShell 5.1 kontrolü PASS. `runtime-build-v1/` oluşmadığı için bu hata sonrası aynı komut ve varsayılan BuildName yeniden kullanılabilir. Kanıt: [docker-command-resolution-check.json](docker-command-resolution-check.json).

Sonraki kullanıcı denemesinde worker build PASS: 1 paket, 22,9 saniye; yeni image `sha256:00a76905d283882ca2fab1d3093c6ebea635a48e50bf3a1d5bffa52d7d1b74d7`. Build içi 691 paket/Pixi/kaynak ve input manifest denetimi PASS. Ortam lock çağrısında inherited entrypoint `pixi run --locked` yerel editable paketi yeniden kurmaya çalıştı; `--network none` nedeniyle Hatchling indirmesi başarısız oldu. `--locked` kurulumun kapalı olduğunu ifade etmez; `PIXI_NO_INSTALL=true` ortamı yeniden kurmayı engeller. Bu seçenek build lock ve bütün session çağrılarına eklendi, gerçek Pixi 0.81.0 üzerinde yerel kontrol PASS. [Resmî Pixi run seçenekleri](https://pixi.prefix.dev/latest/reference/cli/pixi/run/).

**Mevcut durumda sonraki komut (henüz NOT_RUN):**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_c101_runtime.ps1 -ResumeLock
```

Bu çağrı kayıtlı immutable image kimliğini ve temel lock bağını doğrular; mevcut image üzerinde yalnız ortam lock ve son bağımlılık/input audit çağrılarını çalıştırır. Yeni image oluşturmaz. Önceki hata logu korunur; yeni denemenin logu ve aday lock'u benzersiz isim alır. Audit PASS sonrasında `environment-lock.json` yayımlanır. Çıktısı incelendikten sonra prepare adımına geçilir.

### 2. Yeni ortamda kritik regresyon ve protokol kontrolü

`-ResumeLock` kullanıcı tarafından başarıyla çalıştırıldı. Lock/image/input bağı yerelde salt okunur kontrolle PASS: [lock-binding-verification.json](runtime-build-v1/lock-binding-verification.json). Yeni lock SHA `9369f45ba52acd6962244bf56ced425c0d895e00771bb5c672895196c6e66990`, image `00a76905...`, CPU 0/1, 691 paket. Build/lock adımı tamamlandı; aşağıdaki prepare çağrısı henüz NOT_RUN.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage prepare -SessionName udp-v1
```

Prepare, kaynak/test/script kopyaları ve image/runtime kilidi oluşturur. Linux kritik regresyonunun toplanan test kimlikleri JUnit ile eşleştirilir; ADR-008 tarihsel Windows residual testi dışında dışlama yapılmaz. Beş gerçek worker'a süresi geçmiş istek gönderilerek solver çağrısı başlamadan TIMEOUT döndüğü ayrıca kontrol edilir. Bu protokol kontrolü benchmark değildir. Kanıt: `udp-v1/prepare/gate.json`, `regression.xml`, `protocol/expired-request-probe.json`.

İlk prepare kullanıcı çağrısı PowerShell parametre/JSON değişken adı çakışmasında Docker çağrısından önce durdu. `$EnvironmentLock` ile `$environmentLock` aynı değişkendi; parsed kayıt `$environmentRecord` olarak düzeltildi. Lock dosyası geçerli ve session dizini oluşmadı; aynı prepare komutu yeniden kullanılabilir. Kalıcı Windows mock launcher akış testi 4 PASS: `tests/c1_01/check_session_launcher.ps1`, [kanıt](session-launcher-flow-check.json).

### 3. Beş solver smoke

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage smoke -SessionName udp-v1
```

8 frozen sorgu × 5 yöntem = 40 ölçüm, 50 ms; her worker için 20 ısınma. Beş yöntemin her birinde kolay örnekte bağımsız Profile B deadline başarısı ve altyapı hatası olmaması gerekir. Ham sıra, query/config bağı ve FK kontrolü uygulanır.

### 4. Küçük pilot

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage pilot -SessionName udp-v1
```

Her subset × başlangıç grubundan ilk 2 frozen sorgu: toplam 12 × 2 bütçe × 5 yöntem = 120 ölçüm. Smoke ham kanıtı tekrar doğrulanır. Pilot sonrası yeniden başlatma/ısınma maliyeti incelenir. Eski global stresin kaba doğrusal ölçeklemesi yaklaşık 18 saat verir; yeni tam koşu için süre garantisi değildir. Yeni pilot ve ilerleme çıktısından güncel tahmin yapılacak.

### 5. Tam T-C00

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage full -SessionName udp-v1
```

12.000 × 2 bütçe × 5 geçiş × 5 yöntem = 600.000 ölçüm. Önceki image'deki dört yöntemin sonuçları eklenmez. Her aşamada aynı runtime kilidi ve önceki kapıların hashleri zorunludur. Bitiş durumu `MEASURED_UNVERIFIED` olur.

Bilgisayar prize bağlı, Docker/WSL açık ve uyku kapalı olsun; koşu sırasında başka benchmark veya ağır iş başlatmayın, Docker CPU/RAM ayarlarını değiştirmeyin. Yöntemler tek süreç içinde sırayla koşar; yeni launcher aynı container adı ve ortak kilitle eşzamanlı ikinci oturumu engeller.

İkinci PowerShell'den ilerleme, doğru dizine geçtikten sonra tek satırla görüntülenebilir:

```powershell
Set-Location -LiteralPath 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main'
docker logs --tail 15 c101-main-session
```

Canlı akış için `docker logs --follow --tail 15 c101-main-session`. Bu ikinci pencerede Ctrl+C log izlemeyi kapatır. İlerleme ayrıca `udp-v1/full/progress.json` içinde her 1.000 kayıtta ve yöntem bitiminde güncellenir. Uzun warmup/restart aralıklarında sayaç ilerlemeyebilir. Container bittiğinde `--rm` nedeniyle Docker logları yerine `udp-v1-full-*.log` dosyası kullanılır.

### 6. Bağımsız doğrulama ve özet

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage verify -SessionName udp-v1
```

600.000 satırın sırası, hashleri, kimlikleri, bağımsız FK, limit ve deadline sonuçları kontrol edilir. Genel/subset/local-wide/deadline ve geçiş kırılımları; bütün ve başarılı örnek süreleri ayrı raporlanır. Fatal altyapı hatası varsa doğrulama kapısı FAIL olur. Sayısal TIMEOUT, çözümsüzlük veya limit hataları ham kayıtlarda korunur; PASS yüzde 100 çözüm başarısı anlamına gelmez.

### 7. Görev kabulü ve Aşama 2 commit/push

Yeni sonuçlar incelenip RUN_REPORT, STATUS, TRACEABILITY, görev ve roadmap kapanışı tutarlı hale getirildikten sonra C1-01 kapsamı commit/push yapılır. Büyük full JSONL dosyaları Git dışında kalır; özetlerde hash/bayt/satır/yerel saklama kaydı bulunur ve kalıcı saklama kapanışta kontrol edilir. Kullanıcının ilgisiz Word dosyaları kapsama alınmaz. Bu adımlar tamamlanana kadar C1-01 IN_PROGRESS kalır.

## 27 Eylül 2026 · ADR-010 sonrası geçerli session: udp-v2

Yukarıdaki `udp-v1` komutları önceki hazırlık ve başarısız smoke denemelerine aittir. Bundan sonraki bütün aşamalar `udp-v2` ile yürütülür. Aynı immutable native image kullanılır; yeniden build/lock kurtarma yapılmaz. Yeni launcher bütün aşamalara `--memory 8g --memory-swap 8g` uygular; cgroup v2 gerçek tavanı kontrol edilir. Host MemTotal/MemAvailable aşama gate'lerinde gözlem olarak kaydedilir. [ADR-010](../../docs/adr/ADR-010-c101-bellek-tavani.md).

Şimdi kullanıcı tarafından çalıştırılacak komut (NOT_RUN):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage prepare -SessionName udp-v2
```

Beklenen kritik regresyon: 264 PASS / 1 ADR-008 deselected; beş gerçek worker expired-request probe PASS. Çıktı incelendikten sonra `-Stage smoke -SessionName udp-v2`, ardından aynı session pilot → full → verify → kabul/commit/push sırası uygulanır. Yeni ana ölçüm komutları henüz çalıştırılmadı.

Güncelleme: prepare ve smoke kullanıcı koşusunda PASS. Smoke 40/40 SUCCESS, yöntem başına 20 warmup / bir launch / sıfır restart. Kaydedilmiş gate/ham dosya hashleri ve runtime bağı kontrolü: [udp-v2-smoke-evidence-check.json](udp-v2-smoke-evidence-check.json). Sıradaki pilot komutu (NOT_RUN):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage pilot -SessionName udp-v2
```

Beklenen 12 sorgu × 2 deadline × 5 yöntem = 120 ölçüm, yöntem başına 24 kayıt. Pilot tamamlanınca fatal hata, restart/warmup maliyeti ve süre incelenerek tam T-C00 çağrısına geçilir.

Pilot kullanıcı koşusu PASS; hash/order/status/warmup bağı kontrolü [udp-v2-pilot-evidence-check.json](udp-v2-pilot-evidence-check.json). Global 13 launch / 260 warmup / 11 restart; 13,58 saniyenin 11,30 saniyesi warmup. Doğrusal 120.000 kayıt ölçeklemesi yaklaşık 19 saat verir; küçük alt kümeden güvenilir ETA çıkarılamaz.

Sıradaki tam T-C00 komutu (NOT_RUN; 600.000 ölçüm):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage full -SessionName udp-v2
```

Prize bağlı, uyku kapalı, Docker/WSL açık; koşu sırasında ağır başka iş veya Docker kaynak ayarı değişikliği yapılmaz. İkinci PowerShell'de `docker logs --tail 15 c101-main-session`; stage çıktı/log/gate'leri `udp-v2/full/` ve `udp-v2-full-*.log` konumunda. Bitişte MEASURED_UNVERIFIED beklenir; verify ve kabul henüz yapılmış olmaz.

28 Eylül güncellemesi: full koşu **600.000 kayıt / beş yöntem × 120.000**, MEASURED_UNVERIFIED. SHA/bayt/satır/önceki kapı bağları [udp-v2-full-evidence-check.json](udp-v2-full-evidence-check.json) ile salt okunur PASS; bağımsız satır/FK doğrulaması bu denetimde yapılmadı. Global 60.227 launch / 60.225 restart / 1.204.540 warmup, tam koşu 18,47 saat. Sıradaki komut:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_c101_session.ps1 -Stage verify -SessionName udp-v2
```

Bu adım her ham satırı bağımsız doğrular ve beş summary üretir. PASS olmadan C1-01 kabul veya commit/push yapılmaz.

28 Eylül güncellemesi: kullanıcı verify koşusu PASS, beş yöntem 120.000'er kayıt ve fatal altyapı hatası 0. Full/verify gate SHA ve beş summary bağı [udp-v2-verify-evidence-check.json](udp-v2-verify-evidence-check.json) ile kontrol edildi. REQ-C01/T-C00 kabul kararı ve saklama sınırı [kabul kaydında](RUN-20260928-T-C00-acceptance.md). Stage 2 commit/push bu kayıtlar tamamlandıktan sonra yapılır.

Yerel çalıştırılan kontroller: `PIXI_NO_INSTALL=true; pixi run --locked python -m pytest -q tests/c1_01/test_session_gates.py ... --junitxml=experiments/C1-01/memory-policy-session-tests.xml`: 29 PASS; `powershell -NoProfile -ExecutionPolicy Bypass -File .\tests\c1_01\check_session_launcher.ps1`: dört mock senaryo PASS, fixed-memory argüman kontrolü dahil, gerçek Docker yok; `python scripts/verify_c101_stage1.py`: 52 PASS / 0 FAIL; `git diff --check`: exit 0.

# F0-06 yeniden üretim komutları

Kanonik host native Windows x64; Pixi 0.81.0. Başlangıç F0-05 kapanışı `e7d211f42496f803688e2a510daca97e102092dc`.

## Tek giriş noktası

Depo kökünde; iki hedef dizin de mevcut olmamalıdır:

```powershell
python scripts/run_f06_clean.py --worktree C:/Users/Public/NeuroKinematics-F006-next --output temp/f06-next
```

Sistem Python yalnız standart kütüphaneli süreç yöneticisidir; bütün proje üretim/test komutları yeni worktree içindeki kilitli Pixi Python ile çalışır. Overlay dosyaları hashlenir, önceki ortam/cache/veri kopyalanmaz. Global indirme cache kullanılır.

## Gerçek kabul komutları

### worktree

```text
git worktree add --detach C:\Users\Public\NeuroKinematics-F006-verification e7d211f42496f803688e2a510daca97e102092dc
```

CWD: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main`
UTC: 2026-09-23T22:11:51.075405+00:00 → 2026-09-23T22:11:51.648591+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\worktree.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\worktree.stderr.log`

### head

```text
git rev-parse HEAD
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:11:51.668018+00:00 → 2026-09-23T22:11:51.710095+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\head.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\head.stderr.log`

### pixi-version

```text
pixi --version
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:11:51.742360+00:00 → 2026-09-23T22:11:51.770171+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\pixi-version.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\pixi-version.stderr.log`

### initial-install

```text
pixi install --locked
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:11:51.779454+00:00 → 2026-09-23T22:12:03.440514+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\initial-install.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\initial-install.stderr.log`

### lock-check

```text
pixi lock --check
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:03.447502+00:00 → 2026-09-23T22:12:03.537392+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\lock-check.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\lock-check.stderr.log`

### runtime

```text
pixi run --locked python -c "import sys,platform,importlib.metadata as m; import numpy,pinocchio; print(sys.executable); print(platform.platform()); print(platform.machine()); print(sys.version); print({x:m.version(x) for x in [\"pytest\",\"hatchling\",\"xacro\",\"PyYAML\"]}); print(numpy.__version__,pinocchio.__version__)"
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:03.539108+00:00 → 2026-09-23T22:12:04.281571+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\runtime.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\runtime.stderr.log`

### reproduction-driver

```text
"C:\Users\Arda TEKGÖZ\AppData\Local\Programs\Python\Python311\python.exe" "C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\scripts\run_f06_reproduction.py" --cwd C:\Users\Public\NeuroKinematics-F006-verification --output "C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical"
```

CWD: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main`
UTC: 2026-09-23T22:12:04.295415+00:00 → 2026-09-23T22:12:37.087769+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\reproduction-driver.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\reproduction-driver.stderr.log`

### reproduce-a

```text
pixi run --locked python scripts/reproduce_f06.py --output experiments/F0-06/reproduction/a
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:04.441362+00:00 → 2026-09-23T22:12:21.143448+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\reproduce-a.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\reproduce-a.stderr.log`

### reproduce-b

```text
pixi run --locked python scripts/reproduce_f06.py --output experiments/F0-06/reproduction/b
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:21.268471+00:00 → 2026-09-23T22:12:36.879300+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\reproduce-b.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\logs\reproduce-b.stderr.log`

### lock

```text
pixi lock --check
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:37.102896+00:00 → 2026-09-23T22:12:37.191994+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\lock.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\lock.stderr.log`

### install

```text
pixi install --locked
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:37.191994+00:00 → 2026-09-23T22:12:37.296649+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\install.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\install.stderr.log`

### environment

```text
pixi run --locked env-check --output "C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\environment.json"
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:37.300306+00:00 → 2026-09-23T22:12:37.817462+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\environment.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\environment.stderr.log`

### f00

```text
pixi run --locked test-f00 "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f00.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:37.829714+00:00 → 2026-09-23T22:12:40.660530+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f00.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f00.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f00.xml`; 6 PASS, 0 FAIL / ERROR / SKIP.

### f01

```text
pixi run --locked test-f01 "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f01.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:40.688749+00:00 → 2026-09-23T22:12:42.471054+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f01.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f01.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f01.xml`; 16 PASS, 0 FAIL / ERROR / SKIP.

### f02

```text
pixi run --locked test-f02 "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f02.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:42.497718+00:00 → 2026-09-23T22:12:47.777575+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f02.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f02.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f02.xml`; 102 PASS, 0 FAIL / ERROR / SKIP.

### f03

```text
pixi run --locked test-f03 "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f03.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:47.798458+00:00 → 2026-09-23T22:12:52.002037+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f03.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f03.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f03.xml`; 159 PASS, 0 FAIL / ERROR / SKIP.

### f04

```text
pixi run --locked test-f04 "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f04.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:52.031313+00:00 → 2026-09-23T22:12:54.006482+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f04.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f04.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f04.xml`; 39 PASS, 0 FAIL / ERROR / SKIP.

### f05-unit

```text
pixi run --locked python -m pytest -q tests/f0_05 --ignore=tests/f0_05/test_tf08.py --ignore=tests/f0_05/test_mutations.py --ignore=tests/f0_05/test_stage2_mutations.py "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f05-unit.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:12:54.035226+00:00 → 2026-09-23T22:13:01.574761+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f05-unit.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f05-unit.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f05-unit.xml`; 127 PASS, 0 FAIL / ERROR / SKIP.

### tf08

```text
pixi run --locked python -m pytest -q tests/f0_05/test_tf08.py "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\tf08.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:13:01.602801+00:00 → 2026-09-23T22:13:02.812206+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\tf08.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\tf08.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\tf08.xml`; 16 PASS, 0 FAIL / ERROR / SKIP.

### f05-mutations

```text
pixi run --locked python -m pytest -q tests/f0_05/test_mutations.py tests/f0_05/test_stage2_mutations.py "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f05-mutations.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:13:02.829133+00:00 → 2026-09-23T22:13:15.885683+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f05-mutations.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f05-mutations.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f05-mutations.xml`; 32 PASS, 0 FAIL / ERROR / SKIP.

### f06

```text
pixi run --locked python -m pytest -q tests/f0_06 "--junitxml=C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f06.xml" -p no:cacheprovider -o junit_family=legacy
```

CWD: `C:\Users\Public\NeuroKinematics-F006-verification`
UTC: 2026-09-23T22:13:15.908204+00:00 → 2026-09-23T22:13:17.172018+00:00
Exit: 0
Stdout: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f06.stdout.log`
Stderr: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\logs\f06.stderr.log`

JUnit: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\experiments\F0-06\canonical\regression\junit\f06.xml`; 26 PASS, 0 FAIL / ERROR / SKIP.

## Bütünlük kontrolü

```powershell
python scripts/audit_f06_history.py
python scripts/finalize_f06.py --implementation-commit <uygulama-commit-sha>
```

F0-04/F0-05 eski acceptance runnerları tarihsel HEAD ve üretim yollarına bağlıdır; F0-06 bunları yeniden yazarak çalıştırmaz. Resmi test komutları aynen çalıştırılır. Depoda birleşik Foundations test komutu tanımlı değildir.

İlk worktree denemelerinin logları `clean-baseline/`, `clean-final/` ve `reproduction/` altında saklanır. Kararı belirleyen tam kayıt `canonical/` altındadır. Başlangıçtaki interaktif bootstrap install zaman damgaları dosyaya alınmadığından final kabul için tamamen yeni ikinci worktree kullanılmıştır. İlgili eski deneme başarılı olsa da kanonik kurulum kanıtı sayılmaz.

Factory/query manifestlerindeki legacy reproduction_command varsayılan üretim yoluna aittir; smoke koşusunu aynen üretmek için yukarıdaki açık F0-06 komutu kullanılır.

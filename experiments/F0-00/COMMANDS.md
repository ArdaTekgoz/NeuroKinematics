# T-F00 komut ve sonuç günlüğü

Tarih: 18 Eylül 2026

Çalışma dizini: `C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main`

## Araç kurulumu ve lock

```powershell
winget install --id prefix-dev.pixi --exact --version 0.81.0 --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
pixi lock
pixi install --locked
```

Sonuçlar:

- Pixi 0.81.0 kullanıcı kapsamına kuruldu.
- `pixi lock`, `win-64` ve `linux-64` için çözümü üretti.
- İlk `pixi install --locked` sırasında `libboost` cache açılımı bir kez I/O uyarısı verdi; Pixi otomatik yeniden denedi ve ortamı kurdu.
- Sonraki `pixi install --locked` sıfır çıkışla `The default environment has been installed.` sonucunu verdi.

## Ortam kontrolü

```powershell
pixi run --locked env-check --output experiments/F0-00/environment.json
```

Çıkış kodu: `0`. Ham sonuç: [`environment.json`](environment.json). Durum: `PASS`.

## İlk pytest denemesi ve düzeltme

İlk komut doğrudan `pytest` console script'ini çağırıyordu:

```powershell
pixi run --locked pytest -q tests/f0_00 --junitxml=experiments/F0-00/pytest-junit.xml
```

Çıkış kodu: `101`. Windows entry-point sarmalayıcısı kullanıcı yolundaki `Ö` karakterini bozuk kodlayarak Python sürecini oluşturamadı. Test gövdesi çalışmadı. Kalıcı görev `python -m pytest` olarak değiştirildi.

## Kabul koşusu

```powershell
pixi lock --check
pixi install --locked
pixi run --locked env-check --output experiments/F0-00/environment.json
pixi run --locked python scripts/verify_frozen_robot_sources.py --output experiments/F0-00/source-verification.json
pixi run --locked test-f00 --junitxml=experiments/F0-00/pytest-junit.xml
```

Zaman aralığı: `2026-09-18T11:48:34.4071413+03:00` – `2026-09-18T11:48:41.8889434+03:00`.

```text
Lock-file was already up-to-date
The default environment has been installed.
......                                                                   [100%]
6 passed in 0.28s
LOCK_CHECK_EXIT=0
LOCKED_INSTALL_EXIT=0
ENV_CHECK_EXIT=0
SOURCE_VERIFY_EXIT=0
PYTEST_EXIT=0
```

Ham pytest sonucu: [`pytest-junit.xml`](pytest-junit.xml).

## Robot kaynak bayt doğrulaması

```powershell
pixi run --locked python scripts/verify_frozen_robot_sources.py --output experiments/F0-00/source-verification.json
```

Exact release commit'indeki yedi kritik dosyanın tamamı beklenen SHA-256 değerleriyle eşleşti. KUKA datasheet URL'si doğrudan istekte PDF yerine dinamik HTML indirme portalına yönlendi; bu HTML'nin değişken hash'i belge hash'i olarak kaydedilmedi. Üretici belgesi kimliği ve indekslenen resmî içerik çapraz kontrol kaynağı olarak, bayt kilidi ise `NOT_AVAILABLE` olarak tutuldu.

## Sistem envanteri

PowerShell `Get-CimInstance Win32_OperatingSystem`, `Win32_Processor`, `Win32_ComputerSystem`, `Win32_VideoController`, `nvidia-smi` ve `wsl.exe --status` kullanıldı. Yapılandırılmış sonuç [`system-inventory.json`](system-inventory.json) içindedir. WSL kurulu değildir; GPU performansı çalıştırılmamıştır.

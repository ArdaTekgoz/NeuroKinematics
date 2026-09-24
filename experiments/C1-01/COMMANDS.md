# C1-01 Aşama 1 komut kaydı

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

# C1-05 Aşama 2 gerçek komut kaydı

8 Ekim 2026 · r1. Exact argv/cwd/UTC/exit/stdout-stderr SHA için `../commands/s2-*/command.json` ve `clean/commands/*/command.json`.

| Komut kimliği | Exit | Başlangıç UTC | Bitiş UTC |
|---|---:|---|---|
| s2-artifact-audit | 0 | 2026-10-08T08:39:26.559267+00:00 | 2026-10-08T08:39:27.043805+00:00 |
| s2-clean-reproduce | 0 | 2026-10-08T14:04:03.572407+00:00 | 2026-10-08T14:06:48.709988+00:00 |
| s2-domain | 0 | 2026-10-08T08:43:35.622311+00:00 | 2026-10-08T08:43:53.399367+00:00 |
| s2-e-c03-2026100201 | 0 | 2026-10-08T08:53:43.757891+00:00 | 2026-10-08T08:58:33.212889+00:00 |
| s2-e-c03-2026100202 | 0 | 2026-10-08T08:58:33.359341+00:00 | 2026-10-08T09:02:58.358540+00:00 |
| s2-e-c03-2026100203 | 0 | 2026-10-08T09:02:58.542836+00:00 | 2026-10-08T09:07:32.188142+00:00 |
| s2-e-c04-2026100201 | 0 | 2026-10-08T09:07:32.329492+00:00 | 2026-10-08T09:08:23.645925+00:00 |
| s2-e-c04-2026100202 | 0 | 2026-10-08T09:08:23.829629+00:00 | 2026-10-08T09:13:08.839055+00:00 |
| s2-e-c04-2026100203 | 0 | 2026-10-08T09:13:08.974740+00:00 | 2026-10-08T09:14:00.671066+00:00 |
| s2-e-c05-2026100201 | 0 | 2026-10-08T09:14:00.819017+00:00 | 2026-10-08T09:14:54.137767+00:00 |
| s2-e-c05-2026100202 | 0 | 2026-10-08T09:14:54.281071+00:00 | 2026-10-08T09:19:38.995815+00:00 |
| s2-e-c05-2026100203 | 0 | 2026-10-08T09:19:39.136230+00:00 | 2026-10-08T09:20:33.515035+00:00 |
| s2-final-stage1-audit | 0 | 2026-10-08T14:14:35.387821+00:00 | 2026-10-08T14:14:38.790020+00:00 |
| s2-fk-full | 0 | 2026-10-08T08:43:45.972296+00:00 | 2026-10-08T08:44:01.902683+00:00 |
| s2-fk-smoke | 0 | 2026-10-08T08:40:43.546276+00:00 | 2026-10-08T08:40:47.419026+00:00 |
| s2-matrix | 0 | 2026-10-08T08:53:41.086039+00:00 | 2026-10-08T09:20:34.008262+00:00 |
| s2-mutants | 1 | 2026-10-08T08:51:15.225802+00:00 | 2026-10-08T08:51:18.535769+00:00 |
| s2-mutants-v2 | 0 | 2026-10-08T08:51:38.996694+00:00 | 2026-10-08T08:51:42.462844+00:00 |
| s2-physics-tests | 0 | 2026-10-08T08:43:53.565509+00:00 | 2026-10-08T08:43:57.529536+00:00 |
| s2-pilot | 0 | 2026-10-08T08:51:57.144456+00:00 | 2026-10-08T08:52:14.531375+00:00 |
| s2-pip-check | 0 | 2026-10-08T08:39:25.480175+00:00 | 2026-10-08T08:39:26.434440+00:00 |
| s2-regression | 0 | 2026-10-08T08:40:33.215512+00:00 | 2026-10-08T08:41:25.125972+00:00 |
| s2-results-audit | 0 | 2026-10-08T14:00:29.959793+00:00 | 2026-10-08T14:00:47.254991+00:00 |
| s2-stage1-audit | 0 | 2026-10-08T08:39:24.778039+00:00 | 2026-10-08T08:39:25.351065+00:00 |
| s2-training-tests | 0 | 2026-10-08T08:51:07.526160+00:00 | 2026-10-08T08:51:14.996276+00:00 |
| s2-witness-record | 0 | 2026-10-08T14:02:59.240213+00:00 | 2026-10-08T14:03:02.887462+00:00 |

Temel gerçek komutlar (depo kökü, her biri c105_command.py ile kaydedildi):

```powershell
.venv/c103/Scripts/python.exe scripts/check_c105_stage1.py --check
pixi run --locked .venv/c103/Scripts/python.exe -m pip check
pixi run --locked .venv/c103/Scripts/python.exe scripts/audit_c103_artifacts.py
pixi run --locked .venv/c103/Scripts/python.exe scripts/validate_c105_domain.py --output experiments/C1-05/stage2/domain
pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_05/test_physics.py --junitxml=experiments/C1-05/stage2/physics-tests.xml
pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_05/test_training.py --junitxml=experiments/C1-05/stage2/training-tests.xml
pixi run --locked .venv/c103/Scripts/python.exe scripts/mutate_c105.py --output experiments/C1-05/stage2/mutations-v2
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c105.py pilot --attempt attempt-001
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c105_matrix.py
pixi run --locked .venv/c103/Scripts/python.exe scripts/audit_c105_results.py --output experiments/C1-05/stage2/results-audit.json
pixi run --locked .venv/c103/Scripts/python.exe scripts/c105_witness.py record --witness experiments/C1-05/stage2/fixed-validation-witness.json
```

Matrix driver her E-C03/04/05 × seed2026100201–03 için ayrı `run_c105.py full
--experiment <E-Cxx> --seed <seed>` başlatır; dokuz child komutun logları vardır.
Mevcut tamamlanmış seed/attempt üzerine tekrar koşma; CLI reddeder. Kaynak,
seçim, budget ve büyük artifact hashleri kontrol edilir. Tekrar eğitim yeni
attempt/ön kayıt/bütçe incelemesi ister; bu kayıt sonuçtan sonra değiştirilemez.

`reproduce_c105.py --checkout <boş managed checkout> --output <yeni evidence>
--witness <fixed-validation-witness.json>` gerçek clean kurulumu yönetir.
`pixi install --locked`, yeni venv, iki hashli pip lock, pip-check, exact artifact
audit ve witness8 komutla tamamlanır. İlk iki .pixi/.venv dizininin yokluğu,
checkout commit ve clean Git başlangıcı `clean/start.json` içindedir. Büyük
weight dosyaları hash kontrolüyle özgün LOCAL_ONLY konumundan okunur; ortam
kopyalanmaz. Dataset shardı veya final test temiz witness için açılmaz.

Başarısız `s2-mutants` exit1 korunur: bir invalid-denominator mutantı geçerli
çıktı yerine IndexError oluşturdu (ERROR). Kaynak uygulama değiştirilmedi;
mutant, yanlış biçimde daraltılmış fakat hesaplanabilir payda üretecek biçimde
düzeltildi. `s2-mutants-v2`:19 KILLED/0 SURVIVED/0 ERROR. İlk hata PASS değildir.

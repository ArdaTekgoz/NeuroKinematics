# C1-04 Aşama 2 gerçek komutlar

3 Ekim 2026 · Windows x64 CPU. Komutlar depo kökünden çalışır. `commands/*/command.json` exact argv, UTC başlangıç/bitiş, exit kodu, thread değişkenleri, stdout/stderr SHA ve **asıl baytların base64 karşılığını** tutar. `.log` çalışma ağacındaki görünür kopyadır; Git metin satır sonu dönüşümünde base64 içerik asıl raw kanıttır. İlk seed ve pilot kayıt aracı eklenmeden çalıştı; onların gerçek sonuçları epoch/summary JSONL/JSON ve bu rapordaki gözlenen terminal exit'leriyle tutulur, stdout byte arşivi olduğu iddia edilmez.

## Giriş ve pilot

```powershell
python scripts/check_c104_stage1.py --check
pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_04/test_contract.py --junitxml=experiments/C1-04/stage2/unit-junit.xml
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c104.py preflight
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c104.py pilot
```

Stage1 62 girdi/12 çıktı PASS. İlk unit 5/5 PASS; sonra genişletilmiş fail-fast test 12/12 PASS (`unit-junit-v2.xml`; test dosyası daha sonra C1-03 ile isim çakışmasını gidermek için `test_c104_contract.py` olarak yeniden adlandırıldı). Preflight 16.800 train/3.600 validation satır ve 15.204/3.249 etiket doğruladı, kaymış 64/64 etiketi bağımsız FK ile reddetti. İlk iki pilot exit1 hazırlık hatası [attempts.json](attempts.json) içinde korunur. Son pilot exit0, iki model T-C03 öğrenme eşiği PASS; 200 epoch, batch64, 64 train ve 32 ayrı validation satırı. Sonuç [pilot-summary.json](pilot-summary.json) ve iki pilot epoch JSONL dosyasında.

## Tam E-C01

```powershell
pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c104.py full --seed 2026100201
python scripts/c104_record_command.py --name seed-2026100202 -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c104.py full --seed 2026100202
python scripts/c104_record_command.py --name seed-2026100203 -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c104.py full --seed 2026100203
python scripts/c104_record_command.py --name audit -- pixi run --locked .venv/c103/Scripts/python.exe scripts/run_c104.py audit
python scripts/c104_record_command.py --name paired-summary-profile -- pixi run --locked .venv/c103/Scripts/python.exe scripts/summarize_c104.py
python scripts/c104_record_command.py --name ambiguity-diagnostic -- pixi run --locked .venv/c103/Scripts/python.exe scripts/diagnose_c104.py
python scripts/c104_record_command.py --name witness-write -- pixi run --locked .venv/c103/Scripts/python.exe scripts/c104_witness.py --write
python scripts/c104_record_command.py --name witness-check -- pixi run --locked .venv/c103/Scripts/python.exe scripts/c104_witness.py --check
```

Üç full komut exit0: her eşli seed 200 epoch, model başına 3.000 optimizer step; aynı etiketli sıra/bütçe. Audit exit0: altı best checkpoint SHA/byte/metadata, epoch JSONL ve 3.600×6 validation satırı doğrulandı. E-C01 summary, ambiguity ve witness komutları exit0. Profil A her model/seed için 0/3.600. İlk `paired-summary` komutu Profil A sütunu eklenmeden önce exit0 ile çalıştı; `paired-summary-profile` aynı ham sonuçlardan bu metriği ekleyen güncel özetini üretti. Eşik/split/model değişmedi.

## Regresyon ve ortam

```powershell
python scripts/c104_record_command.py --name c104-c102-c103-regression-v2 -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/c1_04 tests/c1_02/test_tc07_acceptance.py tests/c1_03 --mutation-output=experiments/C1-04/stage2/c103-mutations-v2 --junitxml=experiments/C1-04/stage2/regression-c104-c102-c103-v2.xml
python scripts/c104_record_command.py --name f01-regression -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/f0_01 --junitxml=experiments/C1-04/stage2/regression-f01.xml
python scripts/c104_record_command.py --name f02-regression -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/f0_02 --junitxml=experiments/C1-04/stage2/regression-f02.xml
python scripts/c104_record_command.py --name f03-regression -- pixi run --locked .venv/c103/Scripts/python.exe -m pytest -q tests/f0_03 --junitxml=experiments/C1-04/stage2/regression-f03.xml
python scripts/c104_record_command.py --name environment -- pixi run --locked .venv/c103/Scripts/python.exe scripts/record_c104_environment.py
```

Birleşik regresyon 129/129, F0-01/02/03 ayrı 16/102/159 PASS. Tarihsel aynı adlı pytest modüllerini tek çağrıya koyan iki önceki toplama denemesi exit1 ile `commands/` altında korunur; sayısal test başarısızlığı değildir. Uyarılar JUnit/terminal loglarında görünür. Ortam kayıtları [environment.json](environment.json) dosyasında.

## 3 Ekim 2026 · Temiz checkout ve kapanış

Uygulama commit'i `7fde45052a5fe3ba3f31dd832541595b7c02313d` Codex yönetimli `c104-clean-reproduction` worktree'sine alındı. Aşağıdaki sekiz komutun exact argv/cwd/UTC/exit ve ham çıktısı `commands/clean-*/command.json` altındadır. Kayıt aracı asıl depodan `--cwd <temiz checkout>` ile çalıştırıldı.

```powershell
pixi install --locked
pixi run --locked python -m venv --system-site-packages .venv/c104-clean
pixi run --locked .venv/c104-clean/Scripts/python.exe -m pip install --ignore-installed --require-hashes --no-deps -r experiments/C1-03/requirements-win-cpu.lock
pixi run --locked .venv/c104-clean/Scripts/python.exe -m pip install --ignore-installed --require-hashes --no-deps -r experiments/C1-03/stage2/runtime-supplement.lock
pixi run --locked .venv/c104-clean/Scripts/python.exe -m pip check
pixi run --locked .venv/c104-clean/Scripts/python.exe scripts/c104_witness.py --check
git rev-parse HEAD
git status --short
```

Sekizi de exit0; temiz HEAD beklenen commit, `git status --short` boş. Witness 12 frozen Stage1 çıktı hashini, altı yerel checkpoint SHA/byte kimliğini ve her checkpoint için 10 sabit validation çıkarımı ile bağımsız FK'yi doğruladı: en büyük q ve FK matris elemanı farkı `0`. Taze checkout'ta yerel C1-02 shardları yoktur; tam eğitim orada yeniden koşulmadı. Model ağırlıkları asıl depodaki `LOCAL_ONLY` yolundan okundu.

Asıl depoda aşağıdaki kapanış komutları çalıştırıldı:

```powershell
python scripts/finalize_c104.py --write
python scripts/finalize_c104.py --check
python scripts/manifest_c104.py --write
python scripts/manifest_c104.py --check
```

`finalize_c104.py --write` ve `--check` PASS: 23 komutun raw stdout/stderr SHA'sı, altı koşunun paydaları ve doğrudan IK NO-GO kararı [acceptance.json](acceptance.json) kaydında. `manifest_c104.py --write` ve `--check` PASS: 170 dosyanın kanonik LF hash envanteri [evidence-manifest.json](evidence-manifest.json) ve [SHA256SUMS](SHA256SUMS) içinde.

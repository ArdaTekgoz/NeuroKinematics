# F0-04 çalıştırılan komutlar

21 Eylül 2026 kapanış koşusu `pixi run --locked python scripts/run_f04_acceptance.py`
ile başlatıldı. Aşağıdaki 14 alt komutun tamamı exit `0` verdi; UTC başlangıç,
bitiş, stdout, stderr ve gerçek exit kodları `commands.json` içindedir.

1. `pixi lock --check`
2. `pixi install --locked`
3. `pixi run --locked test-f00 --junitxml=experiments/F0-04/f00-junit.xml`
4. `pixi run --locked test-f01 --junitxml=experiments/F0-04/f01-junit.xml`
5. `pixi run --locked test-f02 --junitxml=experiments/F0-04/f02-junit.xml -o junit_family=legacy`
6. `pixi run --locked test-f03 --junitxml=experiments/F0-04/f03-junit.xml -o junit_family=legacy`
7. `pixi run --locked python -m pytest -q tests/f0_04/test_unit.py --junitxml=experiments/F0-04/f04-unit-junit.xml`
8. `pixi run --locked python -m pytest -q tests/f0_04/test_tf05_determinism.py --junitxml=experiments/F0-04/tf05-junit.xml`
9. `pixi run --locked python -m pytest -q tests/f0_04/test_tf06_split.py --junitxml=experiments/F0-04/tf06-junit.xml`
10. `pixi run --locked python -m pytest -q tests/f0_04/test_tf07_accuracy_coverage.py --junitxml=experiments/F0-04/tf07-junit.xml`
11. `pixi run --locked python -m pytest -q tests/f0_04/test_mutations.py --junitxml=experiments/F0-04/mutation-junit.xml`
12. `pixi run --locked python -m neurokinematics.data.cli generate --output data/generated/F0-04/run-a --evidence experiments/F0-04`
13. `pixi run --locked python -m neurokinematics.data.cli generate --output data/generated/F0-04/run-b`
14. `pixi run --locked python -m neurokinematics.data.cli verify --output data/generated/F0-04/run-a --manifest experiments/F0-04/dataset-manifest.json`

Son doğrulama: `pixi run --locked python scripts/run_f04_acceptance.py --verify-only`;
39 hashli dosya doğrulandı. Linux komutu çalıştırılmadı.

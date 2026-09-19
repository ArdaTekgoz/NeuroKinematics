# F0-03 gerçek komut kaydı

19 Eylül 2026. Tam argüman, stdout/stderr, UTC zaman ve gerçek exit kodları
`commands.json` içindedir. Frozen runner bütün komutlarda exit 0 aldı.

```powershell
pixi run --locked python scripts/run_f03_acceptance.py
```

Runner sırası: `pixi lock --check`, `pixi install --locked`, ardından locked
`test-f00`, `verify-robot-a`, `test-f01`, `test-f02`, `test-f03-unit`,
`validate-jacobian`, `validate-metrics`, `test-f03`. JUnit/output argümanları
commands.json içindedir. verify-robot-a çıktısı F0-03'e yönlendirilir; eski
F0-01 raporu yeniden yazılmaz. test-f02 tam 10000-q testini içerir.

Geliştirmede gerçekten çalıştırılan ek komutlar:

```powershell
pixi run --locked test-f03-unit --junitxml=experiments/F0-03/development-unit-junit.xml -o junit_family=legacy
pixi run --locked test-f03-unit --junitxml=experiments/F0-03/development-fixed-unit-junit.xml -o junit_family=legacy
pixi run --locked validate-jacobian
pixi run --locked validate-metrics
```

İlk unit exit 1, sonraki komutlar exit 0. Geliştirme özetleri
development-attempts.json; ilk başarısız JUnit korunur. `inspect-jacobian`
tanımlıdır; inspect işlevi validate-jacobian içinde çalışır, ayrı Pixi komutu
çalıştırılmış sayılmaz.

Kanıtı yeniden üretmeden doğrulama:

```powershell
pixi run --locked python scripts/run_f03_acceptance.py --verify-only
```

Commitlenmiş kanıta dokunmadan push öncesi tam tekrar:

```powershell
pixi run --locked python scripts/run_f03_acceptance.py --output temp/f03-prepush
```

Bu son komutun gerçek sonucu push öncesi terminal kaydı ve temp/f03-prepush
altında saklanır. Sayısal JSON karşılaştırması ayrıca yapılır. F0-04 çalıştırılmaz.

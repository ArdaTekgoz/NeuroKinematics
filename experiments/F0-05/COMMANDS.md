# F0-05 gerçek komut kaydı

`commands.json`, tam kabul runner'ının çalıştırdığı 16 alt komutu, UTC başlangıç
ve bitiş zamanlarını, çıkış kodlarını ve çıktıları saklar. Aşağıdaki çağrı
23 Eylül 2026'da `main` üzerinde `84adc24a03f619d7079e4e3900b8f51ab2591ef5`
HEAD'iyle yürütüldü ve PASS verdi:

```powershell
pixi run --locked accept-f05
```

Runner sırasıyla Git/immutable girdi ve F0-04 preflight'i, `pixi lock --check`,
`pixi install --locked`, F0-00–F0-04 regresyonları, F0-05 unit/T-F08/mutasyon
testleri, iki query üretimi, bağımsız query doğrulaması, tam benchmark,
bağımsız JSONL doğrulaması, özetleme ve kanıt hash denetimini yaptı. İlk gerçek
hatada durur. `commands.json` bu alt komutların kaydıdır; preflight ve Python
içi hesaplar `preflight.json`, manifestler ve özetlerde saklanır.

Analitik dış erişim sınıflandırması son kez sıkılaştırıldıktan sonra şu hedefli
komut ayrıca yürütüldü:

```powershell
pixi run --locked python -m pytest -q tests/f0_05/test_tf08.py --junitxml=experiments/F0-05/tf08-junit.xml -o junit_family=legacy
```

Sonuç 16/16 PASS. `failure-cases.json` aynı 120.000 satırlık, SHA-256'sı
değişmeyen sonuç dosyasından yeni kanıt doğrulayıcısıyla tekrar üretildi. Tam
benchmark, bu kanıt raporlama değişikliği için yeniden çalıştırılmadı; ölçüm
satırları veya dört frozen dosya değişmedi. Son kanıt bütünlüğü için:

```powershell
pixi run --locked verify-f05-evidence
```

Yeniden üretim ve ayrı kanıt kökü için:

```powershell
pixi run --locked accept-f05 --output experiments/F0-05/reproduction --generated-root data/generated/F0-05/reproduction
```

Tam tekrar yalnız başlangıç commit'i ve `origin/main` preflight şartı
karşılandığında çalışır; kapanış commitlerinden sonra yeni bir checkout'ta
runner'ın bu başlangıç denetimi tarihsel sabit nedeniyle uygun olmayabilir.
Mevcut kanıtı ve Git dışındaki tam JSONL dosyalarını tekrar çalıştırmadan
denetlemek için `verify-f05-evidence` kullanılır. `query-manifest.json` ve
`result-manifest.json` tekil üretim komutlarını ve hashleri içerir.

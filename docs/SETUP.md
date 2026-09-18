# Geliştirme ortamı kurulumu

Bu belge F0-00 için tekrar üretilebilir Python ortamını tanımlar. Kanonik
geliştirme ve kabul hostu **native Windows 11 x64**'tır. WSL kanonik ortam
değildir. `linux-64`, aynı bağımlılık grafiğinin çözülebildiğini lock dosyasında
denetlemek için ikincil platformdur; Linux testleri çalıştırılmadan doğrulanmış
sayılmaz.

## Dondurulan araç ve ortam sözleşmesi

| Alan | Karar |
|---|---|
| Ortam yöneticisi | Pixi 0.81.0 |
| Paket kanalı | Yalnız `conda-forge` |
| Lock platformları | `win-64`, `linux-64` |
| Python | CPython 3.12.*; gerçek yapı `pixi.lock` ile sabitlenir |
| Referans kinematik | Pinocchio 4.1.0 |
| Sayısal/test bağımlılıkları | NumPy 2.*, pytest 8.*; gerçek yapılar `pixi.lock` ile sabitlenir |
| Paket yerleşimi | `src/` yerleşimi, editable yerel paket, Hatchling 1.27.0 |

`.python-version` yalnızca editörlerin Python 3.12 serisini seçmesine yardım eder.
Ortamın normatif tanımı `pixi.toml` ve Git'te izlenen `pixi.lock` ikilisidir.

## Native Windows 11 x64 kurulumu

PowerShell'de Pixi'yi exact sürümle kullanıcı kapsamına kurun ve sürümü doğrulayın:

```powershell
winget install --id prefix-dev.pixi --exact --version 0.81.0 --scope user --accept-package-agreements --accept-source-agreements --disable-interactivity
pixi --version
```

Son komut `pixi 0.81.0` göstermelidir. Yeni bir PowerShell oturumu gerekebilir.
Temiz checkout'ta mevcut lock dosyasını değiştirmeden ortamı kurun:

```powershell
pixi install --locked
pixi run --locked env-check
pixi run --locked test-f00
```

`env-check`, T-F00 raporunu JSON olarak standart çıktıya yazar. Kanıt dosyası
için hedef klasörü önce oluşturun; komut eksik üst klasör oluşturmaz:

```powershell
New-Item -ItemType Directory -Force experiments/F0-00 | Out-Null
pixi run --locked env-check --output experiments/F0-00/environment.json
```

Komut ancak Python 3.12, 64 bit işlem, NumPy `float64`, Pinocchio 4.1.0 ve
`pinocchio.SE3.Identity()` smoke kontrolü birlikte geçerse `0` ile çıkar. Rapor
bağımlılık eksikken de JSON üretir ve `FAIL`/sıfır olmayan çıkış verir.

## Lock dosyasını oluşturma veya bilinçli yenileme

`pixi.lock` ilk kez oluşturulurken veya manifest bilinçli değiştirildiğinde:

```powershell
pixi lock
git diff -- pixi.toml pixi.lock
pixi install --locked
pixi run --locked test-f00
```

`pixi lock`, manifestteki iki platformu da çözer. Bunun Windows'ta başarılı olması
Linux ikililerinin çalıştırıldığı anlamına gelmez. Bağımlılık güncellemesi,
manifest ve lock farkı birlikte incelenmeden kabul edilmez. Normal kurulum ve CI
akışları `--locked` kullanır; böylece örtük bir yeniden çözüm yapılmaz.

## Linux ikincil platform durumu

Linux için hedef CPython ve paket grafiği aynıdır. `linux-64` lock girdisi
oluşturulacak, fakat F0-00 kapsamında Linux kurulumu ve T-F00 yürütmesi
**ÇALIŞTIRILMADI** olarak kalır. Daha sonra gerçek bir x86-64 Linux hostta doğrulama
yapılırsa kullanılan dağıtım, kernel, CPU ve gerçek komut çıktıları ayrı bir
RUN_REPORT kaydına eklenmelidir.

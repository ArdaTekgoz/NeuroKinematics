# ADR 012 — C1-03 Windows runtime overlay düzeltmesi

Durum: KABUL EDİLEN UYGULAMA KARARI; sayısal kabul ayrı
Tarih: 29 Eylül 2026 · Core C1-03 / REQ-C02

## Bağlam

Onaylı Stage1 kurulumu iki sorunu gerçek koşuda yakaladı. Python 3.11 üzerinden
3.12 hedefli metadata çözümü `setuptools; python_version >= "3.12"` koşulunu
atlamıştı. Gerçek 3.12 `pip check` eksik setuptools verdi. Ayrıca conda NumPy
BLAS `libomp.dll` ve Torch `libiomp5md.dll` birlikte OMP Error #15 ile abort etti.
İlk analitik deneme paket kapısı hatasına rağmen aynı kabuk komut dizisinde
başlatılmıştı; exit 3 ile durdu, kabul kanıtı sayılmaz. Ana smoke/full başlamadı.
Komut/loglar `stage2/commands/003`–`007` kimlik önekleriyle korunur.

## Alternatifler

Duplicate OpenMP kontrolünü devre dışı bırakmak doğru sonuç garantisi sağlamaz;
kullanılmadı. Foundations conda paketlerini değiştirmek G0 kilidini bozar.
Ayrı overlay'de aynı NumPy 2.5.3 sürümünün resmi Windows wheel'ini kullanmak
BLAS runtime yüzeyini ayırır; gerçek import/matmul/FK ve regresyon kapısı gerekir.

## Karar ve gerekçe

Yalnız `.venv/c103` içine exact NumPy 2.5.3 wheel ve setuptools 82.0.1 eklenir.
Gerçek Python 3.12 metadata çözümü ve URL/SHA'lar
`experiments/C1-03/stage2/runtime-dependency-resolution.json`, `runtime-pins.json`
ve `runtime-supplement.lock` içinde. Stage1 config, samples ve lock korunur;
bu dosya **runtime protocol r2** ekidir. Torch 2.10.0+cpu, Pinocchio 4.1.0,
Python/NumPy sürümleri ve matematik tolerans/örnekleri değişmez. Runtime import,
pip-check ve analitik smoke geçmeden tam koşu yapılmaz. İkinci fresh ortam aynı
iki lock'u da kurmalıdır; gerekirse bağımsız referans ayrı süreçte tutulabilir.

## Sonuçlar

Foundations Pixi lock/ortamına yazılmaz; venv dışındaki NumPy uninstall edilmez.
Setuptools eksikliği ve ilk abort FAIL girişleri korunur. Her iki ortamda paket
yolu/version/build hashleri kaydedilir, ilgili Foundations regresyonları yeniden
çalıştırılır. Bu ortam düzeltmesi FK sonucu görüp eşik/örnek ayarı değildir.

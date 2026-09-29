# ADR 011 — C1-03 sınırlı Torch zinciri ve ayrı CPU ortamı

Durum: AŞAMA 1 TASARIM KARARI; uygulama açık kullanıcı onayı bekler
Tarih: 29 Eylül 2026
Etkilenen faz/görev: Core v1.0.0 / C1-03 / REQ-C02

## Bağlam

G0 robot, frame, reference ve Pixi lock hashleri değişmez. Torch henüz kurulu
değildir. Diferansiyellenebilir FK, q hesap grafiğini ve iki dtype'ı korumalı;
Pinocchio yalnız bağımsız oracle olarak kalmalıdır.

## Alternatifler

`pytorch-kinematics==0.10.0` MIT ve pinlenebilir wheel sunar; URDF root/end,
batch/device ve autograd API'si vardır. Ek parser, bağımlılıklar ve tensor FK'nin
analitik backward yolu ayrıca doğrulama gerektirir. Küçük Torch kernel mevcut
parserı korur, fixed/revolute kapsamıyla incelemeyi daraltır; yeni matematik
uygulamasının tüm kabul testlerinden geçmesi gerekir.

## Karar ve gerekçe

`native-torch-serial-v1` planı seçildi. Constructor frozen girdileri doğrular,
Torch Rodrigues/origin çarpımları graph içinde hesaplanır. NumPy kardeşi aynı
parserı kullandığından bağımsız referans değildir. Pinocchio 4.1.0 değişmez.
Exact torch 2.10.0+cpu ve sekiz geçişli bağımlılık Windows CPython 3.12 için
[hashli overlay kilidinde](../../experiments/C1-03/requirements-win-cpu.lock).
Runtime kurulumu bu kararın doğrulanmış sonucu değildir.

Foundations pixi.lock değiştirilmez. Aşama 2'de locked Pixi Python üstünde
ayrı venv (`--system-site-packages`) oluşturulur; hashli overlay yalnız oraya
kurulur. Gerçek import/pip-check/versiyon kapısı başarısızsa durulur. İkinci
temiz checkout kendi fresh Pixi env/venv'sini oluşturur; ilk env kopyalanmaz.
CPU kanonik profildir; CUDA/Linux yürütmesi NOT_RUN kalır. Bu karar G0
platformunu veya Core faz kabul kapsamını daraltan bir G1 kararı değildir.

## Sonuçlar

T-C01 float64/32, T-C02 32 iç q, fixture/Jacobian/sensitivity/mutation/regression
ve ikinci temiz kurulum zorunlu. Onay öncesi uygulama/PASS/eğitim yok.
Kernel yetersiz kalırsa yeni ADR/protokol ile aday yeniden değerlendirilebilir;
aynı kabul eşikleri korunur. [Aşama 1 incelemesi](../../experiments/C1-03/STAGE1_REVIEW.md)
API, lisans, hashler ve riskleri ayrıntılandırır.

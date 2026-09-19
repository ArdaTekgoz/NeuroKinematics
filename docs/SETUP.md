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

## F0-01 robot varlığı

F0-01, PyPI `xacro==2.1.1` aracını ve transitif `PyYAML 6.0.3` paketini
`pixi.lock` içinde iki hedef platform için kilitler. Normal üretim ve doğrulama
yalnız locked ortamda yapılır:

```powershell
pixi install --locked
pixi run --locked build-robot-a
pixi run --locked verify-robot-a
pixi run --locked test-f01
```

`build-robot-a`, raw-byte upstream snapshot'ını değiştirmez. Native Windows'ta
ROS kurulumu bulunmadığı için yalnız Xacro `$(find)` include ifadelerini geçici
bir dizindeki kopyalarda mutlak yollara uyarlar; çözülmüş kalıcı URDF'yi,
canonical `robot_spec.json` dosyasını ve `manifest.json` dosyasını üretir.
`verify-robot-a` aynı girdiden yeniden üretim, tüm kayıtlı dosya hashleri, mesh
URI'leri, seri zincir sözleşmesi ve Pinocchio 4.1.0 parse kontrolü birlikte
geçmeden sıfırla çıkmaz.

## F0-02 bağımsız ileri kinematik

F0-02 yeni bağımlılık eklemez; mevcut lock aynen kullanılır. Depo kökünde:

```powershell
pixi run --locked fk-inspect
pixi run --locked test-f02-unit
pixi run --locked validate-fk-small
pixi run --locked validate-fk
pixi run --locked test-f02
```

`fk-inspect` immutable hashleri, sekiz elemanlı base–TCP yolunu ve Pinocchio
joint/frame eşlemelerini gösterir. `validate-fk-small` 1000-q smoke koşusudur;
eşikler geçse bile nihai kabul için `INCONCLUSIVE` yazar. `validate-fk`, önceden
dondurulmuş config ile tam 10000 q kullanır. `test-f02` bu 10000-q hesabını
yeniden çalıştırır; test varsayılan olarak atlanmaz.

Tam kapanış sırası ve kanıt kaydı tek komutla yeniden üretilebilir:

```powershell
pixi run --locked python scripts/run_f02_acceptance.py
```

Bu komut lock/install, F0-00, immutable robot kontrolü, F0-01, F0-02 unit,
10000-q CLI ve tam F0-02 testlerini sırayla çalıştırır; ilk hatada durur.
`experiments/F0-02/` altındaki JSON/JUnit/komut çıktıları ve SHA256SUMS yeniden
yazılır. Config/seed sabittir; örnek matrisi hash'i aynı kalmalıdır. JUnit süreleri,
çalıştırma zamanları ve HEAD kaydı koşuya bağlı olarak değişir. Eski koşuyu
korumak için yeniden çalıştırmadan önce kanıtları Git'te saklayın.

Tek CLI koşusunun kanıtlarını başka bir dizine yazmak için:

```powershell
pixi run --locked validate-fk --output temp/f02-reproduction
```

Servis örneği:

```python
from neurokinematics.kinematics import IndependentFK, load_robot
from neurokinematics.kinematics.pinocchio_fk import PinocchioFK

inputs = load_robot()
custom = IndependentFK(inputs)
reference = PinocchioFK(inputs)
q = [0.0] * 6  # manifest sırası; radyan
T_base_tool0 = custom.forward_kinematics(q)
T_reference = reference.reference_forward_kinematics(q)
```

Sonuç `(4,4)` float64, `T_A_B` sütun-vektör sözleşmesindedir. URDF RPY sırası
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`; joint dönüşümü `origin @ motion`.
Pinocchio referans instance'ı mutable Data tutar; threadler arasında paylaşmayın.
Linux yürütmesi ve Jacobian doğrulaması F0-02 kapsamında yapılmadı.

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


## F0-03 Jacobian ve metrik doğrulaması

Yeni bağımlılık yok; F0-02 API ve immutable robot girdileri korunur.

```powershell
pixi run --locked inspect-jacobian
pixi run --locked validate-jacobian
pixi run --locked validate-metrics
pixi run --locked test-f03-unit
pixi run --locked test-f03
pixi run --locked python scripts/run_f03_acceptance.py
pixi run --locked python scripts/run_f03_acceptance.py --verify-only
```

Tam runner 10 komutu sırayla çalıştırır ve ilk hatada durur. 256 PCG64/20260919
örnek ile 21 elle seçilmiş q, h=1e-5/1e-6/1e-7 için üç yöntem çiftiyle sınanır.
test-f03 bu kabul hesabını atlamaz. T-F04 ayrı validate-metrics/JUnit ile de
doğrulanır. Kanıtlar experiments/F0-03 altında; eski koşuyu korumak için tam
tekrarı `--output temp/f03-reproduction` argümanıyla başka klasöre yazabilirsiniz.
Config ve sayısal sonuçlar sabittir; komut/JUnit zamanları koşuya göre değişir.

```python
from neurokinematics.kinematics import load_robot
from neurokinematics.kinematics.jacobian import IndependentJacobian
from neurokinematics.kinematics.pinocchio_jacobian import PinocchioJacobian
from neurokinematics.kinematics.finite_difference import CentralDifference
from neurokinematics.kinematics.jacobian_validation import characteristic_length
from neurokinematics.kinematics.metrics import singularity_metrics

inputs = load_robot()
q = [0.0] * 6
geometric = IndependentJacobian(inputs).jacobian(q)
reference = PinocchioJacobian(inputs).jacobian(q)
central = CentralDifference(inputs).jacobian(q, h=1e-6)
metrics = singularity_metrics(geometric, characteristic_length())
```

Çıktı 6×6 float64, TCP noktasında ve base eksenlerinde `[linear; angular]`.
Sonlu fark q±h limit dışındaysa açık hata üretir. Norm/tekillik metrikleri fiziksel
robot güvenliği veya çarpışmasızlık kanıtı değildir. F0-04 başlatılmadı.

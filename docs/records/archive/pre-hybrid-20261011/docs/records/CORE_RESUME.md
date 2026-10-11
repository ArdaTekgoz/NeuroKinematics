# Core sonrası ara verme ve geri dönüş paketi

11 Ekim 2026 · Belge r1. En son karar:
[G1_DECISION](../../experiments/C1-07/G1_DECISION.md).
Araştırma modeli kartı: [MODEL_CARD](../../experiments/C1-07/MODEL_CARD.md).
Akademik rapor: [PDF](../publication/core/Core_Negatif_Sonuc_Raporu.pdf).

## Nerede kaldık?

Foundations/G0 tamamlandı. Core deneyleri ve C1-06R tanıları tamamlandı;
ürün hedefi karşılanmadı. H2 reddedildi; yeni C1-06R finali oluşturulmadı.
C1-07 temiz ortam ölçümü ve G1 kararı ayrı kanıtlara bağlıdır. Hybrid ve
Studio başlamadı. Geçmiş tarihli PENDING kayıtlar tarihsel anlık durumdur;
nihai karar için G1_DECISION ve acceptance.json kullanılır.

İlk yeni görev H2-01 / REQ-H01 / T-H01: SolverRequest/Result ve doğru hata
politikasının tanımlanması. Ardından H2-02 bütçeli refinement/restart ve
H2-03 eşli H1 deneyi gelir. Eski launcher ile yeni uzun eğitim başlatma.

## Neler nerede saklanıyor?

| İçerik | Konum | Kurtarma sınırı |
|---|---|---|
| Kod, küçük sonuçlar, raporlar, kaynak hashleri | Git branch `codex/c1-06r` | Push makbuzundaki remote commit ile doğrula |
| Veri, checkpoint, C1-01 raw ve stderr | Desktop/NeuroKinematics-Core-Archive-20261011/research-artifacts.zip | Git dışında; inventory.json ile her dosya SHA doğrulanır |
| Git geçmişinin çevrimdışı kopyası | Aynı klasörde repository.bundle | Bundle sonrası yerel değişiklikleri içermez |
| Bu görev öncesindeki kullanıcı değişiklikleri | Aynı klasörde user-work/ | STATUS/TRACE tam kopyaları ve değişiklik patch'i; otomatik uygulama yapma |
| Ortamlar, paket cache'i ve geçici checkout'lar | Arşivlenmedi | Sürüm/hash kilitlerinden yeniden kur |

Arşiv aynı bilgisayarın aynı diskindedir. Disk kaybına dayanıklı ikinci
aygıt veya uzak büyük-dosya arşivi **NOT_CONFIRMED**. Arşiv klasörünün
tamamını başka bir diske veya seçtiğin kalıcı depoya kopyalamak gerekir;
sonrasında aşağıdaki verify komutunu o konumda çalıştır.

Eski final/sealed dosyalar arşivleme için yalnız opak bayt olarak kopyalanır;
içeriği analiz edilmez, modele verilmez veya yeniden model seçimi için açılmaz.
Bu saklama işlemi yeni bağımsız test sayılmaz.

## Arşivi kontrol etme ve geri yükleme

PowerShell 5.1 yeterlidir; `pwsh` zorunlu değildir. Depo kökünde, Python ile:

```powershell
python scripts/archive_core.py verify --archive 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-Core-Archive-20261011'
```

Başka makinede önce Git'ten branch'i klonla veya bundle ile çevrimdışı klonla.
Model/veri için boş bir `artifact-store` klasörüne geri yükle. Restore hiçbir
mevcut dosyanın üzerine yazmaz; zip üye yollarını ve SHA'ları doğrular:

```powershell
git clone --branch codex/c1-06r 'D:\NeuroKinematics-Core-Archive-20261011\repository.bundle' NeuroKinematics
Set-Location NeuroKinematics
python scripts/archive_core.py restore --archive 'D:\NeuroKinematics-Core-Archive-20261011' --destination 'D:\artifact-store'
```

Bu örnekte D: taşınan yedeğin konumudur; kendi gerçek yolunla değiştir.
Arşivde `.venv` ve `.pixi` bulunmaz.

## Temiz çıkarımı yeniden çalıştırma

Pixi 0.81.0 ve Python erişilebilir olmalı. Yeni boş checkout ve çıktı yolu
kullan; eski kanıtların üzerine yazma. Windows/CUDA overlay seçili decoder
testinde NVIDIA GPU kullanır; yalnız CPU makine bu protokolün tamamını
geçti sayılmaz. CPU witness ise CPU'da çalışır.

```powershell
git clone --no-hardlinks . temp/c107-new-clean
python scripts/reproduce_c107.py --checkout temp/c107-new-clean --output temp/c107-new-evidence --artifact-root 'D:\artifact-store'
```

Bu işlem yeni eğitim yapmaz: hashli altı checkpoint, sabit 48 sorgu,
bağımsız FK ve kritik regresyonlar çalışır. Aynı platform/runtime/thread/
batch sözleşmesinde birebir sonuç beklenir; farklı platformda exact eşliğin
garanti edildiği iddia edilmez. Komut hata verirse logu koru; eşiği genişletme.

## Dönüşte karar sırası

1. Bundle/Git commit, arşiv receipt ve inventory SHA'larını doğrula.
2. G1 kararı, nihai model kartı, negatif sonuç raporu ve ADR-025'i oku.
3. T-C06 replay ile ortamın hâlâ aynı sonuçları verdiğini kontrol et.
4. H2-01 görevini aç; girdiler, deadline, limit, collision NOT_CHECKED,
   invalid ve timeout durumlarını tek doğrulayıcıyla tasarla.
5. H1 için ayrı veri kimliğini, root ayrımını ve erişim protokolünü model
   seçiminden önce dondur. Ana adaylar FK_TANH'ın üç seed'idir; LOCAL_RAW
   yalnız keşifsel yerel karşılaştırmadır.
6. Aynı sayısal motor, CPU/thread, batch 1 ve toplam 10/50 ms bütçelerde
   current/merkez/neural/klasik restart başlangıçlarını karşılaştır.
   Neural giriş hazırlama ve forward maliyeti bütçeye dahildir.
7. Faz raporundaki H1 hedefini koru: P95 toplam sürede en az %20 azalma,
   başarı kaybı en fazla 1 yüzde puanı; eşli %95 güven aralıklarıyla karar.
   En az 10000 sabit sorgu, beş geçiş; tekrarlar bağımsız hedef değildir.
8. H1 faydası yoksa sayısal motor varsayılan kalır. Hata sınıfı başına en
   fazla iki hedefli düzeltme turu, ardından ayrı araştırma kapsamı;
   Studio'nun güvenilir sayısal motorla ilerlemesi korunur.

Collision, fiziksel robot, ONNX, trajectory ve GUI bu kapanışta doğrulanmadı.
Yeni faz, kullanıcı projeye dönüp başlatana kadar NOT_STARTED kalır.

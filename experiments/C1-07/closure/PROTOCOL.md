# T-C06 kapanış tekrar protokolü

11 Ekim 2026. Ön kayıt: ölçümden önce bu kaynak ve commit sabitlenir.
Önce akademik negatif sonuç raporu hazırlanır; ardından kullanıcının yeni
yetkisiyle C1-07/G1 yürütülür. Önceki üç aşama bekleme sınırı bu genişletilmiş
talep için uygulanmaz. Hybrid uygulaması kapsamda değildir.

Altı aday önceki hazırlık manifestinde sabittir; seçim değiştirilmez.
Validation family main/boundary/singularity ×local/wide ilk8'er=48 sorgu;
başarıya göre örnek seçimi yok. Model girdisi yalnız target pose ve current.
CPU/thread1/batch48; her model için q ve bağımsız FK/Profil A-B değerlendirmesi.
Referans çıktı mevcut kilitli ortamda yazılır; temiz Git checkout'ta taze
Pixi + yeni venv, aynı SHA-pinned Torch2.10.0+cu128 kurulur. Environment
kopyalanmaz; paket indirme cache'i yeniden kullanılabilir.

Kabul:6 checkpoint×48=288 çıktı ve metrik birebir; manifest/robot/scaler
hashleri; mevcut temiz testler; bozuk girdi,hash/robot mutantları ve limit
dışı çıktı reddi. Aynı kaynak/runtime/batch için exact eşlik beklenir;
fark varsa tolerans gevşetmek yerine neden incelenir. Baseline/FK/gradyan
kapanış kanıtları korunur, seçili FK/physics/decoder regresyonu yeniden koşar.
Temiz test veri shard'larını açmaz; yalnız portable witness ve hashli
checkpoint dosyalarını external artifact root üzerinden okur.

T-C06 ve kritik regresyonlar geçerse G1 araştırma kapanışı PASS/ACCEPTED
değerlendirilebilir. Direct IK NO_GO, H2 REJECTED, C1-06R ürün NOT_MET ve
yeni final NOT_CREATED görünür kalır. Yazılım hedefi v1.0.0; Foundations
pixi.toml sürümü değişmez; release/tag bu protokolün sonucu değildir.

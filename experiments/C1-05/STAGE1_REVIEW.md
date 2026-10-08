# C1-05 Aşama 1 inceleme paketi

8 Ekim 2026 · Belge r1 · Yazılım hedefi v1.0.0 (sürüm etiketi verilmedi)

**STAGE_1_COMPLETE / T-C04 NOT_RUN / genel görev PARTIAL.** Yeni physics-aware
model eğitilmedi; iyileşme NOT_MEASURED. Bu paket uygulama onayına sunulan deney
sözleşmesidir. C1-04 deneysel baseline COMPLETE / T-C03 PASS / doğrudan IK NO-GO
değişmez. [Kaynak görev](SOURCE_REQUEST.md) iki aşamalı uygulama ister.

## Doğrulanan girdiler

Fetch sonrası `HEAD=main=origin/main=404268b20b584aac30f836812b2d574667766f77`.
Başlangıçtaki kullanıcı değişiklikleri [workspace-start](workspace-start.json)
ile kayıtlı; STATUS/TRACEABILITY içindeki denetim ekleri commit kapsamına alınmaz.
G0 robot/TCP, joint1–6, m/rad/wxyz, C1-02 split/train-only scaler, C1-03 kernel ve
kilitli runtime kimlikleri doğrulandı. [Input manifest](input-hashes.json)287
dosyayı; [erişim](input-access.json)34 shard/altı checkpointi ve ham sonuçları bağlar.

Altı checkpointten21600 validation çıkarımı eski per-row q ile tam eşleşti.
C1-02 üretiminin kaynak shard/config/solver baytları erişilebilir; yeniden
üretim yolu [saklama planında](STORAGE_AND_REPRODUCTION.md). Yeni veri üretimi
NOT_RUN. Shard ve checkpoint LOCAL_ONLY; uzak arşiv NOT_CONFIRMED.

| Conditioned seed | Geçerli ham q /3600 | Limit dışı | Geçerli-altküme konum medyan m | Yönelim medyan ° | Profil A |
|---|---:|---:|---:|---:|---:|
| 2026100201 |3331|269|0.206133|74.6479|0/3600|
| 2026100202 |3416|184|0.207037|78.1453|0/3600|
| 2026100203 |3412|188|0.210282|76.5182|0/3600|

Tam hassasiyet/P95 ve pose-only sonuçları `input-access.json`; bu tablo eski
sonucun özetidir, yeni ölçülmüş iyileşme değildir. Medyanlar geçerli-altküme
koşulludur; geçersiz q çıkarılarak başarı paydası daraltılmaz. Yeni rapor ayrıca
geçersizlere∞ atayan3600 paydalı muhafazakâr quantilleri gösterecek.

## Dondurulan karar

[Config](config.json), [deney matrisi](EXPERIMENT_MATRIX.md),
[ADR-013](../../docs/adr/ADR-013-c105-training-fk-domain.md) birlikte uygulanır:
conditioned 3×256 SiLU,13 giriş,6 sınırsız mutlak q;15204 labeled train,
aynı AdamW/seed/order/batch1024/200 epoch tavanı. Kontrol Lq, müdahale Lq+Lp+LR;
ℓ=0.9015 m, λq=λp=λR=1. LR chordal matris kaybıdır; derece metriği bağımsız FK'dir.

Her arm için bir sabit aday ve üç seed; en fazla dört özgün config. E-C04
normalize ReLU limit cezasını ekler. E-C05 yalnız E-C03'te ham limit ihlali varsa,
limit cezası olmadan tanh başlığını ayrı sınar. E-C06/07/08, Res-MLP/curriculum
gerekçeli SKIP. Kalan arama slotlarını sonuç sonrasında doldurmak yasaktır.

Kamusal C1-03 limit reddi değişmez. Ayrı opt-in eğitim FK yolu tüm sonlu ham
açılarda ideal revolute zinciri hesaplayacak; clamp/wrap/örnek eleme yok.
32 iç+48 dış örnek sonuçlardan önce [donduruldu](fk-domain-samples.json).
Bu uzantının ileri/FD/gradcheck testi **NOT_RUN**; güvenilirliği henüz
kanıtlanmadığı için eğitim kapısı kapalı. Eski FK testlerinin PASS olması yeni
uzantıyı otomatik kabul etmez.

Checkpoint seçimi C1-04 ile aynı en düşük etiketli validation Lq, eşitlikte ilk
epoch. Araştırma birincil metriği ise Profil A başarı sayısıdır. İki ölçüt
karıştırılmaz; validation FK başarısına bakıp başka epoch alınmaz. Taze eşli
kontrol ve tarihsel C1-04 ayrı raporlanır; her comparator tekrarı görünür kalır.

## Kapılar, kaynak ve sınırlar

[Test/negatif kontrol matrisi](TEST_MATRIX.md) teknik kapıları tanımlar.
Mevcut129 regresyon ve tam T-C01/T-C02 yeniden PASS. Pilot192 train/96 validation
satırını sabit ID ile20 update/arm izler; bileşen/katman/çıktı gradyanları,
NaN/Inf,≥175° ve limit dışı davranış ölçülür. Pilot ve yeni domain testleri
başarısızsa tam eğitim yok; λ ayarı için yeni ön kayıt gerekir.

Ana bütçe en fazla18 model-seed/54000 step/36 saat cap;4 GiB süreç, tek CPU thread.
Gerçek yeni eğitim süresi ve etkin insan emeği NOT_MEASURED. Kaynak tavanı
gözlenen performans diye sunulmaz. Bütün3600 validation satırı,351 unlabeled
wide, pair_mode/family/etiket kırılımları korunur. Test ve10000 ana sorgu
C1-06 için SEALED_NOT_RUN. Çarpışma ve fiziksel güvenlik NOT_CHECKED.

## Onayla açılacak iş

`SHA256SUMS` ve bu paketin commit'i üzerinden açık uygulama onayı alındıktan sonra
önce tekrar hash/erişim/regresyon, sonra domain testleri ve pilot, sonra
ön kayıtlı karşılaştırmalar çalıştırılır. [Gerçek komut planı](COMMANDS.md).
Şu anki blokaj açık Aşama2 onayıdır; ayrıca yeni FK/pilot kapıları henüz
uygulanmadı. Yeni checkpoint ve C1-06 model devri yoktur. Genel görev tamamlandı
veya T-C04 PASS diye işaretlenmez.

# Girdi erişimi, saklama ve yeniden üretim

8 Ekim 2026 · r1. `input-hashes.json` 287 dosyanın gerçek byte/SHA kaydıdır.
`input-access.json` erişim, checkpoint yükleme, 21600 birebir çıkarım ve
train/validation sayımlarını bağlar. Test ve benchmark yalnız bayt hash'i için
okundu; açılıp değerlendirilmedi veya seçime sokulmadı.

## C1-02

`data/generated/C1-02/v1`: **34/34 shard erişilebilir**, toplam15598536 byte.
Canonical content SHA `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`.
Train16800=8400 local+8400 wide; labeled15204=8400+6804. Validation3600=1800+1800;
labeled3249=1800+1449. Etiketsiz wide1596 train ve351 validation korunur.

`q_current` sorguda mevcut başlangıçtır; local satırda kaynak hedef q çevresinde
≤0.1 rad perturbasyon, wide satırda bağımsız örnektir. `q_target` local satırda
kaynak hedef; wide satırda sabit bütçeli öğretmenin q_current'a normalize en
yakın geçerli adayıdır veya NaN maskeli etikettir. `q_target` hiçbir model
girdisine girmez. Train-only position normalizasyonu, limit ölçeği ve root
splitleri değişmez; ana FK eğitimi de aynı15204 labeled satırla sınırlıdır.

12 F0-04 kaynak shardı, dondurulmuş üretici config/kod, DLS ve ilgili kilitler
gerçek SHA ile doğrulandı. `input-access.json` reproduction input sayısını verir.
`.gitattributes` tarihsel C1-02 girdi kaydında olduğundan yalnız bu politika dosyası
eski SHA karşılaştırmasından hariçtir; C1-03/04 için eklenen Git metin kuralları
veri üretici girdisi değildir. Bu istisna robot/veri/config/kod için uygulanmaz.

Veri kaybolursa mevcut CLI ile **yeni, boş** bir kökte yeniden üretim:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage full -Output data/generated/C1-02/c105-recovery
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage verify -Output data/generated/C1-02/c105-recovery
```

Bu komutlar C1-05'te NOT_RUN. Kaynak SHA'lar ve kilitli Pixi ortamı önce
doğrulanır. Üretim sonunda34 shardın file/content SHA'sı ve canonical content
SHA kabul edilmiş C1-02 manifestiyle karşılaştırılır; ancak eşleşme sonrası
veri kökü açıkça bağlanır. Eski v1 üzerine yazılmaz. C1-02'nin önceki iki
üretiminin eşliği bu oturumda yeni üretim yapılmış anlamına gelmez.

## C1-04 altı seçilmiş checkpoint

| Seed | Model | Byte | SHA-256 |
|---|---|---:|---|
| 2026100201 | pose_only | 1632963 | `b67400c83008c979ed33782945bc97f5e7ebe7590842f567e033f42cdf49a6dc` |
| 2026100201 | conditioned | 1651459 | `0d931d90fd7081ecf36868b21ccd830fc0a9daa0ff036fb96fff1d53ba8b6a5a` |
| 2026100202 | pose_only | 1632963 | `520096538b02d343d1e2300a0d476442efb7ee15704735215d1294e63668434d` |
| 2026100202 | conditioned | 1651459 | `27bf75601645b4a1e6adf228062a75a2409aca8f1ecad41c1aa2784b4506a00c` |
| 2026100203 | pose_only | 1632963 | `4cd060ab82045362d990b2f448e894392c39b1cb14e4f0866e2f8bc7567afa82` |
| 2026100203 | conditioned | 1651459 | `2f05c5ee8c4e6251f8c081aaf6affe31cec8ee8c10d3b5cff9e7f1d34541babf` |

Yol kuralı `data/generated/C1-04/v1/seed-<seed>/<model>-best.pt`; gerçek mutlak
yollar ve erişim true `input-access.json` içindedir. Altısı da metadata kontrolüyle
yüklendi; tüm validation q'ları eski kayıtla sıfır fark verdi. Tarihsel ağırlıkların
yeniden eğitilmesi gerekmiyor. C1-05 kontrolü taze eşli başlangıç/bütçe ve yeni
gradyan kayıtları için yeniden eğitilir; eski checkpointin yerine geçmez.

## C1-05 saklama ve devir

Yeni ağırlıklar henüz YOK. Configteki ignored `data/generated/C1-05/v1/...`
kökünde atomic best/last, config/input/normalization/robot/TCP/seed/epoch/source
SHA metadata'sıyla tutulacak. Her attempt benzersizdir; başarısız deneme silinmez.
Ham epoch/validation JSONL, komut argv/UTC/exit/stdout/stderr, özet, model kararı
ve küçük manifest `experiments/C1-05/stage2/` altında kaydedilecek. NaN/Inf JSON'a
standart dışı sayı olarak yazılmaz; null ve açık failure code kullanılır.

Veri ve ağırlıklar **LOCAL_ONLY**, uzak arşiv **NOT_CONFIRMED**. Normal Git
takibine shard/checkpoint/venv eklenmez. Uzak kopya yokmuş gibi tekrar kontrol
edilir; paylaşılabilir arşiv veya fiziksel robot güvenliği iddiası kurulmaz.
C1-06 için gelecekte bütün seedlerin checkpoint yolu/byte/SHA, config/veri
kimliği, seçim sırası ve test mührü birlikte devredilir. Şu anda model devri
**BLOCKED_PENDING_STAGE2**, test ve10000 sorguluk benchmark **SEALED_NOT_RUN**.

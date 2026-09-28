# C1-02 Aşama 1 inceleme ve dondurma · 28 Eylül 2026

Durum: **IN_PROGRESS / STAGE_1_COMPLETE**. REQ-C03 / T-C07 **NOT_RUN**. G0 `PASS / ACCEPTED` ve C1-01 `COMPLETE / T-C00 PASS` kaynak kayıtları incelendi. C1-01, C1-02'nin resmî ön koşulu değildir; C1-06 için ayrı baseline kanıtıdır.

## Girdi kimliği ve kaynak envanteri

`handoff-inputs.json` içindeki 15/15 dosya, C1-01 `frozen-hashes.json` içindeki erişilebilir 17/17 dosya ve `data/generated/F0-04/run-a` altındaki 12/12 shard dosyası SHA-256 ile eşleşti. Beklenmeyen hash: **0**. F0-04 dataset içerik SHA `5cb4e64580ecaf99afd11b3c8b98e06ed00c712e83bf2d9ee4d8c3acd58173fe`; manifest `3e2d8919ee59bcc7a96a0ce9c862633f4a0f4290747e346cf039d41c8fc490f3`. F0-05 sorgu listesi SHA `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`; C1-01 frozen kayıtla aynı. Tam SHA tablosu [input-hashes.json](input-hashes.json) içindedir. Bu kontrolde shard içerikleri tekrar FK ile doğrulanmadı; F0-04 kabul kanıtı ve byte hashleri kullanıldı.

C1-01 full/verify gate SHA değerleri kabul raporuyla eşleşti (d7b91265..., eda4bf6f...); verify gate PASS, beş özet hash bağı doğrulandı. G0/C1-01 örtüşen yollar ve yeni config/kod dahil toplam 43 hashli dosya [ön kontrol çıktısında](stage1-check.json) doğrulandı. Linux satır/FK doğrulaması bu oturumda yeniden koşulmadı.

F0-04 kaynağı: 10.000 main LHS (10 shard), 1.000 boundary (1), 1.000 singularity (1); mevcut `group_id` splitleri sırasıyla `7000/1500/1500`, `700/150/150`, `700/150/150`. `sampling_class` alanı kaynak aileyi taşır; source `q` alanı yalnız kök `root_q_target` olarak okunur. F0-04 şeması geriye dönük değiştirilmez. `normalization.json` yalnız 7.000 main train kökünden hesaplanmıştır; yeni çiftlerin normalizasyonu ayrıca yalnız yeni train satırlarından üretilir. Mevcut F0-04 `split-audit.json` ve `duplicate-audit.json` çapraz split grup/q ve alt küme exact çakışmalarını sıfır bildirir; C1-02 türevleri ayrıca denetlenecektir.

Robot `kuka_kr6_r900_sixx`, `base_link → tool0`, altı aktif eklem, metre/radyan, kanonik `wxyz`, sınırlandırılmış revolute eklemler ve Pinocchio/IndependentFK çapraz kontrolü değişmez. URDF `83d140b0...`, RobotSpec `4f97a205...`, TCP `52e96ebf...`; tam hashler manifestte. F0-05/C1-01'in 12.000 sorgusu (10.000 ana + 1.000 sınır + 1.000 tekillik) bağımsız değerlendirmedir; C1-01'in 600.000 ölçüm satırı 600.000 hedef değildir. Ham C1-01 sonuçları bu hostta mevcut, ama Aşama 1'de öğretmen etiketi olarak okunmadı.

## Dondurulan türetme sırası ve sayı planı

1. Önce kaynak manifest ve tüm shard file/content hashlerini doğrula. `sample_id`, `group_id`, `split`, `sampling_class`, `q`, poz ve quaternion alanlarını kaynakla eşleştir. Kimliği eksik satırı reddet.
2. Kaynak köklerin mevcut F0-04 splitini kalıt; türevden veya öğretmenden önce split haritasını dondur. Aynı konfigürasyonun, hedef pozun, kaynak ailenin ve herhangi yörünge soyunun çapraz split eşleşmesini audit et. Bu kaynakta zamanlı yörünge yoktur.
3. Kök başına bir `local`, bir `wide` satırı üret. Toplam **24.000 planlanan** çift: train 8.400+8.400; validation 1.800+1.800; test 1.800+1.800. Her splitte ve her alt kümede 50/50. Gerçekleşen değerler Aşama 2'de ölçülür; sapma sessizce telafi edilmez.
4. Hedef poz kaynağın limit içi kök `q` değerinin referans FK'sıdır. `local` için altı eklemde bağımsız `U(-0.1,+0.1) rad`; limit dışı veya exact eşit draw atılır ve aynı row stream devam eder, en çok 1.000 aday. Clamp yok. Sapmanın her eklem dağılımı, norm quantilleri ve exact eşitlik sayısı raporlanır. Yakınlık bir trajectory, hız veya jerk değildir.
5. `wide` başlangıcı farklı PCG64 stream ile tüm limit kutusundan bağımsız uniform çekilir; hedef `q` çizime veya kabul kararına girmez. Bir draw, hiçbir hedefe göre yakınlık filtresi yok. Her iki modda kaynak-kök, seed, deneme sayısı ve soy korunur.

Yeni `schema.json` `root_q_target`, `q_current`, `q_target` etiketi, `pair_mode`, öğretmen aday ailesi ve kaynak manifestini ayırır. NPZ `q_target` eksikliği yalnız açık `label_present=false` + tüm altı elemanda NaN sentinel ile temsil edilir; JSON audit'te `null`. Yükleyici sentinel satırını girdi yapmaz. Model girdisinin kesin allowlist'i `position_m`, `quaternion_wxyz`, `q_current`; `q_target` ve provenance/teacher/split alanları girdi dışıdır. Shardlar `data/generated/C1-02/v1` altında yeni sürüm olarak üretilir; F0-04 dosyalarına dokunulmaz.

## Öğretmen ve seçim yanlılığı

Local etiketi doğrulanmış köktür. Wide için çoklu çözüm dalını q_current'a yaklaştırma amacıyla **yeni** sayısal öğretmen gerekir. C1-01 başarı satırları bu amaçla kullanılmaz. Dondurulan yöntem F0-05 DLS kaynak/config'i ile 4 çağrı: q_current, ardından bağımsız limit içi uniform 3 restart; her çağrı en çok 200 iterasyon, adaptif eşik yok. Per-row seed ve aday aile kimliği stable SHA türetimidir. İlk pilot 90 wide satırdır: her source family × split için `source_sample_id` sırasındaki ilk 10 kök; en çok 360 çağrı/72.000 iterasyon, tek worker/tek thread, 4 GiB, 30 dakika job tavanı. Tavan aşımı bireysel adayın deterministik seçimini değiştirmez; job durur. Tam 12.000 wide etiketleme **NOT_RUN**.

Her aday önce boyut/sonluluk/limit, sonra ortak bağımsız Pinocchio FK ile Profile B `≤0.001 m`, `≤0.5°` denetlenir. Solver `SUCCESS` bayrağı doğrulama değildir. Geçerliler arasında `sqrt(sum(((q_candidate-q_current)/(upper-lower))²))` en küçüğü seçilir; exact float64 eşitlikte aday sıra numarası, sonra little-endian q byte sırası. Joint 6 dahil tüm eklemler sınırlıdır; circular wrap veya yalnız sin/cos kullanılmaz. Örnek dalı dağılımı, aday sayısı, geçerlilik ve maliyet `local/wide × split × source_family` ayrı raporlanır. Adaylar yoksa etiket `null`, hata sınıfı açık, train supervised loss maskelenir; validation/test envanteri olduğu gibi korunur. Train maske adedi ve öğretmen başarısızlığı seçim yanlılığı olarak raporlanır. Pilot için teknik sözleşme ve kaynak tavanı kapıdır; başarı oranı eşiği uydurulmaz. Öğretmen sonucu test hedefi seçimine asla girmez.

## Sızıntı ve benchmark audit tablosu

| Denetim | C1-02 train/validation/test | F0-05/C1-01 bağımsız sorgular |
|---|---|---|
| Kök grup / kaynak aile soyu | Aynı kökün iki modu aynı splitte; farklı splitlerde kök/soy kesişimi sıfır olmalı | `query_group_id` ve source root id ile kesişim sıfır |
| Exact konfigürasyon | Kök, etiket, başlangıç ve öğretmen restart q byte anahtarları splitler arasında ayrı; `-0` `+0` sayılır | 12.000 query `q_target` ve `q_current` ile exact kesişim raporlanır; sonuçtan silinmez |
| Hedef poz | Kanonik pose ve aynı kaynak köke bağlı semantik varyantlar ayrık; olası sayısal eş pose ayrıca incelenir | Kanonik pose örtüşmesi raporlanır; benchmark değişmez |
| Öğretmen aday ailesi | Row/group tabanlı aile ID'si tek splitte; restart soyları çapraz splitte yok | C1-01 solver satırları yeni öğretmen ailesi sayılmaz |
| Sayı ve rol | Plan 24.000 çift, 12.000 hedef kök | 12.000 sorgu, beş yöntem × iki deadline × beş tekrar = 600.000 ölçüm |

F0-05 sorguları yeni eğitim havuzuna katılmaz. F0-05'in F0-04 exact q dışlama kanıtı geçmişten taşınır; C1-02 türevi `q_current` ve yeni pose/kök aile karşılaştırması Aşama 2'de yeniden yapılacaktır. Exact pose karşılaştırması kanonik quaternion ile, ayrıca rotasyon matrisi bazlı semantik tolerans incelemesi ile yapılır; tolerans yeni kabul eşiği değildir, şüpheli eşleşmeyi insan incelemesine çıkarır. Hash tutmazsa beklenen/bulunan/yol/etki raporlanır ve üretim durur.

## T-C07 matrisi ve kapılar

| Test | Pozitif kontrol | Negatif / mutation kontrol | Aşama 1 |
|---|---|---|---|
| Girdi kimliği | handoff, config, shard ve query hash eş | bozuk SHA, eksik shard/soy | SHA salt okunur kontrolü; mutations NOT_RUN |
| Şema/girdi | alan/dtype/şekil ve allowlist | q_target girdi sızıntısı, NaN/Inf, ters quaternion | statik taslak kontrolü; üretim NOT_RUN |
| Çift | 0.1 rad, limit, wide bağımsız RNG, 50/50 sayılar | limit dışı q_current, yanlış split, teacher'a göre test filtreleme | üretim NOT_RUN |
| FK/öğretmen | bağımsız FK, B profili, sabit 4 aday/tie-break/null | solver SUCCESS ama kötü FK, yanlış aday, çember uzaklığı | pilot NOT_RUN |
| Sızıntı | group/source q/pose/soy/restart/exact q ayrık, benchmark audit | çapraz split kopya, restart family ve pose kopyası | tümü NOT_RUN |
| Normalizasyon/determinism | yalnız train fit; iki temiz üretimde kanonik içerik hash eş | validation/test istatistiği sızması, config/seed drift | NOT_RUN |
| Regresyon | F0-04, F0-05, C1-01 arayüzleri | eski kanıtları yerinde değiştirme | NOT_RUN |

**Kapı 0:** Aşama 1 sözleşmesi/hash kontrolü ve kullanıcı açık onayı. **Kapı 1:** source loader, split ve küçük sentetik pozitif/negatif testleri. **Kapı 2:** 90 wide öğretmen pilotu ve kaynak/başarı/yanlılık raporu; teknik hata veya tavan aşımı varsa tam üretim durur. **Kapı 3:** 24.000 planlanan çift, shard/manifest/normalizasyon. **Kapı 4:** T-C07, mutasyonlar, iki bağımsız temiz üretim, regresyonlar ve kabul kararı. Kapı 1–4 bu oturumda **NOT_RUN**.

## Ortam ve sınırlar

Windows x64 + `pixi run --locked` F0-04/05'in kanonik Python/NumPy/Pinocchio ortamıdır; ROS/Docker öğretmen DLS için gerekli değildir. `pixi.lock` SHA manifestte. Aşama 2'de gerçek OS/Python/NumPy/Pinocchio/CPU/thread/RAM ve lock kaydedilir; platform değişirse byte/content farkı tanılanır. C1-01 Ubuntu/ROS runtime yalnız geçmiş baseline'dır. Büyük shardlar Git dışı, hashli manifest ve komut Git içi. Collision ve fiziksel robot güvenliği **NOT_CHECKED**. Etkin insan emeği ölçülmedi. Aşama 2 açık onay bekliyor.

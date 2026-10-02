# C1-04 Aşama 1 tasarım incelemesi

2 Ekim 2026 · yazılım hedefi v1.0.0 · belge r1

Karar: **IN_PROGRESS / STAGE_1_COMPLETE; T-C03 NOT_RUN, E-C01 NOT_RUN**. Bu dosya eğitim sonucu değildir.

## Giriş kapısı ve kaynak kararı

Başlangıç `main` = yerel `origin/main` = uzak `refs/heads/main`: `93146c82108a60462b3108d799ba83e610a8bfef`. Başlangıçta üç ilgisiz izlenmeyen kullanıcı dosyası vardır; kapsam dışındadır. G0 24 Eylül **PASS / ACCEPTED** ([karar](../F0-06/G0_DECISION.md)); C1-02 **COMPLETE / T-C07 PASS** ([kabul](../C1-02/RUN-20260929-T-C07-acceptance.md)); C1-03 **COMPLETE / T-C01–02 PASS / ACCEPTED** ([kabul](../C1-03/stage2/acceptance.json)). Eski `NOT_STARTED` metinleri tarihsel kayıttır. C1-03 kabulü **Windows x64 CPU** iki koşusudur; Linux/CUDA NOT_RUN.

`AGENTS.md`, C1-04 görev/roadmap, Core raporu §2–6, TEST_PROTOCOL, G0/önceki kabul ve ADR-011/012 okundu. Girdilerin bayt kimlikleri [input-hashes.json](input-hashes.json) ile, yerel erişimleri `scripts/check_c104_stage1.py --check` ile denetlenir. C1-02 `dataset-manifest.json` içindeki `GENERATED_UNVERIFIED` üretim anı işaretidir; sonraki [acceptance.json](../C1-02/acceptance.json) PASS kararını taşır.

## Değişmez robot ve veri sözleşmesi

KUKA KR 6 R900 sixx, `joint_1`…`joint_6`, radyan/metre; `base_link` → `tool0`, TCP sabit `Ry(pi/2)` içerir. URDF SHA-256 `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`; robot manifest `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1`; TCP `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`. `load_robot` doğrulanmış joint limitlerini sağlar. Torch FK kaynak SHA-256 `1756a43356b2b177ce48c9f5b67bbece06734e132b09e39469e6b665f1b97478`; bağımsız değerlendirme Pinocchio referansıyla yapılır. Kinematik doğruluk çarpışmasızlık veya fiziksel güvenlik değildir.

C1-02 üretim kökü `data/generated/C1-02/v1`: 24.000 satır, 34 NPZ shard, canonical içerik SHA-256 `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`. `v1-repro` ikinci temiz üretimdir; [karşılaştırma](../C1-02/determinism-summary.json) 34/34 aynı dosya/içerik hashini raporlar. İki büyük kök Git dışı `LOCAL_ONLY`; uzak arşiv `NOT_CONFIRMED`. Kaynak F0-04 kökleri `data/generated/F0-04/run-a`; üretim/tekrar komutları [C1-02 COMMANDS](../C1-02/COMMANDS.md) dosyasındadır. Stage 2 girdileri kayıpsa eğitim durur; önce aynı frozen config ve kaynak hashlerle ayrı dizinde yeniden üretim, C1-02 verify ve 34/34 karşılaştırma gerekir.

Her F0-04 kökünden biri local, biri wide iki çift çıkar; kök `group_id` ve split daha önce dondurulmuş, iki çift aynı splitte kalır. `q_current` local modda kaynak q çevresinde her eklemde ±0,1 rad (limit içinde, exact eşit değil); wide modda limitler içinde bağımsız uniform başlangıçtır. `q_target` local modda doğrulanmış kaynak q, wide modda dört adaylı DLS teacher'ın bağımsız FK/limit kontrolünden geçen en yakın geçerli adayıdır. Aday yoksa `label_present=false`, `q_target` altı NaN sentinel; satır korunur. `teacher_status`, aday, kaynak q, split/soy hiçbir zaman girdi değildir.

| Split | Local toplam/etiketli | Wide toplam/etiketli | Wide etiketsiz | Etiketli toplam |
|---|---:|---:|---:|---:|
| train | 8.400 / 8.400 | 8.400 / 6.804 | 1.596 | 15.204 |
| validation | 1.800 / 1.800 | 1.800 / 1.449 | 351 | 3.249 |
| test (mühürlü) | 1.800 / 1.800 | 1.800 / 1.466 | 334 | 3.266 |
| toplam | 12.000 / 12.000 | 12.000 / 9.719 | 2.281 | 21.719 |

Ham her split 50/50 local/wide; etiketli train %55,25 local / %44,75 wide. Bu öğretmen seçim yanlılığıdır. Etiketsiz 2.281 wide satır silinmez; train supervised kaybından maskelenir, validation geometri envanterinde kalır, test C1-06'ya kapalıdır. Aynı kökün local/wide çiftlerinde hedef poz aynıdır: pose-only 7 özellik tekrar eder, bazen etiket dalları farklıdır. E-C01 yorumunda bağımsız hedef sayısı 24.000 varsayılmaz; model başına aynı etiketli çiftler kullanılır.

## E-C01 dondurulan seçimler

[config.json](config.json) normatiftir. Pose-only girdi `[(p-mean)/std, canonical quaternion wxyz]` = 7; conditioned buna `(q_current-lower)/(upper-lower)` ekler = 13. Pozisyon ortalama/std, C1-02'nin **16.800 train satırından** (etiketsizler dahil) `ddof=0` hesaplanmış [normalization.json](../C1-02/normalization.json) değerleridir. Quaternion normalize/kanonik `wxyz`, eklem limitleri robot varlığından; validation/test istatistiği fit edilmez. `q_target` yalnız etiket. Sıra ve hash uyuşmazlığı abort sebebidir.

Her iki model üç 256 SiLU gizli katman ve altı lineer mutlak q çıktısı taşır. Pose-only **135.174**, conditioned **136.710** parametre: ilk katman sırasıyla `(7+1)×256=2.048` ve `(13+1)×256=3.584`; iki `(256+1)×256=65.792`; çıkış `(256+1)×6=1.542`. Parametre farkı girdi genişliğinin zorunlu etkisidir, raporlanır. Çıktı ve etiket joint limitleriyle [0,1] ölçeğindedir; başlık sınırsız lineerdir, limit dışı ham q kırpılmaz. Kayıp örnek başına altı boyutun normalize kare hata **toplamı**, ardından etiketli örnek ortalaması; boyutsuz. Hata ayrıca rad cinsinden raporlanır. AdamW LR 1e-3, weight decay 0,01, beta 0,9/0,999, eps 1e-8, sabit LR; en çok 200 epoch, etkin batch 1024, validation patience 20, katı küçük loss seçimi ve tam eşitlikte erken epoch korunması donduruldu. Her seed'de modeller **birlikte durur**: ikisinin de 20 epoch iyileşmemesi veya 200 epoch; böylece epoch/step sayısı eş kalır. Üç eşli seed `2026100201`–`03`. Model sıraları aynı etiketli `pair_id` dizisi ve seed başına aynı epoch permütasyonudur; rastgele başlangıçlar input boyutu farkına uygun ayrı katmanlardır.

`micro_batch=1024` başlangıcıdır. Bellek zorunlu kılarsa tek etkin batch içindeki mikro kayıp `micro_count/effective_count` ile ağırlanıp **tek optimizer step** atılır; son eksik batch de aynıdır. Bu yalnız gradyan cebirsel eşdeğerliğidir, bit düzeyinde eşitlik iddiası değildir; gerçekleşen mikro boyutu raporlanır. Mimari, kayıp veya bütçe değişirse yeni config ve gerekçeli ADR gerekir. Res-MLP, delta-q, 6D, FK/limit/tekillik eğitim kayıpları ve H2 C1-05 kapsamıdır.

Checkpoint [şeması](checkpoint-schema.json), config/scaler/veri/robot/Torch FK hashlerini ve giriş/çıkış sırasını doğrular; fiziksel q ve limit durumu döndürür. Ağırlıklar `data/generated/C1-04/v1` içinde `LOCAL_ONLY`; yol, byte, SHA-256 ve erişim durumu küçük Git kayıtlarında tutulacaktır. Uzak arşiv `NOT_CONFIRMED`. Yalnız ağırlık dosyası taşınabilir teslim sayılmaz.

## Öğrenme, değerlendirme ve kaynak kapıları

[T-C03 test matrisi](TEST_MATRIX.md), ilk 64 main/train/local ve 32 ayrı main/validation/local sabit `pair_id` sıralı satırı, ilk seed, 200 epoch/batch64 ile her iki modelin gerçek etiket öğrenmesini dondurur. Başlangıç ve son train loss ile rad cinsinden medyan mutlak q hatası açık iyileşme gösterir; validation yalnız izlenir. Döndürülmüş etiketin Pinocchio FK hedef uyuşmazlığı, yanlış feature sırası, missing label/limit, NaN ve split sızıntısı fail-fast negatif kontrolüdür. T-C03 geçmeden altı tam koşu başlamaz. Küçük train ezberi genelleme değildir.

Validation için her seed/modelde etiketli q loss ve tüm 3.600 satırda ham geçerlilik/limit sayısı, geçerli q için bağımsız FK konum (m) ve geodezik yönelim (derece) hatası; `source_family`, `pair_mode`, `label_present` kırılımı. Geçersiz ham q FK'ye verilmez, ayrı sayılır; projeksiyon yoktur. Etiketsiz wide validation için q loss tanımsız, FK hedef hatası ölçülebilir. Model seçimi yalnız 3.249 etiketli validation loss ile yapılır. Nihai test ve 10.000 sorguluk benchmark C1-06 kapısına kadar açılmaz; Profile A/B tanımları değişmez.

Gerçek host Windows 11 Pro 10.0.26200, Ryzen 7 250, 25.025.695.744 byte RAM; Stage 2'nin C1-03 `torch 2.10.0+cpu` hashli overlay'i kullanılacak. Thread/worker 1/0. C1-02 44.479.998 byte/39 dosya ölçülmüş; model başına ağırlık yaklaşık 0,55 MB (`float32` parametre baytları, checkpoint/optimizer ek yükü hariç). **Eğitim süresi ve peak bellek ölçülmedi**. Plan: önce pilotta süre/RSS ölç, süreç RAM tavanı 4 GiB, tam koşu başına 120 dk cap; altı koşu için üst sınır 12 saat wall, gerçek kullanım loglanır. En iyi/son checkpoint ve epoch başına tek JSONL kayıt; train satır başına log yok. Cap aşımı/eksik seed `IN_PROGRESS` olarak kalır, başarıya çevrilmez.

## Aşama 1 kararı

Bu aşamada yalnız sözleşme, hash erişim audit'i ve komut planı vardır. Neural kod, eğitim, checkpoint, T-C03/E-C01 sayısal sonucu **NOT_RUN / NOT_MEASURED**. Aşama 2 için önce dondurulmuş hash audit'i, sonra yükleyici/MLP/test uygulaması gerekir. Kullanıcıya bu dondurmayı sunup açık Aşama 2 onayı beklenir. C1-05 ve G1 açılmadı.

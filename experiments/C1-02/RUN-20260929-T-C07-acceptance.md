# C1-02 Aşama 2 çalışma ve kabul kaydı

Kimlik: RUN-20260929-T-C07-acceptance
Durum: COMPLETE · REQ-C03 / T-C07 PASS / ACCEPTED
Görev ve gereksinim: C1-02 / REQ-C03
Tarih ve sorumlu: 28–29 Eylül 2026, Codex; proje sahibi Arda Tekgöz

## Soru ve değişiklik

[Aşama 1](RUN_REPORT.md) sözleşmesinin gerçek F0-04 köklerinden durumla şartlandırılmış çift üretebildiği; teacher hatalarını kayıt dışına atmadan, split/benchmark sızıntısı olmadan ve aynı veriyi ikinci temiz koşuda tekrar oluşturabildiği sınandı. Kullanıcının açık Aşama 2 onayı [stage2-approval.json](stage2-approval.json) içinde. Dondurulmuş [config](config.json), [schema](schema.json) ve 43 dosyalı [girdi manifesti](input-hashes.json) değiştirilmedi. Kabul eşikleri öğretmen başarı oranına göre ayarlanmadı.

Gereksinim → değişiklik → test → kanıt:

| Gereksinim | Değişiklik | Test | Kanıt |
|---|---|---|---|
| F0-04 kökleri, local/wide ve kaynak FK | `pairs.py`, pilot/full scriptleri | 24.000 satır, kaynak FK, local pertürbasyon | [dataset-manifest.json](dataset-manifest.json), [acceptance.json](acceptance.json), [pilot-assessment.json](pilot-assessment.json) |
| Teacher adayı, hata sınıfı, bütün test envanteri | DLS dört başlangıç; adayları bağımsız doğrulama; başarısız wide etiketi null | 48.000 aday, 2.281 missing, 0 test satırı çıkarma | [teacher-summary.json](teacher-summary.json), [acceptance.json](acceptance.json) |
| Train-only normalizasyon, input izolasyonu, split/benchmark soy ayrımı | `pair_validation.py`, normalizasyon ve audit | T-C07 7/7, mutation 12/12, `verify` PASS | [normalization.json](normalization.json), [leakage-audit.json](leakage-audit.json), [tc07-junit.xml](tc07-junit.xml), [mutation-junit.xml](mutation-junit.xml) |
| Tekrar üretilebilir veri ve arayüz uyumu | Sabit seed/NPZ düzeni; ikinci temiz koşu | 34/34 shard eş; seçili regresyon 42/42 | [determinism-summary.json](determinism-summary.json), [interface-regression-junit.xml](interface-regression-junit.xml) |

## Tekrar üretim

Başlangıç dalı `main`, başlangıç HEAD ve `origin/main`: `14e7de7c815690d2dd23a77dd36b60f5962a034d` (Aşama 1 commit'i). Çalışma ağacında iki kullanıcı Word dosyası ilgisiz/untracked idi; işleme alınmadı. Yazılım hedefi v1.0.0; belge revizyonu bağımsız. Bu raporun commit kimliği self-reference oluşturmamak için rapora yazılmadı; Git geçmişi ve `origin/main` kapanış kanıtıdır.

Ortam: native Windows 11 x64, AMD Ryzen 7 250, Pixi locked Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, Pinocchio 4.1.0. `pixi.lock` SHA-256 `56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` = 1; tek worker. GPU ve ROS/Docker kullanılmadı. Ayrıntılı kaynak ve script hashleri [environment.json](environment.json) içinde.

Robot URDF SHA-256 `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`; robot manifest `aec85ca4d2774bafe6e6412b7a4022e703a5a6bbd9143b647ba228d263b2bfd1`; TCP `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`; F0-04 dataset manifest `3e2d8919ee59bcc7a96a0ce9c862633f4a0f4290747e346cf039d41c8fc490f3`; F0-05/C1-01 benchmark query list `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`. C1-02 config `46eb7c0c13e3eb1382ae6137d0eea574c3e1e78148d6cdbf9d727ad97ba6b9c0`, schema `5aaebf4b717b934527ed130283630ab467ad2ec0017de53f6c7f29bed739bb0f`; PCG64 data seed `2026092802`, eğitim seed'i null / eğitim yapılmadı. Model/checkpoint yok.

Tam tekrar komutları [COMMANDS.md](COMMANDS.md) içinde. Üretim kökleri `data/generated/C1-02/v1` ve `data/generated/C1-02/v1-repro`; her biri 39 dosya ve yaklaşık 44,48 MB, Git dışı LOCAL_ONLY. Üretim aday/NPZ ham dosyaları, kaynak F0-04 ve benchmark raw girdileri bu makinede korunur; uzak büyük veri arşivi NOT_CONFIRMED. Küçük kanıtların SHA listesi [SHA256SUMS](SHA256SUMS). İlk tam koşu teacher wall `2336.032 s`, ikinci koşu `2374.562 s`; etkin insan emeği NOT_MEASURED. Pilot 18.734 s, 30 dakika cap içinde; ölçülen peak RSS `80.715.776` byte < 4 GiB. Tam koşu RAM için sürekli peak ölçümü yapılmadı; [resource-observation.json](resource-observation.json) iki süreç anlık/OS peak gözlemidir, kapsamı bu kadardır.

## Test ve ham kanıt

| Test / kontrol | Ölçülen sonuç | Karar ve kanıt |
|---|---|---|
| Aşama 1 sabit girdiler | 43/43 SHA eş | PASS; `python scripts/check_c102_stage1.py --check` |
| Pilot kaynak/teacher/kapı | 90 wide satır, 360 çağrı, 67 valid / 23 missing; 137 valid aday | PASS_TO_FULL_GENERATION; [pilot-summary.json](pilot-summary.json), [pilot-assessment.json](pilot-assessment.json) |
| Tam üretim | 12.000 F0-04 kökünden 24.000 çift, 34 shard; 12.000 local + 12.000 wide | PASS; [dataset-manifest.json](dataset-manifest.json), [acceptance.json](acceptance.json) |
| T-C07 kabul / mutation | 7/7 ve 12/12 | PASS; JUnit dosyaları yukarıda |
| F0-04/F0-05/C1-01 seçili arayüz regresyonu | 42/42 | PASS; [interface-regression-junit.xml](interface-regression-junit.xml). Tüm depo testi olduğu iddia edilmez. |
| Soy, sızıntı ve normalizasyon | root/seed, grup, aday ailesi, exact q/pose, near pose, benchmark kesişimi 0; train-only normalizasyon PASS | PASS; [acceptance.json](acceptance.json), [leakage-audit.json](leakage-audit.json), [normalization.json](normalization.json) |
| İkinci temiz üretim | Veri content SHA her iki koşuda `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`; 34/34 file/content shard eş | PASS; [determinism-summary.json](determinism-summary.json) |

Split × mode: train 8.400 local + 8.400 wide; validation 1.800 + 1.800; test 1.800 + 1.800. Main kökleri 7.000/1.500/1.500; boundary ve singularity ayrı ayrı 700/150/150. Ham mod karışımı her splitte 50/50; test envanterindeki eksik teacher etiketli satırlar tutuldu. Kaynak FK bağımsız yeniden kontrolünde maksimum konum farkı `6.697942338406922e-16 m`, rotation Frobenius farkı `9.462642277185446e-16`.

Teacher 48.000 aday çağrısı: 19.771 SUCCESS/bağımsız geçerli, 20.238 STALLED, 7.991 MAX_ITERATIONS. Geniş çiftlerin 9.719'u etiketli, 2.281'i `NO_VALID_CANDIDATE` ile etiketsiz. Eksik wide: train 1.596, validation 351, test 334. Train supervised joint etiketli local 8.400 / wide 6.804: %55,25 / %44,75. Ham 50/50 dağılımın etiketli alt kümede kayması seçim yanlılığıdır; başarısız train etiketleri masked, test satırları çıkarılmadı. Teacher geçerlilik denetimi solver statüsüne dayanmaz; finite/limit/bağımsız FK Profile B eşiklerini uygular. Geçerli adaylar mevcut `q_current` ile joint aralığına bölünmüş Öklid uzaklığına göre seçilir; tam float64 eşitlikte düşük ordinal, sonra canonical q bayt sırası kullanılır. Yerel pertürbasyon mutlak medyan `0.0498343285 rad`, maksimum `<0.1 rad`, exact `q_current=q_target` sıfır.

Soy auditinde çapraz split source root, türetim seed'i ve candidate family sıfır. Geçerli/restart aday q kesişimi sıfır. Yalnız **atılmış geçersiz solver çıktılarında** 9 çapraz split exact q anahtarı görüldü; bunlar input ya da etiket değildir, görünür tanı sayısı olarak korunur. Benchmark ile exact q, pose, grup ve near-pose kesişimi sıfır. Near-pose taramasında eşleşen yakın pozisyon çifti de sıfır; rapordaki `position_near_pairs_checked: 0` alanı aday çifti olmadığı anlamındadır.

## Sonuç ve yorum

REQ-C03 / T-C07 **PASS / ACCEPTED**; C1-02 COMPLETE. Kaynak kökler ve splitler korunarak 24.000 çift üretildi, etiket başarısızlıkları ölçüldü, input alanına `q_target` veya teacher sonucunun sızmadığı negatif testlerle kontrol edildi, iki temiz üretimin canonical verisi eş. `dataset-manifest.json` içindeki `GENERATED_UNVERIFIED` üretim-anı durumudur; ayrı [acceptance.json](acceptance.json) sonradan PASS vermiştir.

Hata/ara deneme kaydı: İlk tam üretim, pilot kaynak/thread koşulu henüz ölçülmediği fark edilince durduruldu; 6.608 aday satırlık kısmi dosya hashli [attempt-archive.json](attempt-archive.json) ile tutuldu ve kabul verisine katılmadı. İlk bellek probu Windows handle türü nedeniyle başarısız, kısmi pilot ayrı korundu; düzeltmeden sonra v3 pilotu çalıştı. İlk `verify` dtype yazımını, ikinci `verify` train istatistik toplama sırasını yakaladı; eşdeğer dtype karşılaştırması ve üretim sırasıyla doğrulama düzeltildi, frozen schema/eşikler değişmedi. Son PASS kanıtı yenidir. Ham aday JSONL iki koşuda elapsed_ns nedeniyle farklıdır; süre dışındaki semantik SHA `7edb3a3ea361fd28690d3d69ac839e87d829cda6d489c5419ca9d6a502b63a60` eş. Linux, fiziksel robot, çarpışmasızlık ve güvenlik doğrulanmadı. Sayısal IK hatası erişilemezlik kanıtı sayılmadı.

## Sonraki adım

F0-04/G0 ve Aşama 1 frozen girdileri, iki raw veri kökü, küçük hashli kanıtlar ve bu karar korunur. Roadmap sırasındaki C1-03 diferansiyellenebilir FK sıradadır ve henüz başlatılmadı. C1-04, C1-02 yanında C1-03 kabulünden sonra açılabilir. Core/G1 faz kabulü yapılmadı.

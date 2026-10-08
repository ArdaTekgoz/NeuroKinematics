# C1-06 ön kayıt ve devralma çalışma kaydı

Kimlik: RUN-20261008-C106-STAGE1
Durum: STAGE_1_COMPLETE / T-C05 NOT_RUN; Aşama 2 açık onay bekliyor.
Görev ve gereksinim: C1-06 / REQ-C04, REQ-C05
Tarih ve sorumlu: 8 Ekim 2026, Codex; proje sahibi Arda Tekgöz.
Yazılım hedefi v1.0.0; belge/protokol r1. G1/C1-07 başlatılmadı.

## Soru ve değişiklik

Yarım kalan C1-06 devralındı. Başlangıçta C1-06 kod/kanıt dizini bulunmuyordu;
C1-01–05 kabul/devirleri ve görev planı incelendi. Bağımsız final test görülmeden
girdi kimlikleri, üç seedli FK_TANH–Q H2 ve 21 checkpointlik karşılaştırma
matrisi, paired bootstrap/karar, failure/timing ve saklama protokolü hazırlandı.
Kaynak görev dosyası SOURCE_REQUEST.md olarak aynı baytlarla korundu.

| Gereksinim | Değişiklik | Test | Kanıt |
|---|---|---|---|
| REQ-C04, aday ve eşli kontrol sabitliği | config ve 21 checkpoint matrisi | Gerçek byte/SHA ve üç seed | input-hashes.json, COMPARISON_MATRIX.md |
| REQ-C05, test korunması/kimlik | Manifest-only sorgu kapısı, mühür ve soy politikası | 12.000 query hash/count, 600.000 baseline hash/count | test-seal.json, input-hashes.json |
| REQ-C04/05, dürüst paired istatistik | Root kümeli bootstrap ve H2 üç durum kararı | Sentetik etki/sıfır, overlap, eşit zor ağırlık, eksik/tekrar/seed negatifleri | synthetic-tests-final.xml, tests/c1_06/test_protocol.py |
| REQ-C05, bağımsız değerlendirme | Input whitelist, NumPy FK, raw geçerlilik ve timing çekirdeği | Pinocchio analitik tanıkları, NaN/Inf/shape/limit/pose negatifleri | synthetic-smoke.jsonl, synthetic-tests-final.xml |

## Tekrar üretim

Başlangıç HEAD/main/origin/main ve uzaktan doğrulanan main:
`82a9e114e65c5f3a4981c13b7c07fe71d854c76f`. Ayrıntı start.json.
Çalışma ağacında kullanıcıya ait STATUS/TRACEABILITY audit değişiklikleri,
AUDIT-20261003 klasörü, PDF ve iki Word dosyası vardı; bunlar bu görevin commit
kapsamı dışındadır. Yerel korunmuş kopyalar temp/c106-preserved içindedir.
Bu raporun commit'i kendine referans vermeden Git geçmişinden alınır;
teslimde commit ve uzak SHA ayrıca doğrulanır.

Ortam: mevcut Windows / Python overlay .venv/c103, kilitli Pixi ortamı.
Python executable/sürüm/platform input-hashes.json/environment içinde;
OMP/OPENBLAS/MKL/NUMEXPR=1 gerçek komut kayıtlarında. Yeni ortam kurulmadı;
temiz ortamda C1-06 tekrar koşusu NOT_RUN, GPU/Linux/fiziksel robot NOT_RUN.
Bu oturum için donanım/RAM tepe ölçümü ve etkin insan emeği NOT_MEASURED.
Komut başlangıç/bitiş UTC zamanları logs içinde; genel görev süresi tahmin edilmedi.

Robot URDF SHA `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`;
TCP `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`;
query `120b41f07109aeaca10e10fbb04783167cdc4ba282e7976941468c7dccfa4976`;
C1-02 content `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`.
C1-05 devir config SHA `ef3c9097a2603a511c2266e80963b9ae1e5deff14cf27ca6aa580c993cd58775`.
Tam dosya/model/SHA/byte listesi input-hashes ve config içindedir.
Bootstrap seed 2026100806, 10.000 tekrar; eğitim seedleri 2026100201/02/03.

## Test ve ham kanıt

| Kontrol | Gerçek sonuç | Kanıt |
|---|---|---|
| İlk girdi audit | FAIL; olmayan inference-witness.json yolu | commands/input-audit; input-audit-attempt-001.json |
| Düzeltilmiş girdi audit | PASS; 322 dosya, 21 checkpoint, 600.000 baseline satırı | commands/input-audit-002; input-hashes.json |
| İlk sentetik test | 26 PASS, 0 fail/skip | commands/synthetic-tests; synthetic-tests.xml |
| Son sentetik/negatif kontroller | 31 PASS, 0 fail/skip | commands/synthetic-tests-final; synthetic-tests-final.xml |
| Analitik smoke | 10 kayıt; 7 beklenen başarı, NaN/limit/pose için 3 beklenen başarısızlık | commands/synthetic-smoke; synthetic-smoke.jsonl |
| T-C05, gerçek final H2/CI, baseline satır join | NOT_RUN | test-seal.json |

Son test sayısı 31'dir; iki koşu toplanmaz. Testler yeni bileşenleri ve hata
kapılarını sınar; tüm depo regresyonu olduğu iddia edilmez. Önceki C1-03/04/05
matematik/eğitim kabulleri hashli girdidir, bu görevde yeniden çalıştırılmadı.
Freeze/verify komutları ve sonraki Git kontrolleri COMMANDS.md sırasıyla
commands altında tutulur; yalnız gerçek exit0 kaydı freeze doğrulamasını kanıtlar.

## Sonuç ve yorum

Aşama 1 ön kayıt tamam. Test satırları parse edilmedi, model çıkarımı yapılmadı,
eşik/λ/model/seed seçimi değiştirilmedi. Birincil H2 tablo/CI **NOT_RUN**:

| Birincil fark | Nokta | %95 CI | Karar |
|---|---|---|---|
| Üç seed ortalama FK_TANH−Q, main | NOT_RUN | NOT_RUN | Bekliyor |
| Boundary | NOT_RUN | NOT_RUN | Bekliyor |
| Singularity | NOT_RUN | NOT_RUN | Bekliyor |
| Zor eşit ağırlıklı ortalama | NOT_RUN | NOT_RUN | H2 NOT_EVALUATED |

600.000 ham baseline kaydı 1.112.301.428 bayt ve erişilebilir LOCAL_ONLY.
Checkpointler/shardlar LOCAL_ONLY; uzak arşiv NOT_CONFIRMED. Hash erişimi,
yeniden satır eşleştirmesi yerine geçmez; bu Aşama 2 kapısıdır. C1-04 ve C1-05
Q validation eşliği önceki kabulde birebirdir; bağımsız iki kontrol diye sayılmaz.
FK_TANH best epoch 7/186/8; E-C04/05 kısa eşli bütçeleri açıkça ayrılmıştır.

Zaman adaleti sınırı: C1-01 Ubuntu/ROS ve neural Windows CPU farklıdır;
yeniden eş platform koşusu planlanmadı, hız üstünlüğü iddiası yoktur.
Benchmark local ±0,05 rad, eğitim ±0,1 rad farkı önceden belgelendi.
Başarı = kinematik doğrulama; collision/physical safety NOT_CHECKED.

## Sonraki adım

Stage1 commit/push ve SHA tesliminden sonra kullanıcıdan açık Aşama 2 onayı.
Onay sonrası final yürütücüsü/raporlama CLI'si tamamlanacak, henüz mevcut
olmayan komutlar COMMANDS.md'de açıkça PLANLANDI olarak ayrılmıştır. Mevcut
çekirdek ve karar taslağı donduruludur. Final açılışından önce wrapper kodu,
checkpoint load/10 witness ve negatif testleri doğrulanır; sonra tek kampanya.

C1-07 devri henüz üretilemez: gerçek final H2/CI, failure analysis, raw ve
bağımsız join audit yoktur. T-C05 PASS ve C1-06 COMPLETE ilan edilmedi.
Sonuç olumsuz olsa bile teknik olarak eksiksiz araştırma kapanabilir;
eksik kanıt, sızıntı veya yanlış payda ile kapatılamaz.

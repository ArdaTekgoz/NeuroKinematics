# C1-05 Aşama 1 çalışma kaydı

Kimlik: RUN-20261008-C105-STAGE1  
Durum: **STAGE_1_COMPLETE / T-C04 NOT_RUN / C1-05 PARTIAL**  
Görev ve gereksinim: C1-05 / REQ-C03, REQ-C04  
Tarih ve sorumlu: 8 Ekim 2026 · Codex; proje sahibi Arda Tekgöz  
Yazılım hedefi v1.0.0; belge revizyonu r1; sürüm etiketi yok.

## Soru ve değişiklik

Görev dosyasındaki aşamalı talep uyarınca physics-aware deney sözleşmesi
uygulama öncesi hazırlandı. Soru: aynı conditioned model ve veri bütçesinde FK
kaybı, ardından ayrı limit düzenlemesi gerçek validation kinematiğini iyileştirir
mi? Sonuç henüz bilinmiyor. C1-04'ün **deneysel baseline COMPLETE / T-C03 PASS /
doğrudan IK NO-GO** kararı korundu.

| Gereksinim | Değişiklik | Test | Ham kanıt / karar |
|---|---|---|---|
| REQ-C03 değişmez girdiler | `audit_c105_inputs.py`, 287 dosya manifesti, erişim kaydı | 34 shard hash, 42 yeniden üretim girdisi, 6 checkpoint load; 21600 q yeniden çıkarımı | `input-hashes.json`, `input-access.json`, `commands/input-access-v2`: PASS |
| REQ-C03 doğru FK/gradyan | ADR-013 ayrı opt-in uzantı tasarımı; 80 frozen domain örneği | mevcut 129 regresyon ve full T-C01/T-C02 | `regression/`, `commands/{regression,fk-smoke,fk-full}`: PASS; yeni uzantı NOT_RUN |
| REQ-C04 etki yalıtımı | sabit Q/FK, FK/limit, koşullu FK/tanh matrisi; her arm bir aday/üç seed | tasarım incelemesi; yeni deney yok | `config.json`, `EXPERIMENT_MATRIX.md`, `TEST_MATRIX.md`; T-C04 NOT_RUN |
| İzlenebilirlik/kanıt | fail-closed SHA checker ve exact command logger; mevcut task planına tarihli ek | audit negative controls; final frozen check | `commands/audit-self-test` (3/3), `SHA256SUMS`, final audit log |

## Tekrar üretim

Başlangıç `main`; fetch sonrası HEAD/main/origin-main
`404268b20b584aac30f836812b2d574667766f77`. Başlangıç çalışma ağacındaki
STATUS/TRACEABILITY kullanıcı denetim ekleri ve untracked PDF/DOCX/AUDIT dizini
`workspace-start.json` içinde kayıtlıdır; C1-05 commit'ine alınmaz. Yeni durum
satırları seçilerek stage edilir. Kapanış commit kimliği kendi manifestinin
içine yazılmaz; Git geçmişi ve push/HEAD eşitliği son teslimde verilir.

Windows11, AMD Ryzen7 250 (8core/16thread), Python3.12.14, Torch2.10.0+cpu,
NumPy2.5.3, Pinocchio4.1.0, C1-03 hashli overlay kullanıldı; dört BLAS/OpenMP
değişkeni1, Torch1 thread. Ortam `environment.json` ve `commands/hardware`.
Foundations `pixi.lock` SHA
`56987eb3c4a3da13a5545d97e652046dbf4d3dc5394a2adacc31c4b87e9eee1a`.

Robot URDF SHA `83d140b03558e4b8ad428d0e07d16a31bc38c0fee643af049e4b75868a4d0a96`;
TCP SHA `52e96ebfadedbc2191d1d0b2dac646c81119973c8151b3d91e800ae0bea13e18`.
`base_link`→`tool0`, joint1–6, m/rad, canonical wxyz değişmez. C1-02 canonical
veri SHA `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`,
data seed2026092802. Eğitim seedleri2026100201–03, yeni eğitim NOT_RUN;
domain sample seed2026100805. `config.json` ve kaynak/kanıt kimlikleri
`SHA256SUMS` ile dondurulur. Altı eski checkpointin tam yol/byte/SHA/erişimi
`input-access.json`; yeni checkpoint YOK. Shard/checkpoint LOCAL_ONLY,
uzak arşiv NOT_CONFIRMED. Veri yeniden üretimi bu oturumda NOT_RUN.

[COMMANDS.md](COMMANDS.md) gerçek CLI ve sonraki uygulama sırasını verir;
`commands/*/command.json` UTC başlangıç/bitiş, argv, exit, raw log SHA içerir.
Yeni eğitim performansı ve etkin insan emeği NOT_MEASURED. Linux/CUDA/fiziksel
robot NOT_RUN. Çarpışma/fiziksel güvenlik NOT_CHECKED.

## Test ve ham kanıt

- C1-04 Stage1: 62 girdi/12 dondurulmuş çıktı PASS; 34 shard15598536 byte.
- C1-02 train16800/labeled15204; validation3600/labeled3249;351 unlabeled wide
  validation satırı korundu. Model loader test/benchmark açmadı.
- Checkpoint6/6 erişim/hash/metadata/load PASS;21600 q çıktısı eski JSONL ile
  tam eş, max fark0 rad. Her altı validation koşusu hâlâ Profil A0/3600.
- 129 unit/negatif/arayüz testi PASS, skip0;24 gerçek source mutant öldürüldü.
  NaN mutantından beklenen NumPy warning logda korundu.
- Tam T-C01:1086 q/dtype, float64 max konum4.47545209131181e-16 m,
  R8.763908156876301e-16; ikisi≤1e-9. Float32 max konum1.7329206910447826e-7 m,
  R4.04344059421814e-7; ikisi≤1e-5.
- T-C02:32 gradient/gradcheck/Jacobian, max FD fark5.289169102695723e-10;
  h1e-6, atol1e-5+rtol1e-3×abs(FD). Batch/edge/sensitivity kapıları PASS;
  `regression/fk-full/failures.jsonl` boş.
- Yeni80 domain örneği sadece girdi olarak üretildi. Eğitim FK uzantısı,
  physics-aware pilot, E-C03/04/05, T-C04 ve temiz C1-05 checkpoint çıkarımı
  **NOT_RUN**. Tam Foundations regresyonu bu tasarım turunda NOT_RUN.

Başarısız hazırlık: `commands/input-access` exit1, scalar predict API'sine batch
verildiği için. Audit özgün evaluation batch yolunu kullanacak şekilde
düzeltildi; `input-access-v2` exit0. Hata logu tutuldu, eski model/veri/eşik
değiştirilmedi. Girdi okuma sırasında henüz var olmayan run_c103_acceptance.py,
run_c103_stage2.ps1/run_c104_stage2.py adlarına yönelik keşif okumaları başarısız
oldu; gerçek CLI yolları source/önceki command kayıtlarından çözüldü.

## Sonuç ve yorum

**Aşama1 tamamlandı; C1-05 görevinin bütünü PARTIAL.** Dört olası config,
aynı üç seed, birer adaylık eşit arama fırsatı ve sabit q-loss checkpoint seçimi
önceden kaydedildi. E-C06/07/08 ayrı gerekçeyle SKIP; temsil üstünlüğü iddia
edilmez. FK uzantısı mevcut kamusal limit reddini değiştirmeyecek; clamp veya
invalid satır düşürme çözüm olarak kullanılmayacak.

C1-04 conditioned medyanları geçerli altkümede0.206–0.210 m ve74.6–78.1°;
Profil A0/3600. C1-05 karşılaştırma sonucu ve iyileşme **NOT_MEASURED**.
Test/10000 benchmark **SEALED_NOT_RUN**. C1-06 model devri hazır değil; G1 veya
v1.0.0 etiketi verilmedi. Kabul eşiği düşürülmedi, orijinal plan silinmedi.

## Sonraki adım

[STAGE1_REVIEW](STAGE1_REVIEW.md) ve commit/SHA paketi üzerinden açık Aşama2
onayı beklenir. Kaynak görev dosyasının kuralı: **“Uygulama için açık onay
gelmeden Aşama2'yi başlatma.”** Onay sonrası SHA/erişim/regresyonu yeniden
doğrula, ADR-013/negatif kontrollü FK ve pilotu uygula; ancak teknik kapılar
geçerse ön kayıtlı deneyleri çalıştır. C1-06'ya yalnız doğrulanmış config,
checkpoint/veri kimliği, erişim ve seçim protokolüyle devir yapılacak.

# T-C01 / T-C02 ve kabul matrisi — r1

Tüm satırlar **NOT_RUN**. Sayılar planlanan kapsamdır; test sayısı/başarı iddiası değildir.
Uygulama test yolları Aşama 2'de oluşturulacak; aşağıdaki kimlikler normatiftir.

| Kimlik | Gereklilik / değişiklik | Örnek ve oracle | Kabul / planlanan ham kanıt |
|---|---|---|---|
| S00 | Hash/ortam ön kapısı | Frozen manifest; Python/NumPy/Pinocchio/Torch versions, pip check | Tüm hash/sürüm eş, import çalışır; environment ve commands JSON |
| S01 | Küçük forward ve analitik zincir | Aşağıdaki A1–A5 fixture'lar | Normlar float64 1e-9, float32 1e-5; analytic JSON/JUnit |
| S02 | Smoke | Configteki 5 id; iki dtype; ilk 2 grad q | FK, p ve R backward, bağımsız FD geçmeden tam koşu yok; smoke gate |
| T-C01-64 | Torch relatif FK | Tüm 1086 q64 × Pinocchio 4.1.0 | Her iki norm <=1e-9; tc01-f64.jsonl, max/id özeti |
| T-C01-32 | Float32 korunması | Tüm 1086 q32; aynı sayıları float64'te Pinocchio'ya ver | Her iki norm <=1e-5; tc01-f32.jsonl; dtype ve giriş rounding ayrı |
| T-C01-B | Batch bağımsızlığı | 1086 q, batch 1/2/7/32/1024, reverse ve son chunk | Singleton eşliği, shape/dtype/device, ayrı row-id; batch JSON |
| T-C02-F | Graph akışı | 32 gradient q; 12 bileşen ve 3 skaler | p/R ayrı backward, finite, graph bağlı; gradients JSONL |
| T-C02-D | Bağımsız türev | 32 farklı q × 15 çıktı/loss × 6 eklem = 2880 | Pinocchio central h1e-6; atol1e-5+rtol1e-3*abs(fd); tüm satırlar |
| T-C02-G | Ek gradcheck | 32 q, 12 çıkış bileşeni, float64 | PyTorch gradcheck aynı eşikler, fast_mode=False; JUnit |
| T-C02-J | Jacobian ve frame | 32 q + A1/A3; dp ve vee(dR R.T) | F0-03 geometrik ve Pinocchio base/LWA dönüşümü; aynı türev eşikleri |
| T-C02-S | Sensitivity | grad-0000/1/2, frozen loss/yön | Her iki backend loss azalır, lineer yaklaşım config toleransında |
| T-C02-B | Batch graph | İlk 7 grad q; float64/32 | Cross-sample türev blokları sıfır, diagonal singleton eş; batch-grad JSON |
| E01 | Sınırlar | 24 exact/near, iki dtype forward; f64 edge FD | Tek taraflı/merkez stencil sütun bazında açık; aynı tolerans, 32'ye eklenmez |
| E02 | Tekillik adayları | Wrist -1e-8/0/+1e-8, zero | T-C01 + bileşen FD geçer; SVD/sigma_min raporlanır, örnek çıkarılmaz |
| E03 | Pi/quaternion ayrımı | Tek eksenli fixture q=pi-1e-7,pi,pi+1e-7 | R ve bileşen türevleri düzgün; wxyz, q/-q ve pi metrik regresyonu; quaternion branch türevi kapıya karıştırılmaz |
| E04 | Device/dtype | CPU float64/32, noncontiguous input, art arda dtype çağrıları | Input dtype/device, sabitlerin hassasiyeti; list/int/half/complex reddi; CUDA NOT_RUN |
| N/M | Negatif/mutasyon | Ayrı matristeki tüm satırlar | Her mutant gerçek üretim yolunda öldürülmeli; survive/error ayrımı |
| R01 | F0-01/02/03 regresyon | Mevcut ilgili test dizinlerinin tamamı | Eski eşikler ve hashler korunarak JUnit; eski kanıt dosyaları üzerine yazılmaz |
| R02 | C1-02 arayüz regresyonu | C1-03 altında yeni shard bağımsız fixture; C1-02 `load_contract/make_base/validate_one/checked_input_projection` gerçek fonksiyonları | Frozen config/schema, local/wide+unlabeled, input projection, yanlış sıra/pose/split/nonfinite/limit ve target leakage; JUnit |
| R03 | Bağımsızlık | Pinocchio/custom_fk forward importunu reddeden alt süreç | Torch zinciri çalışır; yalnız test oracle süreci Pinocchio çağırır |
| C01 | Temiz tekrar | İkinci fresh Windows checkout/env/overlay; S00→C01 kapısı | İki koşu tam PASS; girdiler/paket/sample SHA eş; JSON/JUnit/log audit |

## Elle hesaplanabilir fixture'lar

Fixture kaynağı gerçek frozen URDF'ye yazılmaz. İç kernel için ayrı test kimliği.

R02 için mevcut `tests/c1_02/test_tc07_mutations.py` test desenleri okunmuştur,
ancak onun `roots()` fixture'ı büyük F0-04 shardlarını açar. Temiz C1-03
kurulumu bu dosyayı doğrudan çalıştırmayacak. Yeni `tests/c1_03` fixture'ı
grad-0000/1/2 q'larından Pinocchio ile pose üretip ayrı `c103-fixture-*` sample/group
id ve train/validation/test root alanları kuracak; C1-02 üretim fonksiyonlarını
gerçekten çağıracak. `q_current` Torch FK'ye etiketten bağımsız verilecek;
label_present=False wide satır korunacak. C1-02 kaynak/test/kanıt dosyası değişmez.
Bu seçili arayüz regresyonudur; tam T-C07 yeniden kabul iddiası değildir.

- **A1:** F0-02 `small_robot` (`tests/f0_02/conftest.py` hashli). World→base
  xyz=(3,4,5), Rz(pi/2); j1 Z, j2 öncesi x=1, spacer x=2, TCP x=1 ve Rx(pi/2).
  q=(pi/2,-pi/2): relatif p=(3,1,0), R=Rx(pi/2),
  T satırları `[[1,0,0,3],[0,0,-1,1],[0,1,0,0],[0,0,0,1]]`.
  Dünya pozu farklıdır; base mount değiştirilince relatif sonuç sabit kalmalı.
- **A2:** j1 origin'e Rx(pi/2) ekle, q=(pi/2,0): p=(0,0,4).
  Local axis'i world axis sanma ve origin/motion sırası hatasını yakalar.
- **A3:** A1'de spacer j1 ile j2 arasına taşınır; q=(pi/2,-pi/2): p=(1,3,0).
  Fixed dönüşümün aktif eklemler arasında korunmasını sınar. XML joint sırası
  karıştırılır; q ad eşlemesi değişmez.
- **A4:** Tek Z eklem, origin xyz=(1,2,3), TCP xyz=(2,0,0), Rx(pi/2),
  world→base xyz=(3,4,5), Rz(pi/2). q=pi/2 için p=(1,4,3),
  R=`[[0,0,1],[1,0,0],[0,1,0]]`; dp/dq=(-2,0,0), omega=(0,0,1).
  q=0/±pi/2 ve pi±1e-7 de test edilir. Limit [-4,4].
- **A5:** Gerçek robot zero için elle toplanan p=(0.980,0,0.435), R=Ry(pi/2).
  Bu tek vaka yeterli doğruluk kanıtı değildir; tüm T-C01 zorunludur.

## Tam kapı

Sıra: onay kaydı + Stage1 SHA audit → kurulum → küçük f64 forward → A1–A5 →
S02 smoke → T-C01-64/32/B → T-C02-F/D/G/J/S/B → edge/device → mutation →
regression → ikinci temiz koşu → raw/SHA/TRACEABILITY audit → karar.
Her zorunlu satır geçmeden **PASS / COMPLETE yok**. Fail, nonfinite, eksik satır,
yanlış id, kayıp mutation veya eksik temiz koşu kabulü durdurur. Model eğitimi,
C1-04, G1 veya v1 etiketi bu görev kapsamında başlatılmaz.

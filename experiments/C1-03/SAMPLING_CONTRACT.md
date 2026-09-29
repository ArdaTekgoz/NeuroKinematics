# Örnekleme, tolerans ve kanıt sözleşmesi — r1

Bu sözleşme sayısal sonuç görülmeden 29 Eylül 2026'da donduruldu.
`samples.jsonl` her örneğin id/grup/q64/q32 ve little-endian C-order q SHA-256'sını
taşır. Dosya ve config hashleri `stage1-check.json` / `SHA256SUMS` içindedir.
Üretici `scripts/check_c103_stage1.py`; yalnız yapı/örnekleme yapar, FK çağırmaz.

## Envanter

1024 FK iç bölge q: PCG64 seed 2026092903. Bağımsız 32 gradient q: seed
2026092904. İki grup da frozen alt+0.001 / üst-0.001 aralığında uniform;
redraw, teacher filtresi veya sonuç temelli seçim yok. 30 elle seçilmiş vaka:
zero, limit midpoint, `[.3,-.6,.8,-1,.5,-.7]`; her altı eklem için alt/üst
exact ve 5e-7 rad içeride (diğerleri midpoint); mixed vektörün beşinci eklemi
-1e-8/0/+1e-8 olan üç wrist adayı. Toplam **1086**, her iki dtype'ta hepsi T-C01.
32 gradient q birbirinden farklıdır; hand vakaları 32 sayısına eklenmez.
Zero ve midpoint 'neutral' adını belirsiz bırakmaz; Pinocchio neutral tahmini yapılmaz.

Float32 input bir kez round edilir. Limit dışına round olan endpoint yalnız
**örnek üretiminde**, önceden tanımlı `nextafter` ile bir ULP içeri alınır;
q32 dosyada exact kaydedilir. Forward kendisi q kırpamaz. Pinocchio'ya aynı q32'nin
float64 promotion'ı verilir; q64 ile q32 giriş farkı ayrı kayıttır. Endpoint
float32 en yakın iç temsil testidir, exact matematik sınırı yalnız float64'tür.

## Norm ve türev

Konum `sqrt(sum((p_torch-p_pin)^2))` metre. Rotasyon
`sqrt(sum_ij((R_torch-R_pin)^2))`, boyutsuz Frobenius; quaternion/geodesic değil.
Farklar raporlama için float64'te hesaplanır, Torch forward dtype'ı değişmez.
Float64 iki norm <=1e-9; float32 iki norm <=1e-5. Her örnek ve iki norm geçmeli.
Alt satır, ortonormallik Frobenius ve `abs(det(R)-1)` için configteki
homogeneous_atol kullanılır. Nonfinite tek başına FAIL; ortalama aşımı örtemez.

32 gradient q'da her 3 p + 9 R bileşeni ve configteki üç skaler fonksiyonun
altı q türevi denetlenir: **32 × 15 × 6 = 2880** bağımsız merkezi fark karşılaştırması.
Merkezi fark Pinocchio çıktısından `(f(q+h e_j)-f(q-h e_j))/(2h)`, h=1e-6 rad.
`abs(a-fd)<=1e-5+1e-3*abs(fd)` bileşen bazında; bağımsız mutlak ve göreli
maksimumlar da raporlanır. Göreli rapor denominator `max(abs(fd),1e-12)`;
sıfıra yakın FD'de bu büyük olabilir, kapı yukarıdaki birleşik formüldür.

Ek PyTorch gradcheck 32 q'da 12 çıktı bileşenine, aynı h/atol/rtol ve
fast_mode=False ile zorunlu. Bu Torch'un kendi sayısal kontrolüdür; bağımsız
Pinocchio merkezi farkın yerine geçmez. Diğer hand iç vakalarda aynı türev
kontrolü ek edge kanıtıdır; 32 iç örneğe sayılmaz.

24 exact/near sınır vakasında her sütun için q±h önce kontrol edilir; iki taraf
uygunsa merkez, alt tarafta taşarsa ikinci derece forward, üst tarafta taşarsa
ikinci derece backward stencil (configte formüller). q±2h de limit içinde olmalı;
değilse açık FAIL, sessiz h değişimi yok. Aynı tolerans, ayrı sonuç tablosu.

SO(3): `A_j=(dR/dq_j) R.T`, `omega_j=vee((A_j-A_j.T)/2)`;
vee(S)=(S[2,1],S[0,2],S[1,0]). Base eksenlerinde `[dp;omega]` altı sütun,
F0-03 IndependentJacobian ve PinocchioJacobian ile karşılaştırılır.
TCP-local `R.T dR` ve Euler türevi bu sözleşmeye eşdeğer değildir.

Sensitivity grad-0000/1/2: sabit smooth loss gradyanının negatif birim yönünde
h=1e-6 perturbasyon. Norm>1e-8; iki bağımsız backend loss azalmasını ve
`delta_loss ~= h*g.dot(direction)` (atol1e-8+rtol1e-3) sağlar. Ayrıca q_j ±h
R türevleri anlamlı olmalı. Konumun son eklem türevinin sıfır olması otomatik FAIL değildir.

## Batch ve dtype

1,2,7,32,1024 batch; tüm 1086 q sırayla chunk edilerek her boyutta test edilir,
son kısa chunk korunur. Singleton ile matris eşliği ilgili dtype norm sınırında.
Satır-id eşlemesi korunur, batch reverse/permutation geri açılınca aynı sonuç.
İlk yedi gradient q için full batch autograd Jacobian: çapraz örnek blokları
tam sıfır, diagonal singleton gradyanları gradient toleransında. Float32 graph,
finite backward ve dtype/device ayrı kontrol; float32 FD resmi T-C02 değildir.

## Satır sonu, ham sonuç ve tekrar üretim

Yeni C1-03 JSON/JSONL/MD/Python/lock metinleri UTF-8, BOM'suz LF ve son LF.
`.gitattributes` bu kapsamı sabitler. `SHA256SUMS` açıkça **canonical LF bytes**
SHA-256'sıdır; dosyanın kendi satırı dışlanır. Yeni kayıtlarda raw==canonical
beklenir. Eski girdilerde `input-hashes.json` raw çalışma ağacı, Git blob içeriği
SHA-256 ve canonical LF SHA-256 değerlerini ayrı alanlarda saklar. Bunlar Git'in
SHA-1 object id'si değildir. G0/robot bayt hashleri hiçbir zaman normalize edilmez.

Stage 2: all-results JSONL tüm örnekleri; failures JSONL tüm başarısızları,
q/id/dtype/device/robot/config/SHA/iki T/hata normu/türevin autograd/FD/epsilon/
stencil/atol/rtol/kararını saklar. NaN/Inf JSON'da standart dışı sayı değil
ayrı durum ve string/null ile açık kodlanır. Traceback ve stderr/stdout korunur.
Exception örneği kaybolmaz. Smoke FAIL ise ana koşu NOT_RUN; tam koşuda
hatalı satırlar toplanır ve final exit nonzero. JUnit + count/max/failure özeti
JSON, mutation mutant/diff/killing-test/exit ve regression JUnit zorunludur.

İkinci temiz Windows checkout/kurulumda bütün kapı tekrar edilir. Input/config/
sample/dependency hashleri eş olmalı; matematik değerler aynı toleranslarla geçmeli.
Zamana bağlı log/JUnit byte eşliği aranmaz. Büyük ham dosya olursa path/bytes/
rows/SHA ve doğrulanmış uzak arşiv yoksa LOCAL_ONLY yazılır. Etkin emek ve
performans ölçülmedikçe NOT_MEASURED. Yeni eşik veya örnek gerekirse r2 gerekçesi
ve yeni bağımsız koşu gerekir; r1 sonuçları yeniden başarılı etiketlenmez.

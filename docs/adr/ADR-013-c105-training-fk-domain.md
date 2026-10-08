# ADR 013 — C1-05 eğitim FK alanı ve kontrollü deney sınırı

Durum: AŞAMA 1 TASARIMI DONDURULDU; uygulama açık onay ve test kapılarına bağlı
Tarih: 8 Ekim 2026 · Core C1-05 / REQ-C03, REQ-C04 · Belge r1

## Bağlam

C1-04 conditioned başlık altı normalize, sınırsız mutlak q üretir. Kabul edilmiş
`TorchFK` ve `TorchSerialChain` limit dışı q'yu reddeder. E-C03'te FK terimini
eklemek için clamp, geçersiz satır eleme veya tanh uygulamak aynı anda başka bir
değişkeni değiştirir. Başlangıç ağının dahi limit dışı çıktıları olabilir.

## Karar

Aşama 2'de ayrı `neural/training_fk.py` modülünde açık opt-in
`TrainingFK.from_frozen(domain="finite_revolute_extension")` tasarlanır.
Bu API henüz uygulanmadı. Kabul edilmiş `kinematics/torch_fk.py`, PinocchioFK,
robot girdileri ve kamusal limit reddi aynen korunur. Yeni eğitim sınıfı aynı
doğrulanmış URDF zinciri, joint sırası, fixed origin/TCP ve Rodrigues çarpımını
standart Torch işlemleriyle hesaplar; graph içinde NumPy/Pinocchio/detach yoktur.
Kernelin küçük sayısal gövdesi ayrı tutulur; kabul edilmiş kaynak hashlerini
değiştirmemek için bu sınırlı çoğaltma tercih edilir, eşlik testi zorunludur.

Sonlu döner eklem açıları üzerinde sin/cos ile katı dönüşüm matematiksel olarak
tanımlıdır; limitler bu dönüşümün tanım alanını değil robotun izin verilen
konfigürasyonlarını sınırlar. Uzantı limit dışı açılarda fiziksel olarak anlamlı
bir ideal zincir dönüşümü hesaplar; mekanik erişilebilirlik veya güvenlik onayı
vermez. NaN/Inf, yanlış şekil/dtype/kimlik/sıra/birim reddedilir. Q sarılmaz,
kırpılmaz, projekte edilmez: `q_raw == q_fk_input == q_evaluation`.

Bağımsız test oracle'ı ayrı doğrulama kodunda doğrudan Pinocchio modeline ham q
yerleştirip `world_base^-1 @ world_tool0` hesaplar. Mevcut PinocchioFK limit
kontrolü gevşetilmez. Analitik tek/iki eklem fixture'ları aynı parsera bağımlı
olmadan trigonometri ve türevleri denetler. Bu test yolu eğitim backward'ına
girmez. Temsil edilebilir aşırı büyük sonlu değerler için sadece finite çıktı
stresi yapılır; ileri/FD doğruluğu sayısal çözünürlüğün yeterli olduğu dondurulmuş
iç ve kontrollü dış örneklerde sınanır. Her sonlu değerde sayısal FD doğruluğu
iddiası kurulmaz.

## Kapı ve alternatifler

`TEST_MATRIX.md` içindeki float64/32 ileri eşlik, analitik, merkezi fark,
gradcheck, negatif kontroller ve eski T-C01/T-C02 regresyonu geçmeden E-C03
başlamaz. Başarısızlıkta teknik düzeltme/yeni protokol gerekir; sessiz domain
daraltma yoktur. Clamp ve yalnız geçerli minibatch reddedildi. Tanh E-C05'e;
normalize ReLU E-C04'e ayrıldı. FK uzantısı E-C05 başlık etkisi diye sunulmaz.

## Deney bütçesi kararı

İlk dalga yalnız sabit ağırlıklı E-C03 ve E-C04, gerekirse E-C05'tir. Her arm
için **bir** ön kayıtlı konfigürasyon, üç seed kullanılır; eşit arama fırsatı
bir adaydır, adaptif ağırlık taraması yapılmaz. Dört olası konfigürasyon sekiz
üst sınırını aşmaz. E-C06/07/08, Res-MLP ve curriculum bu kayıtta SKIP'tir;
iyileşme gözleyip boş dört slotu doldurmak yasaktır, yeni ön kayıt gerektirir.
Bu tercih REQ-C04 kapsamındaki seçili varyantların FK/limit etkisini ayırır;
q/delta veya quaternion/6D üstünlüğü ölçülmüş sayılmaz.

## Sonuç

Yeni matematik yolu ve pilot henüz NOT_RUN; bu ADR teknik kabul değildir.
C1-04 NO-GO, G0/C1-02/03/04 kayıtları ve C1-06 test mührü değişmez.

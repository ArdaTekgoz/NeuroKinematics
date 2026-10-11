# ADR 014 — C1-06R ayrı araştırma ve CUDA tanı ortamı

Durum: KABUL EDİLEN UYGULAMA KARARI; deney/ürün kabulü ayrı
Tarih: 9 Ekim 2026 · REQ-C02–05

## Bağlam

C1-06 araştırması tamamlandı ancak neural başarı sıfır ve H2 reddedildi.
Kullanıcı planı takiben uygulamanın başlamasını istedi; kişisel RTX 5060
Laptop (8151 MiB) ve 24 GB RAM ile süre sınırı olmadan çalışabilir.

## Karar

C1-06R yeni deney kimliğidir. G0, C1-02–06 ve eski final sabit kalır.
R0/R1 yalnız train/validation ve önceden mevcut FK matematik örneklerini
kullanır. Mevcut CPU Torch 2.10.0 yerine aynı sürümün resmi cu128 build'i,
ayrı `.venv/c106r` içinde, değişmeyen Pixi tabanı üzerinde kurulur.
NumPy/OpenMP için ADR-012 exact wheel yaklaşımı korunur. CUDA cihazında
FP32/FP64, TF32 kapalı, deterministik algoritmalar ve CUBLAS workspace
kullanılır; AMP/torch.compile bu tanıda yoktur. CPU referans bağımsızdır.

Resmi kaynaklar: [PyTorch sürümleri](https://pytorch.org/get-started/previous-versions/),
[Blackwell/CUDA 12.8 desteği](https://pytorch.org/blog/pytorch-2-7/).
Gerçek URL/SHA çözümü `runtime-resolution.json` ve hashli lock'tadır.
Resmi uyumluluk bilgisi gerçek import/forward/backward/FK testinin yerine geçmez.

64 main/local ve 64 dengeli train örneği küçük öğrenme tanısıdır. Aynı
conditioned 256×3 MLP, supervised Q kaybı ve 5000 güncelleme kullanılır;
cosine LR 1e-3→1e-5, AdamW weight decay 0,01, seed 2026100901.
Bu yeni tanıda sabit 5000 adımın sonunda her kümede Profil A 64/64 aranır;
eski T-C03 koşulları veya sonuçları değiştirilmez. Her 100 adım kayıt alınır,
sonuç görüldükten sonra en iyi adım seçilerek başarısız son adım gizlenmez.
Genelleme veya H2-R iddiası yoktur. Tanı başarısızsa eşik düşürülmez.

## Alternatifler ve sonuçlar

Eski ortamı yükseltmek tekrar üretimi bozar; ayrı overlay seçildi.
Docker/WSL gerekirse ayrı runtime revizyonuyla değerlendirilir.
H2-R, ürün hedefi, tam veri/mimari ve final protokolü R2/R3'te dondurulur;
bu ADR onları ölçülmüş kabul etmez. Kullanıcının sınırsız süre tercihi,
her deneyin adım/konfigürasyon bütçesinin ön kaydını kaldırmaz.

## 9 Ekim uygulama eki — tanı ve ilk uzun tur

İlk local64 61/64 FAIL korunur; mixed64 64/64 PASS. Aynı local checkpoint,
model ve Q hedefinde LBFGS r2 yine61/64; aynı başlangıç ve optimizer ile
pozitif1e6 kayıp ölçeği r3 64/64 sağlar. Ayrı config/registration/sonuç
dosyaları bu adaptif **train-only tanıları** saklar; r1 yeniden PASS yazılmaz.
Ana eğitimde bu ölçekleme veya tanı ağırlıkları kullanılmaz.

İlk kullanıcı turu aynı mimari/veride optimizasyon yeterliliğini sınar:
batch256,2000epoch/120.000update, cosine1e-3→1e-6 ve validation Profil A
öncelikli seçim. Eski200epoch/3000update'e göre daha uzun bütçe ve seçim
kuralı ortak olarak bütün arm'lara uygulanır. Q/FK × linear/tanh2×2 tasarım
geçmiş limit ihlali ve birleşik etki atfı sorunuyla gerekçelidir. Üç seed,
12koşu,1.440.000update ön kayıtlıdır; ara iyileşme görüldü diye bütçe değişmez.
Bu tur loss/head etkisini kendi eğitim rejiminde ölçebilir; eski ve yeni
rejim farkı tek başına yalnız scheduler'a atfedilemez. H2-R nihai karar ve
%95 ürün değerlendirmesi yeni bağımsız testten önce ayrıca dondurulacaktır.

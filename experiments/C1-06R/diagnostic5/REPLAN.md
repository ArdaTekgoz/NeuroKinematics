# C1-06R yeniden plan — yerel öğrenme ile global genellemeyi ayır

10 Ekim 2026 · PROPOSED / NOT_RUN · Yeni uzun kampanya henüz hazır değil

## Neden yön değiştiriyoruz?

Optimizer, global kayıp ölçeği, genişlik, göreli pose ve profile-scaled pose
amacı denendi; validation A0. Kinematik doğrulamalar ve teacher oracle PASS.
Train hassasiyeti hâlâ zayıf; Q devam koşusu train'i iyileştirirken validation
bozuluyor. Bu nedenle sonraki çalışma uzun bir hiperparametre taraması değil,
C1-02 veri desteği ve C1-04 öğrenilebilirlik tasarımını ayıran deney olmalı.

## Önerilen ilk müdahale: sürümlü, yön çeşitliliği olan local veri

Yeni C1-02R türevi üretilmeden önce ayrı ADR/config hazırlanacak. F0-04 ve
C1-02/v1 değişmez; root/split sınırı korunur. Train kökleri dışında hiçbir
validation/final örneği yeni train'e giremez. Her yeni satır root, seed,
q_current/q_target üretim kuralı ve bağımsız FK doğrulamasıyla kaydedilir.

Tek değişken aynı kök çevresindeki yerel yön çeşitliliği olmalı:

- Kontrol: mevcut local örnekleri aynı toplam örnek maruziyetini sağlayacak
  biçimde tekrar kullanır.
- Müdahale: aynı train kökleri çevresinde birden fazla deterministik local
  perturbasyon kullanır. Batch sayısı/update, model, loss, head ve decoder
  kontrolle eşleşir; yeni örnek sayısı/tekrar sayısı açıkça ayrı raporlanır.
- Ayrı tanı kümesi: aynı train köklerinde daha önce kullanılmamış local
  perturbasyonlar. Bu, **aynı-kök yerel tanı**dır; validation/genelleme veya
  bağımsız final diye adlandırılmaz.
- Mevcut untouched-root validation aynı tam paydayla ayrıca ölçülür;
  model seçimi için yeni-kök satırları train'e taşınmaz. Validation artık
  araştırmada çok kez kullanılmıştır; nihai başarı yeni bağımsız final ister.

Önce küçük yerel kapsamda, sonra daha fazla kökte öğrenilebilirlik ölçülecek.
Örnek sayısı, seedler, update ve checkpoint seçimi sonuç görülmeden yeni
config'te sabitlenir. Sayısal bütçe bu belgede sonuçlara göre seçilmiş gibi
gösterilmez; çalışma henüz başlamadı. Fiziksel robot modeli ve kabul A/B
eşikleri aynen kalır. Bu yeni veri yönü başarılı varsayılmıyor.

## Sonuçların karar ağacı

1. Aynı-kök yeni yönler de öğrenilemiyorsa veri miktarını artırmadan temsil/
   optimizasyonu incele: göreli girdilerin train-only ölçeği veya periyodik
   eklem temsili, tek seferde bir değişiklik.
2. Aynı-kök tanı iyileşip farklı-kök validation düşük kalırsa state-space
   genelleme/örnekleme sorunu lehine kanıt güçlenir; daha geniş kök kapsamı
   ve geometriyi kullanan mimari ayrı kontrolle ele alınır.
3. Local çözülüp wide başarısız kalırsa wide teacher dal seçimi, çoklu çözüm
   çıktısı ve eksik teacher dağılımı ayrı iş olarak yürütülür.
4. Bağımsız FK/Jacobian/etiket kontrolü bir yerde başarısız olursa öğrenme
   durur; ilgili Core/Foundations girdisi için ayrı hata kaydı ve sürümlü
   düzeltme açılır. Önceki G0 kaydı sessizce değiştirilmez.

Ürün kabulü üç seed main A≥%95 ve ilgili faz ölçütlerine bağlı kalır.
Bu takip tek başına C1-07, final üretimi veya G1 kabulü değildir. Şu anda
kullanıcının eski uzun eğitim komutunu tekrar çalıştırması gerekmiyor.

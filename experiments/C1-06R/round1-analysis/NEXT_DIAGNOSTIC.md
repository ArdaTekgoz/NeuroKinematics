# Round1 sonrası kontrollü tanı planı

9 Ekim 2026 · r1 · PLANNED / yeni eğitim NOT_RUN

## Amaç

%95 hedefi, A/B toleransları ve tam payda korunur. Round1'de hem train
pose hassasiyeti hem validation genellemesi yetersiz. Aynı kampanyanın
süresini uzatmak mevcut bulgularla gerekçelendirilmiyor. Önce aşağıdaki
mekanizmaları ayrı deneylerde ayıracağız; AI kod/protokol/tanı, kullanıcı
yalnız hazır uzun eğitim komutlarını çalıştıracak.

## Sıra ve geçişler

1. **Mevcut eklem durumunu koruyan öğrenme.** Aynı q_current ve hedef pose
   girdileriyle mutlak q çıktısı ile q_current + öğrenilen düzeltme çıktısı
   karşılaştırılır. Önce yalnız local train tanısı, sonra aynı modellerin
   local+wide koşulu; dört hücre bu iki etkiyi ayırır. Her eşli hücrede
   kapasite, seed, örnek sırası ve optimizer güncellemesi aynı tutulur.
   Local-only tanı için yalnız local validation sonucu ayrıca verilir;
   başarı iddiasında wide satırları paydadan çıkarılmaz. Yeni head ve
   başlangıç dönüşümü ADR ve sayısal config ile çalışmadan önce sabitlenir.
2. **Hassasiyet ve ölçek.** Sabit train alt kümelerinde gerçek A/B,
   normalized q kaybı ve fiziksel pose hatası birlikte izlenir. Küçük
   örnekte öğrenme ile geniş train kapsamı ayrılır. Erken/son checkpoint
   ayrışması izlenir. Yeni model küçük train tanısını geçmeden kullanıcıya
   uzun kampanya verilmez. Tolerans, label veya başarısız satır değiştirilmez.
3. **FK teriminin eklenme biçimi.** İlk kontrol yeterli hassasiyet sağlarsa,
   aynı supervised başlangıçtan Q ile Q+FK karşılaştırılır; mevcut rastgele
   başlangıçtan ortak kayıp sonucu tarihsel kontrol olarak kalır. Q/pose
   gradyan norm ve yönleri izlenir. Ağırlık, schedule veya warm start aynı
   deneyde kontrolsüz birlikte değiştirilmez. Yeni ağırlık/bütçe çalışmadan
   önce dondurulur; ölçülen çatışma tek başına bir düzeltmenin kanıtı değildir.
4. **Veri/temsil kararı.** Bu kontrollerin bulgusuna göre kök ailesi ayrılmış
   yeni train/validation kapsamı veya yönelim temsili için ayrı sürüm açılır.
   Daha büyük veri, farklı rotasyon temsili ve çoklu çözüm başlığı aynı
   anda eklenmez. Mevcut veri ve orijinal başarısız sonuçlar korunur.
5. **Yeni teslim kapısı.** Gerçek production metadata ile kesinti/devam,
   kaynak/hash eşliği, negatif kontroller ve makine üzerinde smoke PASS;
   dört hücreli tanıdan seçilecek en küçük adil üç-seed uzun deney matrisi,
   güncelleme bütçesi ve seçim kuralı dondurulur. Sonra kullanıcıya yeni
   PowerShell komutu verilir. Bu belge READY_FOR_USER_TRAINING değildir.

## Final sınırı

Üç seedli main validation ≥%95 kapısı karşılanmadan yeni final üretilmez
ve açılmaz. Tanı için seçilmiş local veya küçük train başarısı bu kapının
yerini tutmaz. H2-R ve ürün kararı ayrı; C1-07/G1 henüz başlatılmaz.

# C1-04 düşük başarı kararı ve C1-05 devri

3 Ekim 2026 · Karar: **DOĞRUDAN IK İÇİN NO-GO; C1-05 KONTROLLÜ ALTERNATİF**

## Ölçülen durum

E-C01'de iki model × üç seed aynı 15.204 etiketli train/3.249 etiketli validation satırını, aynı sıra ve 200 epoch/3.000 optimizer adımı bütçesini kullandı. [Eşli özet](E-C01-summary.md) ve [ham satırlar](.) validation sonuçlarıdır; test/10.000 sorguluk benchmark açılmadı. Altı koşunun **her birinde Profil A başarı 0/3.600** (sonlu, limit içinde, ≤2 mm ve ≤1° birlikte). Conditioned modelin median FK konum hatası 0,206–0,210 m, yönelim hatası 74,6–78,1°; pose-only için 0,571–0,585 m ve 121–124°. Conditioned validation q loss daha düşük olsa da kullanışlı hedef poza yaklaşma eşiği karşılanmadı. Ham limit dışı sonuçlar seed/model başına 120–269; sessiz kırpılmadı. Çarpışma/fiziksel güvenlik NOT_CHECKED.

## Teşhis ve sınırı

[Ambiguity diagnostic](ambiguity-diagnostic.json): train'de 8.400 kökün local/wide satırında pose-only özellikler birebir aynı. İki etiketin de bulunduğu 6.804 kökte `q_target` etiketleri arasındaki medyan L2 fark **5,815 rad**; validation'da 1.449 kökte **5,901 rad**. Aynı 7 özellikli fonksiyon iki farklı q etiketini aynı anda veremez; bu, pose-only supervised q kaybının yapısal bir alt sınırını oluşturur. Conditioned model `q_current` ile dalları ayırabilir ve q loss'u düşürmüştür, fakat bağımsız FK metriği hâlâ çok kötü. Bu, yalnız q etiket kaybını optimize etmenin hedef poz ölçütüyle uyuşmadığına **işaret eder**; tek başına bütün hata nedenini kanıtlamaz. Veri/etiket, split ve bağımsız FK Stage1/C1-02/C1-03 kapılarından geçmiş; yeni bir veri kusuru saptanmadı.

## Uygulama kararı

- Altı checkpoint **karşılaştırma baseline kanıtı** olarak saklanır; doğrudan IK çıktısı veya güvenli robot komutu olarak kullanılmaz. C1-04'ün deneysel sonucu olumsuz olabilir; T-C03 küçük veri öğrenmesi ve E-C01 kanıt tamamlanması ile görev kapanışı, operasyonel kabul değildir.
- Sıradaki iş roadmap'teki **C1-05 / E-C03**: aynı conditioned backbone, frozen C1-02 splitleri ve aynı üç seed üzerinde supervised kayba doğrulanmış Torch FK konum/yönelim terimlerini kontrollü ekle. Core raporundaki normalize kayıp ölçeği ve en çok sekiz validation konfigürasyonu önceden yeni configte dondurulur; önce küçük pilotla loss/gradyan/FK ölçeği incelenir. Her varyantın q loss'u, ham limit ihlali, bağımsız FK metriği ve Profil A başarısı aynı validation envanterinde kıyaslanır. Test seçime kapalı kalır.
- Limit ihlali ayrı etkidir: **E-C04** limit cezası, ardından gerekirse **E-C05** sınırlı başlık ayrı config/karar olarak sınanır. Delta-q/Res-MLP/6D veya çoklu hipotez eşzamanlı değiştirilmez; bu dosya onları onaylanmış çözüm saymaz. Sayısal DLS/harici kabul edilmiş baseline'lar ayrı karşılaştırma referansıdır; neural checkpointin kötü çıktısı başarı diye aktarılmaz.
- C1-05 varyantı da düşük Profil A başarısında kalırsa sonuç olumsuz kaydedilir, önceden kararlaştırılmış başka çözüm ailesi/yeni config/ADR açılır. Kabul eşikleri geriye dönük düşürülmez; C1-06 bağımsız nihai test kapısı değişmez.

Bu karar C1-05 eğitiminin yapıldığı veya iyileşmenin kanıtlandığı anlamına gelmez. C1-05 uygulaması **NOT_STARTED**.

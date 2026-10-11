# Core / Foundations ve literatür incelemesi

10 Ekim 2026 · Kaynak incelemesi + yerel ölçüm · Kabul kararı değildir

## Önce hangi sözleşme neyi kanıtlıyor?

| Aşama | Kanıtlanan kapsam | Bu tur değerlendirmesi | Açık kalan |
|---|---|---|---|
| F0-01/02 robot/FK | Aynı URDF/spec/TCP altında birim, sıra, dönüş, bağımsız hesap | Testler ve4096 cross-check PASS | Ortak kaynak modelinin fiziksel robot kalibrasyonu burada doğrulanmadı; bu, aynı modele göre üretilen hedefleri öğrenememeyi tek başına açıklamaz |
| F0-03 Jacobian | Base/TCP çerçevesi ve sayısal türev doğruluğu | Yeni örneklerde PASS; düşük condition grubunda da başarısızlık | Tekillik bazı sorguları zorlaştırır; bütün düşük başarıyı açıklamaz |
| F0-04 veri | LHS global örnekleme, grup split, deterministik FK etiketleri | Config ve devir sözleşmesi korundu | Ampirik kapsam, hassas inverse fonksiyonun öğrenilebilirlik garantisi değildir |
| F0-06 G0 | Kinematik/tekrar üretim kabulü | Eski G0 kararı değişmedi | Devir belgesi zaten neural başarıyı test etmediğini söylüyor |
| C1-02 çiftler | Bir local+bir wide; local±0,1rad, wide teacher seçimi/maskesi | 7000 local provenance birebir; önceki20.400 audit korunur | Her kökte tek local örnek; sparse global veri ve wide dal seçimi öğrenme tasarımı olarak yeniden ele alınmalı |
| C1-03 türevli FK | Türev ve FK doğruluğu | 110 test PASS; yeni smooth pose loss kontrolleri PASS | Fizik kaybının kullanılabilir olması öğrenmede etkili ağırlıklandırıldığı anlamına gelmez |
| C1-04/05 öğrenme | Tek-vektör regresyon ve kayıtlı Q/FK amaçları | Q/TCP ilişkisinin zayıflığı; pose-only takipte validation0 | Veri/temsil/amaç kombinasyonu hedef hassasiyeti sağlamadı |
| C1-06/R değerlendirme | Sabit A/B, tam payda, checkpoint ve oracle | Endpoint kusuru önceki sürümde ayrılmıştı; yeni decoder korunur | Validation defalarca görüldü; yeni bağımsız final olmadan yeni genelleme iddiası kurulamaz |

Kaynaklar: `experiments/F0-06/CORE_HANDOFF.md`, `experiments/F0-04/config.json`,
`experiments/C1-02/config.json`, `src/neurokinematics/data/pairs.py`,
önceki tanı2 pipeline audit ve bu tur sampling-review.json. Bu tur F0-04/05/06
tam üretim/benchmark kampanyası yeniden çalıştırılmadı; eski final açılmadı.

## Birincil araştırmalar ne söylüyor, bizim soruna ne kadar uyuyor?

**1. IKFlow — Ames, Morgan, Konidaris; RA-L2022.** Çözüm kümesini tek
vektör yerine dağılım olarak modeller. Appendix'te2,5 milyon eğitim noktası
ve büyük coupling ağları; sonuçlarda ham ağ doğruluğu ile sayısal refinement
ayrı ele alınır. Bizim14–15bin etiketli supervised train ve0,14–0,54milyon
parametrelik MLP'miz eşdeğer deney değildir. Bu fark “2,5milyon örnek gerekir”
kanıtı da değildir. Wide çoklu çözüm modellemesi için gerekçe sağlar;
local başarısızlığı yalnız dal ortalamasına bağlayamayız. Refined sonuçlar
bizim solver eklenmemiş direct IK kabulümüzün yerine kullanılamaz.
[Tam metin; VI-B/VII-A](https://arxiv.org/html/2111.08933v3).

**2. Neural Inverse Kinematics —2022.** Eklem zincirinde koşullu sıralı
örnekleme ile çoklu çözümleri ele alır. Deney metrikleri incelendiğinde
2D örneklerde başarı2cm, robot deneylerinde10cm konum eşiğiyle ölçülür.
Yüksek yüzdeyi bizim2mm ve1° ortak kriterine taşımak yanlış olur.
Bu kaynak mimari/dağılım araştırmasına dayanak verir; bizim kabul hedefimizin
mevcut MLP ile kolayca ulaşılabilir olduğunu kanıtlamaz.
[Tam metin; §5](https://arxiv.org/html/2205.10837v1).

**3. Zhou ve diğerleri, CVPR2019 — dönüş temsillerinin sürekliliği.**
SO(3)'ün düşük boyutlu temsillerindeki global süreksizlikleri ve5D/6D
alternatiflerini inceler. Bizim bu local2048 kümede göreli dönüş en çok19,17°,
w≥0,986; canonical quaternion π kesimini geçmiyor. Bu nedenle “quaternion
süreksizliği” bu local tanının gözlenen ana açıklaması olarak desteklenmedi.
Wide/global temsil için araştırma konusu kalır;6D girdinin burada çözüm
getirdiği henüz ölçülmedi.
[Yazarların kaydı](https://arxiv.org/abs/1812.07035).

**4. Wang, Teng, Perdikaris — gradyan dengesizliği,2020.** Birleşik fizik
kayıplarında farklı terimlerin geri yayılan gradyanlarını ölçüp dengelemeyi
inceler; algoritma1 bu istatistikleri kullanır. Çalışma PDE/PINN alanındadır,
bu robot için bir performans garantisi değildir. Bizde Q/P/R normlarının
farklı çıkması bu ölçüm yaklaşımını anlamlı kılar. Ancak local modellerde
Q–P/Q–R çatışması bulunmadı ve basit profile-scaled pose takibi yeterli
olmadı; “fizik kaybı ekleyince çözülecek” sonucuna gidilemez.
[Tam metin; algoritma1](https://arxiv.org/html/2001.04536v1).

**5. Lynch ve Park — Modern Robotics, singularities.** Jacobian rank
kaybını ve erişilebilir uç hızları arasındaki ilişkiyi açıklar. Bizim
ölçümümüz düşük condition512 kümesinde de train A1/512 olduğundan
“yalnız singular örnekler başarısız” hipotezini eliyor. Condition korelasyonu
tek başına nedensel değerlendirme değildir; quartile sonuçları birlikte okunur.
[Yazarların ders kaynağı](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-3-singularities/).

## Çıkarım ve araştırma önceliği

Bu ölçüm ve kaynaklardan yaptığımız çıkarım: mevcut sorun için güçlü bir
Foundations FK/Jacobian arızası kanıtı yok; Core'a geçerken global veri
kapsamının hassas yerel öğrenmeye yeterli görülmesi doğrulanmamış varsayım.
64 örnek ezberleme kapısının aşırı yorumlanması önceki tanıda zaten ortaya
çıkmıştı. Küçük Q, birkaç doğru eğitim örneği veya daha fazla epoch tek
başına hedefe yaklaşmanın güvenilir göstergesi değil.

Öncelik C1-02'nin sürümlü yeni öğrenme veri tasarımı ve C1-04'ün geometrik
temsilidir. Önce yerel bölge içi öğrenme ile yeni robot konfigürasyonlarına
genellemeyi ayıran kontrollü veri deneyi; ardından bulguya göre bir temsil
müdahalesi. Eşikleri düşürmek, eski girdileri yeniden yazmak veya birçok
etkeni aynı anda değiştirmek bu belirsizliği çözmez. Hedefin ulaşılamaz
olduğunu söylemek için de bu deneyler yeterli değildir.

Erişim tarihi10 Ekim2026. İlk üç araştırmanın arXiv kayıtları ve ilk iki
çalışmanın yöntem/deney bölümleri; PINN tam metninin gradyan dengeleme bölümü
ve yazar ders kaynağı incelendi. Kaynak sonuçları proje ölçümü gibi sunulmadı.

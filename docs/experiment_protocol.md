# NeuroKinematics - Deney Protokolü ve Baseline Karşılaştırma Altyapısı

## 1. Amaç (A-0.3)
NeuroKinematics'in geliştireceği "Fizik Farkındalıklı" (Physics-Aware) Nöral Ters Kinematik çözücüsü olan **Res-MLP** modelinin performansının, aynı veri ve test koşulları altında mevcut literatürdeki referans çözücülerle (baseline) objektif olarak karşılaştırılması amaçlanmaktadır.

## 2. Test Edilecek Baseline Modeller
1. **KDL (Kinematics and Dynamics Library):** Klasik Jacobian tabanlı iteratif IK çözücü.
2. **TRAC-IK:** KDL'ye göre daha gelişmiş, hem Jacobian hem de Gradient-Descent optimizasyonunu eşzamanlı çalıştırarak tekilliklerde daha dirençli sonuçlar veren açık kaynaklı C++ referansı.
3. **Baseline MLP:** FK, Joint Limit gibi fizik kayıplarının (physics loss) DAHİL EDİLMEDİĞİ, yalnızca MSE (Ortalama Kare Hatası) kayıp fonksiyonuyla standart backpropagation üzerinden eğitilmiş düz nöral ağ. Hedef: Fizik kayıplarının (Physics-Aware yaklaşımının) gerçek etkisini göstermek.
4. **Res-MLP (Bizim Modelimiz):** Girdi olarak $x_t$ ve $q_{t-1}$ alan, çıktı olarak 6D eklemlerin $(\sin, \cos)$ değerlerini üreten ve İleri Kinematik vb. diferansiyellenebilir fizik kurallarıyla cezalandırılan hibrit/gelişmiş mimarimiz.

## 3. Deney Seti ve Değerlendirme
Veri kümesi üzerinden seçilecek $N=10.000$ adet görünmeyen (unseen) hedef poz, 4 farklı modele beslenecektir. Her bir modelin çıktıları aşağıdaki kriterlere göre değerlendirilecektir:
- Ortalama ve maksimum Kartezyen mesafe hatası (mm cinsinden)
- 1 mm ve 5 mm hata payı toleransında IK Başarı Oranı (%)
- Milisaniye cinsinden donanım üzerindeki ortalama inference (çıkarım) süresi ve P99 latency.

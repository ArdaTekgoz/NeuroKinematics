# Sıfır başarının incelemesi ve mimari düzeltme

11 Ekim 2026 · Belge r1 · C1-06S / ADR-026.

## Önceki sıfır sonuç neydi?

Sıfır, bütün ağ çıktılarının aynı veya rastgele olması değildi. Sonlu ve
limit içindeki çıktının aynı anda <=2 mm ve <=1° sağlaması gereken
başarı sayacıydı. Önceki FK_TANH round1 modellerinin validation konum
medyanı 128,7–134,7 mm, yönelim medyanı 25,1–26,0° idi. Yalnız tolerans
sınırındaki yuvarlama bu açığı açıklamaz.

[Round1 kanıtı](../C1-06R/round1-analysis/RESULTS.md),
[D2 köken/öğretmen denetimi](../C1-06R/diagnostic2/RESULTS.md).

## Bulgular ve kesinlik sınırı

| Soru | Kanıt ve yorum |
|---|---|
| C1-06'da aniden mi bozuldu? | Hayır. C1-04 ve C1-05 validation sonuçları da sıfırdı. C1-06 bunu bağımsız değerlendirmede teyit etti. |
| Sentetik veri yanlış mı? | Önceki 20.400 train/validation köken, 18.453 teacher FK, robot/birim/TCP denetimleri geçti. Bu tur hedef eklemlerini almayan geometri çözümü validation 3.600/3.600 A/B; veri bu ölçütlerde çözülebilir. Sentetikten gerçek robota aktarım burada sınanmadı. |
| Gerçek kod kusuru var mıydı? | Önceki float32 joint decoder sınır kusuru ve TorchVersion checkpoint metadata kusuru kaydedildi. Decoder düzeltmesi eski modellerin sıfır validation sonucunu değiştirmedi. Bu kusurlar açıklamanın tamamı değildir. |
| Süreçte eksik kapı neydi? | Küçük 64 örnekte overfit, daha büyük veride hassasiyet ve yeni kök/yön genellemesinden güçlü kanıt sayıldı. Uzun kampanyadan önce bu üç ayrı davranışın ölçülmesi gerekiyordu. |
| Veri ölçeği yeterli miydi? | İlk train 8.400 kök/16.800 çift; bunların 1.596 wide etiketi yoktu. Tek kökten türeyen tekrarlar bağımsız kök çeşitliliği sağlamaz. Veri azlığının tek neden olduğu kanıtlanmadı. |
| MLP'nin temsil sorunu neydi? | Tek eklem vektörünü doğrudan öğrenirken çoklu IK dalları, tur eşdeğerleri, limitler ve yerel inverse hassasiyeti birlikte yaklaşıklandı. D8'de sıfır-hedef ofseti ve yerel türev uyumsuzluğu ölçüldü. D9 yalnız ofseti düzeltti, gerekli genellemeyi sağlamadı. |
| Yeni bileşen neyi değiştirdi? | Doğruluğun tamamını eklem regresyonuna yüklemek yerine, sabit robotun tam geometrisiyle dalları hesaplıyor ve FK ile kontrol ediyor. Bu, yeni bir geometri mimarisidir; başarısız MLP'nin yeniden eğitilip başarılı olması değildir. |

## Kaynak ölçeği karşılaştırması

CycleIK'in ilgili çalışması robot başına bir milyon eğitim pozu ve
6–8 katmanlı, milyonlarca parametreli ağlar bildirir; hedef eklem etiketini
taklit etmek yerine FK üzerinden pose kaybı kullanır. Bizim küçük conditioned
MLP, öğretmen etiketleri ve sınırlı veriyle aynı deney değildir. Bu karşılaştırma
bir milyon örneğin bu projeye başarı garantisi verdiği veya tek düzeltmenin
daha büyük ağ olduğu anlamına gelmez.
[Birincil kaynak, III-A/B/D ve Tablo I](https://arxiv.org/html/2404.08825v2).

## NeuroKinematics'in yönü

Yeni geometri bileşeni sabit robotta doğrulanmış aday/öğretmen sağlar.
Öğrenme daha sonra gerçek değer üretebileceği göreve bağlanmalıdır:
karmaşık maliyetler altında dal tercihi, sınırlı arama bütçesinin yönlendirilmesi
ve geometri bileşeni bulunmayan robotlarda başlangıç önerisi. Bunlar henüz
ölçülmedi. Basit en-yakın-dal seçimini ağla taklit edip bunu bilimsel katkı
saymak yeterli değildir. H1'de aynı tabana karşı ek fayda gösterilmelidir.

Yeni eğitim gerekirse ayrı veri sürümünde öğretmen eksiklikleri giderilmeli,
dal bilgisi korunmalı; yalnız 64 örnek ezberleme yerine train hassasiyeti,
yeni yön ve yeni kök validation kapıları kullanılmalıdır. Eski final yeni
eğitimin seçim seti yapılamaz. Bu tur yeni neural eğitim yapılmadı.

Yüksek kinematik başarı için somut yeni yol var. Eski bilimsel H2 hipotezi
ile sonraki Hybrid fazının H2-01 görev adı birbirinden farklıdır:
H2 hipotezini desteklenmiş göstermek, Hybrid'e geçiş şartı değildir.

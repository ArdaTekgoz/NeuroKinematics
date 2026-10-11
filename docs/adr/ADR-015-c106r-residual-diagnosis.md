# ADR 015 — Round1 sonrası eşli delta-q tanısı

9 Ekim 2026 · Kabul edilen kısa tanı; ürün/mimari kabulü değil

## Bağlam

Round1'de 12 modelin tüm validation başarıları sıfır. Local validation'da
q_current döndürmek dahi seçilmiş modellerden düşük medyan hata veriyor.
Kullanıcı sonraki tanıyı uygulamamızı, yetersizse proje zincirini kapsamlı
incelememizi istedi. Core raporu §3/§5 delta-q karşılaştırmasını kapsar.

## Karar

`diagnostic2/config.json` ile sekiz kısa train-only optimizasyon koşusu:
64/512 örnek × local/mixed × absolute/residual. Tek tanı seed'i, üç-seed
ürün kanıtı değildir. Aynı 13-256-256-256-6 SiLU ağı kullanılır. Son lineer
katman iki head'de de sıfır; absolute z=0.5+f(x), residual z=x[-6:]+f(x).
Başlangıç parametreleri eşittir; başlangıç tahminleri bilinçli olarak farklıdır.
Bu offset tasarımı mevcut eklem bilgisini korumanın etkisini ölçer; eski
round1 ağıyla fark yalnız head diye yorumlanmaz. Clamp/tanh yok; limit
ihlalleri ham çıktıda sayılır. Loss yalnız normalized Q, aynı AdamW/cosine
ve 5000 güncelleme. Son adım seçilir; başarısızlık epoch seçilerek gizlenmez.

Main train'de local ve etiketli wide satırı bulunan aynı kökler lexical
sırayla seçilir. Local hücre bütün köklerin local satırını, mixed hücre
alternatif köklerde wide satırını kullanır; örnek ve kök sayıları aynıdır.
Her hücre ortak local witness'ta ayrıca ölçülür. Bu tanı teacher başarısına
koşulludur; ürün değerlendirmesi değildir. Validation tüm 3600 satırla
raporlanır. Test/final açılmaz. 64 örnekte A 64/64 önceki küçük tanı hedefi
korunur; 512 sonuçları ölçek tanısıdır, yeni ürün kabul eşiği yaratılmaz.

## Sınırlar

Kısa tanı eğitimlerini AI çalıştırır; yeni uzun kampanya kullanıcıya teslim
edilmeden başlamaz. Mevcut veri, robot ve round1 kaynak hashleri değişmez.
Metadata yeni checkpointte düz string olur; eski TorchVersion dosyaları
yalnız dar kapsamlı uyumluluk okuyucusuyla açılır. Yetersiz sonuçta veri,
FK/Jacobian/gradyan, loss, etiket dalı, split ve metrik zinciri yeniden
denetlenir. Daha fazla epoch veya pozitif sonuç garantisi çıkarılmaz.

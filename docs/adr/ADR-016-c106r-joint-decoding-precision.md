# ADR 016 — Eklem dönüşümünde sınır hassasiyeti

10 Ekim 2026 · KABUL EDİLEN yeni sürüm düzeltmesi; eski sonuçlar korunur

## Kanıt

diagnostic2/project-review/pipeline.json: doğru öğretmen normalized
float32 çıktısı eski PhysicsLoss.raw ile fiziksel q'ya çevrilince 15.204
train etiketinin 100'ü, 3249 validation etiketinin 19'u katı float64
limit kontrolünde dışarı taşar. Azami q hatası yaklaşık 1e-6 rad.
Ham float64 teacher q'larının tamamı Profil B geçer. Yalnız dtype'ı
float64 yapmak train'de üç sınır taşmasını bırakır; endpoint formülü önemlidir.

## Karar

Yeni ayrı `c106r_precision.decode_joints` normalized tahmini float64'e
taşır ve en yakın endpointten affine hesaplar: z<0.5 için lower+z*span,
diğer yarıda upper−(1−z)*span. z=0 ve z=1 tam olarak tanımlı limitleri
üretir. Clamp, tolerans artırma veya limit dışı gerçek tahmini onarma yok.
Jacobian span'dır; autograd korunur. Kayıp/mimari değişmez.

Eski PhysicsLoss, round1/d2 ağırlıkları ve kayıtları değişmez. Yeni
dönüşümle oracle ve mevcut checkpointler ayrı etki analizi olarak ölçülür;
geçmiş sonuçlar yeniden PASS yazılmaz. Sonraki eğitim sürümü bu dönüşümü
ayrı config/code hash ile kullanacak; mevcut eğitim paketine sessizce
eklenmez. Düzeltmenin sıfır başarıyı açıklayıp açıklamadığı gerçek yeniden
ölçümle raporlanır; pozitif sonuç varsayılmaz.

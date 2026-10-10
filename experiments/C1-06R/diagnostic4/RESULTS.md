# C1-06R tanı 4 — Optimizasyon, kayıp ölçeği ve kapasite

10 Ekim 2026 · Dört koşu tamamlandı · Denetim PASS · Ürün hedefi NOT_MET

Aynı 2048 local train örneği, göreli13-boyut girdi, residual head ve
seed2026100901 kullanıldı. Ön kayıt: [config](config.json),
[ADR-018](../../../docs/adr/ADR-018-c106r-optimization-scale-capacity.md).
Üç koşul referansın bir ayar grubunu değiştirir. Eski kanıtlar korunur.

## Ölçülen sonuç

Profil A: sonlu ve limit içi eklemler, konum≤2mm ve yönelim≤1° birlikte.
Profil B: konum≤1mm ve yönelim≤0,5°. Eşikler değişmedi. Medyanlar tam
paydada nearest-rank; geçersiz satırlar +∞ kabul edilir.

| Koşul | Train A /2048 | Train medyan mm / derece | Local validation medyan mm / derece | Tam validation A /3600 |
|---|---:|---:|---:|---:|
| Referans:256, AdamW | 0 | 15,65 / 3,95 | 19,57 / 4,98 | 0 |
| Optimizer paketi:L-BFGS | 0 | 23,77 / 4,66 | 26,33 / 5,23 | 0 |
| Q kayıp ölçeği×1e6 | 0 | 14,88 / 3,81 | 19,23 / 4,98 | 0 |
| Katman genişliği512 | 3 | 10,94 / 2,89 | 19,65 / 5,55 | 0 |

Her koşulda train B0/2048, validation A/B0/3600 ve main A/B0/3000.
Local validation1800, wide1800; wide örnekler başarı paydasından çıkarılmadı.
Local eğitimli bu dört modelin wide medyanları limit ihlalleri nedeniyle
sonlu değil. Wide başarısızlık genel dağılımda kabul edilmediğimizi gösterir;
local-train hassasiyetindeki başarısızlığı tek başına açıklamaz.

| Koşul | Parametre | Son ölçeklenmemiş Q | Optimizer iterasyonu | Gradyan hesabı | Eğitim süresi, s |
|---|---:|---:|---:|---:|---:|
| Referans | 136710 | 0,000323362 | 5000 | 5000 | 17,20 |
| L-BFGS | 136710 | 0,000462613 | 5000 | 5457 | 59,74 |
| Q×1e6 | 136710 | 0,000306935 | 5000 | 5000 | 16,70 |
| Genişlik512 | 535558 | 0,000174118 | 5000 | 5000 | 18,03 |

Toplam kampanya124,46s; bu süre veri hazırlığı, ölçüm ve reload'u içerir.
Eşit hesap bütçesi iddiası yok: L-BFGS closure sayısı ve geniş ağ maliyeti
farklıdır. Bütün modeller sıfırdan başladı; terminal checkpoint ölçüldü.

## Hangi açıklamalar destekleniyor?

- Referans tanı3 son ağırlıkları ve metriklerini birebir yeniden üretti.
  Yeni koşuların farklı bir veri/decoder tabanında karşılaştırılması sorunu yok.
- Bu L-BFGS paketi ve bütçesi AdamW'den daha kötü. Bu sonuç tüm L-BFGS
  ayarlarını elemez; schedule, stopping ve decay optimizer paketi içindedir.
- Global kayıp çarpanı küçük sayısal farklar yarattı, A başarısını artırmadı.
  Bu deney fizik kaybını, eklem bazlı ağırlıkları veya Q/FK dengesini test etmez.
- Geniş ağ eğitim Q kaybını %46,15 azalttı; train A3/2048 (%0,15).
  Ancak local validation konum medyanı19,57→19,65mm, yönelim4,98→5,55°.
  Kapasite bu bütçede öğrenmeyi etkiliyor; tek başına hedef hassasiyeti ve
  genellemeyi sağlamıyor. Tek seed ile genel bir nedensel sonuç çıkarılmaz.

## Hata bileşenleri — sonuç sonrası betimsel inceleme

Bu bölüm ön kayıtlı model seçimi değildir. Sabit son tahminler üzerinde
ek incelemedir; yeni eğitim veya validation'a göre checkpoint seçimi yapılmadı.

| Train satırları | Referans | L-BFGS | Q×1e6 | Genişlik512 |
|---|---:|---:|---:|---:|
| A başarılı | 0 | 0 | 0 | 3 |
| Yalnız konum başarısız | 46 | 24 | 56 | 121 |
| Yalnız yönelim başarısız | 5 | 1 | 10 | 19 |
| İki pose eşiği de başarısız | 1976 | 2000 | 1955 | 1886 |
| Limit nedeniyle geçersiz | 21 | 23 | 27 | 19 |

Geniş ağda1886/2048 (%92,09) satır iki pose eşiğini birlikte aşıyor;
19/2048 limit ihlali bu yaygın başarısızlığı açıklamıyor. NaN/Inf yok.
Eklem bazında train RMSE yaklaşık0,96–2,16°; en yüksek joint4 ve joint5.
Bu iki eklemin hatalarının TCP hatasına nedensel katkısı henüz ölçülmedi.
Teacher'ın q_current'a göre düzeltme RMSE'si her eklemde3,29–3,35°;
model düzeltmeyi kısmen öğreniyor. Sabit açısal bias çok küçük, kalan hata
örnek bazında değişiyor. Bunun kök nedeni henüz kesinleşmedi.

[Ham ayrım](error-decomposition.json), [dört sonuç](results.json),
[denetim](audit.json), [çalışma kaydı](RUN_REPORT.md).

## Karar

Yeni uzun kampanya hazır değil; mevcut eğitim komutu tekrar çalıştırılmayacak.
Üç seed main A≥%95 kapısı karşılanmadı. Eski H2 REJECTED korunur;
yeni final NOT_CREATED, C1-07/G1 açık. Sonraki odak daha fazla rastgele
hiperparametre denemek yerine train örneklerinde kalan hatanın geometrik
duyarlılığı ve öğrenilebilirliğini ölçmektir; [takip planı](NEXT_DIAGNOSTIC.md).

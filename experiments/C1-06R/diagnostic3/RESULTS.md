# Göreli pose deneyi — sonuç

10 Ekim 2026 · r1 · **16 koşu COMPLETE / audit PASS / validation hedefi NOT_MET**

Deney yaklaşık 5 dakika 1 saniyede tamamlandı. 512/2048 train örneği ×
local/mixed × ham/göreli pose × absolute/residual çıktı; her hücre 5000
full-batch AdamW güncellemesi, toplam 80.000 güncelleme. Tek seed
2026100901. Tüm modeller aynı 13-256-256-256-6 SiLU kapasitesinde ve
aynı başlangıç parametreleriyle kuruldu. Son adım raporlandı.

**Bütün 16 modelde validation Profil A ve B: 0/3600.** Main 0/3000,
local 0/1800, wide 0/1800. Göreli temsil bazı sürekli hata ölçülerini
iyileştirdi; %95 hedefi veya herhangi bir Profil A başarı artışı sağlamadı.

## Eşli karşılaştırmalar

Oklar aynı hücrede **ham girdi → göreli girdi** değişimini gösterir.
Hata sütunları yalnız local validation'ın tam 1800 satırını kapsar;
ürün değerlendirmesinden wide satırlar çıkarılmadı.

| Train N | Karışım | Çıktı | Train A sayısı | Local medyan konum, mm | Local medyan yönelim, ° |
|---|---|---|---|---|---|
| 512 | local | absolute | 0 → 0 | 58,12 → 37,29 | 9,47 → 5,86 |
| 512 | local | residual | 8 → 3 | 78,73 → 39,34 | 12,34 → 7,99 |
| 512 | mixed | absolute | 487 → 179 | ∞ → 47,53 | ∞ → 9,16 |
| 512 | mixed | residual | 419 → 186 | ∞ → 43,31 | ∞ → 7,32 |
| 2048 | local | absolute | 0 → 0 | 47,19 → 35,31 | 7,57 → 5,42 |
| 2048 | local | residual | 0 → 0 | 47,82 → 19,57 | 8,05 → 4,98 |
| 2048 | mixed | absolute | 8 → 88 | ∞ → 47,30 | ∞ → 9,87 |
| 2048 | mixed | residual | 0 → 0 | ∞ → 47,57 | ∞ → 9,32 |

∞, geçersiz q'ların paydada +∞ sayılması nedeniyle medyanın sonlu
olmamasıdır; JSON'da null saklanır. Başarısız satırlar elenmedi. Her
hücrenin wide/family, p95, ihlal ve ham satır sonuçları ayrıca kayıtlıdır.

2048-local-residual'daki 19,57 mm/4,98° düşüşü seçilmiş bir **betimsel
örnektir**, en iyi seed/modelin ürün kabulü değildir. Profil A ≤2 mm
ve ≤1°, ayrıca limit içinde q gerektirir. Local sonuçta iyileşme olsa
da bu modelin wide medyanı sonlu değil; wide sorguların yarısından
fazlası geçersiz. Genel IK başarı sonucu çıkarılamaz.

## Deney neyi ayırdı?

Göreli girdi: base çerçevesinde p_target−p_current, TCP current çerçevesinde
R_current.T @ R_target'ın kanonik wxyz quaternion'u, aynı normalize
q_current. Konum farkı mean/std'si bütün 16.800 train girdisinden
hesaplandı. Girdiler 13 boyutta kaldı; teacher q, pair_mode ve split
girdi olarak kullanılmadı. Yeni endpoint decoder bütün hücrelerde ortak.
Ön işlem bir FK hesabı içerir; tahmine IK çözücü adımı eklenmedi.

Göreli koordinatlar ve bunların train-only konum ölçeklemesi birlikte
değişti. Etkiyi yalnız quaternion veya yalnız ölçek değişimine atfetmeyiz.
512 ve 2048 ölçeklerinde güncelleme sayısı aynı, full-batch örnek maruziyeti
farklıdır; eşit duvar süresi iddiası yoktur. Mixed tanı kökleri öğretmen
etiketi bulunanlardan seçildi; validation'da eksik label satırları korundu.

**Gösterilen:** Bu seed ve bütçede göreli temsil local hata ölçülerini
iyileştirebiliyor. Dört 512-raw kontrolün tüm ağırlık tensörleri tanı2 ile
birebir aynı; kontrol eğitiminin kayması gözlenen farkı açıklamıyor.

**Gösterilmeyen:** Geniş başlangıç başarısı, üç-seed tutarlılığı, %95
ürün başarısı veya H2-R desteği. Tek seed üzerinde denenen 16 hücre aynı
validation sorgularını kullanır; bağımsız testler olarak havuzlanmaz.

**Darboğaz:** 2048-local hücrelerin dördünde de train A0/2048. Sorun
yalnız yeni sorgulara genelleme değil; bu mimari/kayıp/optimizer/bütçe
kombinasyonu eğitim satırlarında da pose hassasiyetini sağlayamıyor.
Bu, kesin bir kapasite sınırı veya tek başına optimizer hatası kanıtı
değildir. Temsil değişimini tamamlanmış çözüm saymak desteklenmiyor.

## Doğrulama ve karar

Dört yeni test PASS: bağımsız FK ile hedefi geri kurma, sıfır hareket
kimliği, teacher/metadata bağımsızlığı ve quaternion işaret eşliği,
train-only fit ve girdi boyutu. 16 checkpoint güvenli yüklenerek tüm
validation çıkarımları birebir tekrarlandı. Son audit: 16 hücre,
80.000 güncelleme, ham satır paydaları, SHA, eşli örnek kimlikleri ve
dört raw kontrolün tensör eşliği PASS. Tarihsel 122 frozen girdi korundu.
Önceki 474 test bu ek deney için yeniden çalıştırılmadı.

Bu deney tamamlandı. Yeni final NOT_CREATED; eski final ham sorguları
NOT_READ; C1-07/G1 açık. Yeni uzun eğitim paketi hazır değil. Bir sonraki
araştırma, geniş train üzerinde hassasiyetin neden sağlanamadığını
ayırmalı: aynı veri/temsil üzerinde optimizer/kayıp ile model kapasitesi
ayrı kontrollerde sınanmalı. Göreli temsil local için aday olarak saklanır;
wide için çözülmüş sayılmaz. Yeni konfigürasyon çalışmadan önce kaydedilir;
bu sonraki deney henüz NOT_RUN.

[Makine sonuçları](results.json) · [eşli audit](audit.json) ·
[çalışma kaydı](RUN_REPORT.md) · [ADR-017](../../../docs/adr/ADR-017-c106r-relative-pose-diagnosis.md)

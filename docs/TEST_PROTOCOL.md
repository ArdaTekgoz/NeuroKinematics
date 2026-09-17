# Ortak test ve benchmark protokolü

Belge r1 · 17 Eylül 2026 · Tasarım hedefleri; mevcut deney sonucu yok

## Deneyden önce dondurulacaklar

Robot ve TCP hashleri; joint sırası; birimler; reference backend; karakteristik uzunluk; veri ve split hashleri; sorgu listesi; q_current üretim kuralı; görev toleransı; bütçe; seed; solver ayarları; model seçim metriği; birincil hipotez ve minimum etki.

Bir eşik sonuç görüldükten sonra değiştirilirse yeni protokol sürümü açılır. Önceki deney yeni hedefi karşılamış gibi yeniden etiketlenmez. Son test hata haritası ile training iyileştirilecekse yeni bağımsız final test gerekir.

## Örnek ve tekrar birimleri

Ana IK testi en az 10.000 sorgu; 50/50 yerel/geniş başlangıç. Zor alt kümeler en az 1.000 sınır ve 1.000 tekillik örneği. Örtüşen örnekler toplama iki kez girmez. En az3 eğitim seed'i; gecikme ölçümü 5 geçiş. Yörünge en az 30 bağımsız dizi ve dizi başına 200 zamanlı örnek. Aynı sorgunun tekrarları bağımsız hedef sayılmaz.

## Ortak kabul

Kinematik geçerlilik = sonlu ve doğru boyut + limit uyumu + konum ve yönelim toleransı. Profil A 2 mm/1 derece; Profil B 1 mm/0.5 derece. Collision ve deadline ayrı alanlardır. NOT_CHECKED kontrolü geçmiş sayılmaz. Çözücü başarısızlığı erişilemezlik kanıtı değildir.

## Raporlanacak alanlar

N ve subgroup; başarı/geçersiz/limit/timeout oranları; konum/yönelim median-P95-P99; tüm sorgular ve başarılı sorgular için P50-P95-P99 toplam süre; deadline kaçırma; iteration varsa gerçek sayı; refinement/restart oranları; CPU/GPU/RAM; soğuk başlangıç; model/veri boyutu; eğitim emek ve compute süresi.

## İstatistik

Yöntemler aynı query_id üzerinde eşleşir. Bootstrap sorgu ailesi veya trajectory düzeyinde yapılır; eğitim seed'i değişkenliği ayrıca sunulur. Birincil karşılaştırma ön kaydedilir; diğer taramalar keşifseldir. H1 için süre oranı CI üst sınırı 1 altında, başarı farkı CI alt sınırı −1 yüzde puanından kötü olmamalıdır. H2 için hedef etki ve CI ayrı yorumlanır; belirsiz sonuç açıkça yazılır.

## Hata ve regresyon testleri

Yanlış frame, ters quaternion sırası, sıfır norm, NaN/Inf, eksik limit, yanlış robot ağırlığı, limit dışı seed, kolay hedef, kanıtlı dış hedef, timeout, geç dönen worker, çarpışma kontrolü yapılmamış sonuç, bozuk paket ve kesilen iş senaryoları bulunmalıdır. Test kimlikleri TRACEABILITY dosyasındaki görevle bağlanır.

## Geçerliliğe yönelik tehditler

Yanlış ortak URDF; sentetik veriyle fiziksel doğruluğu karıştırma; teacher seçim yanlılığı; yakın örnek/trajectory sızıntısı; testle tuning; unequal compute; yalnız başarılı sorgularda süre seçme; farklı task/donanım makale sonuçlarını doğrudan sıralama; düşük seed sayısı; gerçek robot ölçümünün yokluğu. Bunların her deney üzerindeki etkisi sonuç raporunda belirtilir.

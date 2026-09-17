# İddialar hipotezler ve kanıt durumu

| Kimlik | İddia veya soru | Kanıt şartı | Mevcut durum |
|---|---|---|---|
| H1 | Neural seed sayısal çözücüye yarar mı | Aynı çözücü/bütçede toplam P95 en az yüzde 20 düşüş; başarı kaybı en çok 1 yüzde puanı; CI | PLANLANDI |
| H2 | FK ve limit bileşenleri zor hedeflerde yarar mı | Zor alt kümelerde en az 2 yüzde puanı hedef; genel düşüş en çok 1 yüzde puanı; paired/seed analizi | PLANLANDI |
| H3 | Aynı pipeline ikinci robotta tekrar çalışır mı | Yeni robot config/varlıklarıyla aynı kod; müdahale ve sonuç kaydı | PLANLANDI |
| Q1 | Core yüzde 95 kinematik başarı | 2 mm/1 derece profilinde bağımsız ana test | ÜRÜN HEDEFİ; SONUÇ DEĞİL |
| Q2 | Çarpışmasız çıktı | Etkin geometri ve açık doğrulama kapsamı | v2 öncesi NOT_CHECKED |
| Q3 | Gerçek zamanlı servis | Tanımlı deadline ve kaçırma oranı | Hard real-time iddiası yok |
| Q4 | Evrensel robot modeli | Ayrık robotlarda zero/few-shot deney | Ana sürüm iddiası değil |
| Q5 | Fiziksel doğruluk | Bağımsız ölçüm, kalibrasyon, robot deneyi | ARAŞTIRMA BİRİKİMİ |
| Q6 | Ticari fayda | Kullanıcı işi/süre/maliyet ve talep araştırması | DOĞRULANMADI |

Physics-aware/model-informed terimi tercih edilir. FK regularizasyonu; hareket güvenliği, dinamik uygulanabilirlik, mekanik ömür artışı veya sınır koşullarının her durumda sağlanmasıyla eş anlamlı değildir. Sonuç bulunmadan literatürde ilk, rakiplerden hızlı veya endüstriyel olarak hazır ifadesi kullanılmaz.

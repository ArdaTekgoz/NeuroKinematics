# Bağımlılık ve robot varlığı envanteri şablonu

Bu kayıt F0-01'de başlatılır, S3-04 paketlemesinde kesin sürümlerle tamamlanır. Boş şablon lisans incelemesinin yapılmış olduğu anlamına gelmez.

| Alan | Kaydedilecek bilgi |
|---|---|
| Bileşen kimliği ve türü | Paket, URDF, mesh, veri veya model |
| Kaynak | Resmi depo veya sağlayıcı bağlantısı |
| Kesin sürüm | Release, commit veya dosya SHA256 değeri |
| Lisans | Kullanılan sürümün lisans metni ve dosya yolu |
| Projedeki rol | Geliştirme, eğitim, runtime veya dağıtılan varlık |
| Dağıtım biçimi | Pakete dahil, sistem bağımlılığı veya indirilen ayrı varlık |
| Bildirimler | Korunacak telif ve lisans dosyaları |
| Değişiklikler | Kaynaktan farklılaştırılan içerik |
| İnceleme durumu | İNCELENMEDİ, AÇIK SORUN veya KAYIT TAM |
| İlgili karar | ADR ve görev kimliği |

Her bileşen için bu alanlarla ayrı kayıt açılır. Ana yazılım, robot meshleri ve veri kaynaklarının aynı lisansa sahip olduğu varsayılmaz. Paketlemede kullanılan kilit dosyasından makine tarafından okunabilir bileşen listesi üretilir; manuel varlık kayıtları buna eklenir. Açık dağıtım sorunu çözülmeden ilgili varlık yayın paketine alınmaz.

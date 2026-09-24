# Foundations r2 belge aktarımı çalışma kaydı

Kimlik: RUN-20260924-F006-DOCS-R2
Durum: TAMAMLANDI
Görev ve gereksinim: F0-06 kapanış belgelerinin yayımlanması
Tarih ve sorumlu: 24 Eylül 2026 · Codex; proje sahibi Arda Tekgöz
Yazılım hedefi: v0.1.0 · Belge revizyonu: r2

## Soru ve değişiklik

Kullanıcının sağladığı güncel README, Foundations sonuç raporu ve F0-05 başarı
grafiği depodaki kanonik konumlarına aktarıldı. Uygulama kodu, deney girdileri,
kabul eşikleri ve önceki r1 raporu değiştirilmedi.

## Tekrar üretim

- Başlangıç dalı: `main`; başlangıç commit'i: `93e9111e1e43e37223142c0fab84642a42201a7d`.
- Kaynaklar: yerel `Downloads` klasöründeki kullanıcı tarafından sağlanan üç dosya.
- Hedefler: `README.md`, `docs/raporlar/01_Foundations_v0_1_sonuc_r2.md`,
  `figures/foundations_f05_basari_r2.svg`.
- Aktarım sonrasında her kaynak/hedef çifti SHA-256 ile karşılaştırıldı.
- Kullanıcıya ait `~$uroKinematics_Model_Kullanim_Plani_r1.docx` geçici Word
  dosyası korunmuş ve commit kapsamı dışında bırakılmıştır.
- Ortam, robot/TCP, veri/split, model/checkpoint ve seed bilgileri: UYGULANMAZ;
  bu çalışma yalnızca belge aktarımıdır.

## Test ve ham kanıt

| Kontrol | Sonuç | Durum |
|---|---|---|
| Üç kaynak/hedef SHA-256 çifti | Her çift aynı | PASS |
| SVG XML ayrıştırması | Hatasız | PASS |
| Markdown yerel bağlantı hedefleri | Eksik hedef yok | PASS |
| `git diff --check` | Sağlanan sonuç raporunun 3–5. satırlarındaki bilinçli Markdown satır kırımları dışında bulgu yok | PASS_WITH_NOTE |
| Yazılım regresyon testleri | Doküman-only değişiklik nedeniyle çalıştırılmadı | NOT_RUN |

Komutlar ile XML ve bağlantı denetimi çıktıları commit öncesi terminal oturumunda
doğrulandı. `git diff --check` uyarıları kullanıcı kaynağındaki üç Markdown hard
break ile sınırlıdır; kaynak/hedef SHA-256 eşliğini korumak için değiştirilmedi.
Kalıcı yeni deney verisi üretilmedi.

## Sonuç ve yorum

Foundations r2 sonuç anlatımı ve ona bağlı grafik, README'deki güncel faz özetiyle
birlikte depoya eklendi. Bu aktarım mevcut G0 kararını değiştirmez ve yeni bir
kinematik ya da performans iddiası üretmez.

## Sonraki adım

Değişiklikler tek bir dokümantasyon commit'i olarak `origin/main` dalına
gönderilecek; ardından uzak dal hızlı-ileri-sarma kuralıyla tekrar çekilerek
yerel/uzak eşitliği doğrulanacaktır.

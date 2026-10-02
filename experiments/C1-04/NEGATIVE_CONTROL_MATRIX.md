# C1-04 negatif kontrol matrisi

2 Ekim 2026 · Aşama 1 ön kayıt. Bütün kontroller Aşama 2'de uygulanacak; burada sonuç **NOT_RUN**.

| Kod | Değiştirilen tek unsur | Beklenen reddetme ve kanıt |
|---|---|---|
| N01 | Etiketi olmayan wide satırın altı NaN sentinelini supervised batch'e kat | Mask/count uyuşmazlığında loader durur; satır geometri envanterinde kalır. |
| N02 | `label_present=true` iken bir q_target bileşeni NaN/Inf | Veri doğrulayıcı model kurmadan reddeder. |
| N03 | `q_current` veya etiket q bir eklemde donmuş limitin dışına çıkar | Limit doğrulayıcı reddeder; q kırpılmaz. |
| N04 | Girdi position/quaternion veya conditioned q sırasını değiştir | Alan isimli feature manifesti/checkpoint input sırası uyuşmazlığı reddeder; yalnız width kontrolü yeterli sayılmaz. |
| N05 | 64 kontrollü train kökünde `q_target` dizisini bir konum döndür; pair_id ve hedef pose sabit | Bağımsız Pinocchio FK ile etiket-hedef Profile B audit'i ilk uyuşmazlıkta durur; eğitim başlamaz. |
| N06 | Aynı `group_id`yi train ve validation arasında taşı | Split/soy audit'i reddeder; seed/split yeniden yazılmaz. |
| N07 | Robot/TCP/config/scaler/dataset/Torch FK hashinin birini değiştir | Checkpoint loader ve giriş audit'i reddeder. |
| N08 | Quaternionu `xyzw`, non-unit veya kanonik olmayan işaretle besle | Şema/kanoniklik denetimi açık hata verir. |
| N09 | Ham model q değerini limit dışı üret | Çıkarım ham değeri ve limit ihlali sayısını kaydeder; FK'ye sokmaz, başarı saymaz. |
| N10 | NaN/Inf train loss veya validation loss oluştur | Koşu abort olur; checkpoint seçilmez. |
| N11 | Kapalı test veya benchmark satırını validation seçim kaynağına sok | Split allowlist reddeder; test metriği model seçimine girmez. |

Bu mutasyonlarda hata mesajı, test kimliği, komut ve exit kodu ham JUnit/JSON'a yazılacaktır. Beklenmeyen geçiş T-C03'ü FAIL yapar; eşik yeniden yorumlanmaz.

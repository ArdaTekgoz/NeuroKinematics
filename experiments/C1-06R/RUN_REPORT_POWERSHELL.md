# Deney veya uygulama kaydı

Kimlik: RUN-20261009-C106R-POWERSHELL

Durum: PASS / READY_FOR_USER_TRAINING; ana eğitim NOT_RUN

Görev ve gereksinim: C1-06R; kullanıcı bilgisayarında eğitim başlatma komutunun çalışması

Tarih ve sorumlu: 9 Ekim 2026; AI uygulama ve denetim, kullanıcı uzun eğitim

## Soru ve değişiklik

Kullanıcının terminalinde `pwsh` bulunamadığı için başlatıcı çalışmadı.
Yönerge r2 ve delivery.json komutu Windows ile gelen `powershell.exe` olarak
düzeltildi. PowerShell 7 kurulumu gerekmez. Dondurulmuş launcher, model,
veri, eğitim ayarları ve kabul eşikleri değiştirilmedi.

## Tekrar üretim

Başlangıç commit: `ce16318a1eaee3b2150b5d30a7bd98cd0798247b`.
Çalışma ağacında kullanıcıya ait STATUS/TRACEABILITY değişiklikleri ve
PDF/DOCX/AUDIT dosyaları vardı; bu düzeltmenin kapsamına alınmadı.
Windows PowerShell sürümü: `5.1.26100.9444`.
RTX 5060 Laptop 8 GB / 24 GB RAM; mevcut exact CUDA overlay kullanıldı.
Runtime, robot/TCP, veri ve kaynak kimlikleri değişmeyen training-freeze.json
ile doğrulandı; bu dosyanın SHA-256 değeri
`cc7c8094611bf4fc26047d9189a1276692845d2909c98be6730d4de31df30642`.
Bu kontrol eğitim veya yeni checkpoint üretmedi; seed/config değişmedi.
Thread/runtime ortamı ve zamanlar command.json içinde kayıtlıdır.

```powershell
python scripts/c106r_command.py 019-windows-powershell-check-only -- powershell.exe -NoProfile -File scripts/Start-C106RTraining.ps1 -CheckOnly
```

Başlangıç/bitiş UTC: 2026-10-09 17:08:40.819332 / 17:08:48.249599.

## Test ve ham kanıt

- Windows PowerShell başlatıcı kontrolü: exit code 0, PASS.
  Kanıt: commands/019-windows-powershell-check-only/{command.json,stdout.log,stderr.log}.
- İç paket kontrolü: exit code 0, PASS.
  Kanıt: commands/user-preflight-20261009-200841/.
- Dondurulmuş girdiler, exact runtime, GPU ve boş disk kontrolü: PASS;
  stdout kaydında sonuç mevcut. Başarısız kontrol yok.
- Kontrol sonrasında data/generated/C1-06R/round1 mevcut değildi.
- Ana eğitim NOT_RUN; eğitim başarısı ve tam kampanya süresi ölçülmedi.
  Önceki 471 test bu belge düzeltmesi için yeniden çalıştırılmadı.

## Sonuç ve yorum

Sorun eğitim algoritmasından önce, yönergede PowerShell 7 komutunun
varsayılmasından kaynaklandı. Windows PowerShell 5.1 ile mevcut başlatıcının
hazırlık yolu gerçekten çalıştı. Eski pwsh denetimi kendi ortamında geçen
tarihsel kanıt olarak korunur; kullanıcının PATH ortamını doğrulamıyordu.
Ana eğitim hazır oluş durumu ve Core kabul durumu değişmedi.

## Sonraki adım

Kullanıcı USER_TRAINING.md r2 içindeki powershell.exe komutunu çalıştırır.
Tamamlanma veya hata sonrası yerel çıktıları AI inceler. Yeni final test
NOT_CREATED; C1-07 başlamaz. Teslim manifesti yeni belge ve ham kayıtları kapsar.

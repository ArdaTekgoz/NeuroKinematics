# C1-04 komut planı ve gerçek Aşama 1 kaydı

Depo kökünden PowerShell. `input-hashes.json` yazımı yalnız Aşama 1 dondurması içindir. Aşama 2'de sadece `--check`; drift varsa manifest sessizce yenilenmez. Komut planındaki neural CLI/testler henüz yoktur ve **çalıştırılmış sayılmaz**.

## Aşama 1 gerçek komutlar

Komut, zaman, exit kodu ve stdout/stderr [stage1-commands.json](stage1-commands.json) içinde tutulur. Kanonik kontrol:

```powershell
python scripts/check_c104_stage1.py --check
```

Bu kontrol kaynak dosyalarının ve 34 yerel shardın SHA-256 kimliğini, C1-02 manifest/normalization eşliğini, kabul sayısını ve C1-03 Torch FK kaynak hashini karşılaştırır. Neural öğrenme veya FK performansı ölçmez.

## Aşama 2 planı · yalnız açık onay sonrası

1. `git status`, HEAD/origin ve `python scripts/check_c104_stage1.py --check` ile Stage1 kimliğini denetle. Her driftte beklenen/bulunan/yol/etki yazıp dur.
2. C1-03 hashli Torch overlay ve `pixi.lock` ile gerçek Windows CPU ortamı; import, `pip check`, exact sürümler. Kod henüz yokken komut uydurma.
3. Veri yükleyici, model, checkpoint/çıkarım ve veri/etiket/FK doğrulamasını uygula. Pozitif, N01–N11 negatif ve küçük sentetik unit testleri gerçek CLI/JUnit ile kaydet.
4. Pilot RAM/süre gözlemi; ardından T-C03 küçük 64/32 kontrollü deney. İki model gerçek etiket öğrenme ve N05 yanlış eşleme kapısı geçmezse tam eğitime geçme.
5. E-C01 iki model × üç seed, frozen train/validation ve eşli sıra/bütçe. Epoch JSONL, checkpoint ve validation FK sonuçlarını hashle; kötü/negatif sonuçları koru.
6. Yüklenen checkpoint ile sabit validation inference/FK, ilgili regresyonlar ve temiz checkout/ortamda en az yükleme+inference+kanıt audit. İkinci tam eğitim yoksa eğitim tekrarını kanıtladı deme.
7. Gerçek komutlar, exit kodları, ham log/JUnit, checkpoint yol/byte/SHA, ortam ve karar tarihli Aşama 2 kaydına eklenecek. C1-05 devri yalnız tamamlanmış C1-04 kararından sonra.

Ham büyük shard, aday ve model ağırlıkları `LOCAL_ONLY`; uzak arşiv `NOT_CONFIRMED`.

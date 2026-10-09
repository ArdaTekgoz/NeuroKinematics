# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-DIAGNOSTIC3

Durum: COMPLETE_DIAGNOSIS / AUDIT PASS / VALIDATION_TARGET_NOT_MET

Görev ve gereksinim: C1-06R; REQ-C03–05; göreli pose eşli temsil tanısı

Tarih ve sorumlu: 10 Ekim 2026; kullanıcı onayı sonrası AI kısa tanı/analiz

## Soru ve değişiklik

Mevcut q_current FK'sine göre konum farkı/göreli quaternion girdisi,
aynı boyut ve model kapasitesinde ham pose'dan daha iyi öğreniliyor mu?
ADR-017 ve config.json eğitimden önce kaydedildi. 16 hücrede aynı
5000 full-batch AdamW güncellemesi, seed ve eşli kökler kullanıldı.
ADR-016 decoder ortak. Eşikler veya eski kod/sonuçlar değiştirilmedi.

## Tekrar üretim

Başlangıç HEAD: `701d54d`, branch `codex/c1-06r`. Kullanıcı STATUS ve
TRACEABILITY üst değişiklikleri ve ilgisiz PDF/DOCX/AUDIT dosyaları
korunur. Aynı locked Pixi + `.venv/c106r`, PyTorch2.10.0+cu128,
RTX5060 Laptop 8GB, 24GB RAM, Windows. Float32 model, float64 decoder;
TF32/AMP kapalı, deterministik, CPU thread1 ve CUBLAS workspace4096:8.

Seed2026100901; örnek sayısı512/2048; local/mixed; raw/relative;
absolute/residual; 16×5000=80.000 update. Tam batch kullanıldığından
2048 hücrede adım başına örnek maruziyeti512 hücrenin dört katıdır.
Model13-256-256-256-6SiLU, iki head'de son katman sıfır; AdamWlr1e-3,
wd0,01; cosine eta_min1e-6. Son adım seçimi, early stopping yok.
Normalizasyon bütün16800 train girdisinden; test/validation fit yok.

Robot/TCP/veri/kilit ve eski kod kimlikleri training-freeze.json ile
korunur. Yeni config, kaynaklar ve normalization hashleri registration.json
ve bütün checkpoint contract'larında bulunur. Son checkpoint ve raw
tahminler data/generated/C1-06R/diagnostic3 altında LOCAL_ONLY;
her hücre JSON'unda tam yol/SHA ve train pair_id contract'ı vardır.

```powershell
python scripts/c106r_command.py 035-relative-pose-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r/test_diagnostic3.py -q --junitxml=experiments/C1-06R/diagnostic3/tests.xml
python scripts/c106r_command.py 036-relative-pose-matrix -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_diagnostic3
python scripts/c106r_command.py 037-relative-pose-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_diagnostic3.py
```

UTC9 Ekim21:42:58.560968–21:48:00.020908; Türkiye'de10 Ekim
00:42:58–00:48:00. Süre301,46s; hücre içi ölçümlerin toplamı293,52s,
kalan süre hazırlık/fit/girdi işlemlerini içerir. Audit UTC21:48:21–21:48:33.
Tam argv/env/exit ve stdout/stderr SHA `../commands/035-*`–`037-*` altında.
Dosyalar immutable deney kimliğiyle yazılır; yeniden çalıştırma için
eski sonucu ezmeyen ayrı revizyon gerekir.

## Test ve ham kanıt

- 035: dört yeni temsil/normalizasyon/sızıntı/çerçeve testi PASS.
- 036: 16 koşu tamamlandı; 16 checkpoint yeniden yüklemesinde bütün
  validation tahminleri EXACT_MATCH; 57.600 satır. Her koşu A/B0/3600;
  main0/3000. Eğitim sayıları ve history her hücre JSON'unda.
- 037: 16 hücre tamlığı, 80.000 update, kayıt/ham SHA, eşli pair_id,
  tam payda ve dört512-raw kontrolün önceki tanıyla tensor eşliği PASS.
- Frozen122 dosya korundu. Önceki474 test yeniden çalıştırılmadı;
  uzun üç-seed kampanya NOT_RUN, final NOT_CREATED.

## Sonuç ve yorum

[RESULTS](RESULTS.md) sekiz eşli karşılaştırmanın tamamını verir. Göreli
temsil local sürekli hatayı azaltıyor; 2048-local-residual local medyanı
47,82mm/8,05°→19,57mm/4,98°. Ancak train A0/2048, validation A0/3600;
wide medyanı geçersiz satırlar nedeniyle sonlu değil. Tek-seed local
iyileşmesi genel başarı, H2-R veya ürün kabulü değildir. Bütün başarısız
hücreler ve ara kayıtlar korunur. Kök neden tek bileşene indirgenmedi.

## Sonraki adım

Aynı temsil/veride optimizasyon-kayıp ile kapasite etkisini ayrı tasarımla
ayırmak; geniş train pose hassasiyeti oluşmadan yeni uzun kampanya vermemek.
Bu takip deneyi NOT_RUN. Eşikler/ham sonuçlar ve eski H2 REJECTED korunur;
C1-07/G1 ve yeni bağımsız final açık kalır.

# C1-02 komut ve aşama kapıları

Depo kökünden PowerShell. Aşama 1'de gerçekten çalıştırılanlar:

```powershell
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
python scripts/check_c102_stage1.py --write-manifest
python scripts/check_c102_stage1.py --self-test
python scripts/check_c102_stage1.py --check
```

`--write-manifest` yalnız Aşama 1 dondurması içindir. Sonraki koşularda `--check`; upstream dosya değişmişse yeni hash ile sessizce manifest yenileme yapılmaz. Bu script çift üretmez ve T-C07 kabulü vermez.

## Aşama 2 · açık kullanıcı onayından sonra

1. Onay tarihi/bağlamını RUN_REPORT'a kaydet; `git status`, `git rev-parse HEAD`, `python scripts/check_c102_stage1.py --check`. Her hash driftini beklenen/bulunan/yol/etki ile raporlayıp dur.
2. Stage 2 CLI ve test kodu henüz mevcut değildir. Önce F0-04 root loader, schema validator, split/soy audit ve küçük sentetik unit/mutation testlerini uygula. Gerçek CLI komutları burada kod yazıldıktan sonra tarihli revizyonla kaydedilir; var olmayan komut koşulmuş sayılmaz.
3. 90 wide satırlık teacher pilotunu frozen config sınırında çalıştır ve hashli pilot log/summary üret. Technical failure, resource cap veya açıklanamayan seçim yanlılığı varsa tam üretim kapısını kapat. Test hedeflerini öğretmen başarısına göre değiştirme.
4. Pilot geçerse sürümlü 24.000 planlanan çiftin gerçek satır sayılarını, NPZ shardlarını, yalnız train'den normalizasyonu, manifest ve iki temiz üretim canonical content SHA'larını kaydet.
5. T-C07 pozitif/negatif/mutation, F0-04/F0-05/C1-01 arayüz regresyonları ve benchmark overlap auditini çalıştır. JUnit, gerçek komut, exit code, ortam lock ve ham kanıt yollarını kaydet.

Üretim yolu `data/generated/C1-02/v1` Git dışındadır. `experiments/C1-02/` yalnız küçük config/schema/manifest/audit/log/rapor kanıtlarını taşır. C1-01 raw sonuçlarına müdahale edilmez. Aşama 2 komutları **NOT_RUN / NOT_IMPLEMENTED**.

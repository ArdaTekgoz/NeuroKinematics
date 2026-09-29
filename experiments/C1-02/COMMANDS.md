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

Üretim yolu `data/generated/C1-02/v1` Git dışındadır. `experiments/C1-02/` yalnız küçük config/schema/manifest/audit/log/rapor kanıtlarını taşır. C1-01 raw sonuçlarına müdahale edilmez. Yukarıdaki liste Aşama 1'deki plandır; gerçekleşen Aşama 2 komutları aşağıdadır.

## 28 Eylül 2026 Aşama 2 gerçek komut kaydı

Kullanıcı açık onayı [approval](stage2-approval.json) içinde. Stage 1 `python scripts/check_c102_stage1.py --check` çıkış 0 / 43 hash PASS. İlk geliştirme pilotları (`pilot-20260928`, `pilot-20260928-v2`) 90 satırda 67 geçerli / 23 başarısız öğretmen sonucu verdi. Fakat thread ve 4 GiB tavanının ölçülmediği görüldüğü için ilk tam üretim kesildi; kısmi ham dosya `data/generated/C1-02/attempt-before-resource-gate` altında korundu, kabul kanıtı değildir. İlk kaynak probu Windows handle tipi nedeniyle hata verdi; kısmi pilot `pilot-v3-memory-probe-failed` altında korundu.

Kaynak probu düzeltildikten sonra kullanılan komutlar:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage pilot
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage environment
pixi run --locked python -m pytest tests/c1_02/test_tc07_mutations.py -q --junitxml=experiments/C1-02/mutation-junit.xml
pixi run --locked python -m pytest tests/f0_04/test_unit.py tests/f0_05/test_query_io.py tests/c1_01/test_main_runner_regressions.py -q --junitxml=experiments/C1-02/interface-regression-junit.xml
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage full
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage full -Output data/generated/C1-02/v1-repro
```

Pilot 90/90, 360 çağrı, 67 etiket / 23 eksik etiket, peak RSS 80.715.776 byte; dört thread ayarı 1; çıkış 0. Bu paragraf yazıldığında mutation 7/7 ve seçili arayüz regresyonları 42/42 geçti; tam üretimler henüz sürüyordu. Sonuçlar aşağıdaki tarihli kapanış kaydındadır. `pixi run --locked pytest` Windows başlatıcısı yol kodlaması nedeniyle exit 1 verdi; `python -m pytest` ile aynı testler exit 0 koşuldu.

## 29 Eylül 2026 · Aşama 2 kapanış komutları ve sonuçları

İki tam üretim de tamamlandı: `v1` ve `v1-repro` ayrı temiz çıktı dizinlerinde 24.000'er satır ve 34'er NPZ shard. Aşağıdaki son doğrulamalar gerçekten çalıştırıldı:

```powershell
python scripts/check_c102_stage1.py --check
pixi run --locked python -m pytest tests/c1_02/test_tc07_acceptance.py -q --junitxml=experiments/C1-02/tc07-junit.xml
pixi run --locked python -m pytest tests/c1_02/test_tc07_mutations.py tests/c1_02/test_tc07_dataset_mutations.py -q --junitxml=experiments/C1-02/mutation-junit.xml
pixi run --locked python -m pytest tests/f0_04/test_unit.py tests/f0_05/test_query_io.py tests/c1_01/test_main_runner_regressions.py -q --junitxml=experiments/C1-02/interface-regression-junit.xml
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_c102_stage2.ps1 -Stage verify
pixi run --locked python scripts/compare_c102_reproduction.py
pixi run --locked python scripts/collect_c102_evidence.py
```

Stage 1: 43/43 hash PASS. T-C07 kabul 7/7, mutation 12/12, seçili F0-04/F0-05/C1-01 arayüz regresyonu 42/42 PASS. `verify` exit 0; 24.000 satır, 34 shard, train-only normalizasyon ve leakage/ancestry/benchmark denetimleri PASS. Reproduction exit 0; 34/34 shard file/content hash eş, canonical veri SHA-256 iki koşuda `2db4667b982934408cb9204eb4f8a598337305fccdaa00b73beff016a87dd7c2`. Candidate semantik hash, süre alanı hariç, eş; ham JSONL süre nedeniyle farklıdır. `collect` küçük kanıtları topladı ve SHA256SUMS üretti.

İlk `verify` çalışması NumPy'nin uint8 dtype gösterimi (`|u1`) ile şemadaki eşdeğer yazım (`<u1`) arasındaki metin karşılaştırması yüzünden başarısızdı; doğrulayıcı dtype eşdeğerliğini NumPy ile karşılaştıracak şekilde düzeltildi, şema değişmedi. Sonraki `verify` train istatistiği toplama sırası değiştiği için yaklaşık 1e-15 fark yakaladı; doğrulayıcı üretimin root/mode sırasını izler hâle getirildi, eşik gevşetilmedi. Son kontrol PASS. Ayrıntı ve açık sınırlar [kabul raporunda](RUN-20260929-T-C07-acceptance.md).

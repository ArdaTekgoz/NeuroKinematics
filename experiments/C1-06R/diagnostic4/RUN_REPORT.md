# Deney veya uygulama kaydı

Kimlik: RUN-20261010-C106R-DIAGNOSTIC4

Durum: COMPLETE_DIAGNOSIS / AUDIT PASS / VALIDATION_TARGET_NOT_MET

Görev ve gereksinim: C1-06R; REQ-C03–05; optimizer/ölçek/kapasite ayrımı

Tarih ve sorumlu: 10 Ekim 2026; kullanıcı onayı sonrası AI kısa tanı/analiz

## Soru ve değişiklik

Göreli residual modelde aynı2048 local train örneğinde optimizer paketi,
global Q kayıp ölçeği veya ağ genişliğinin ayrı değişimi hedef hassasiyeti
sağlıyor mu? ADR-018/config eğitimden önce kaydedildi. Dört taze koşu;
ortak seed, girdi/normalizasyon, çıktı/decoder, örnekler ve A/B eşikleri.
Yeni kod ayrı modülde; tarihsel kod/çıktılar korunur. Geniş modelin başlangıç
fonksiyonu aynı, parametre boyutları farklı. L-BFGS'te weight decay ve cosine
yok; optimizer paketi bir bütün olarak değişti. Eşit hesap iddiası yoktur.

## Tekrar üretim

Başlangıç HEAD `8be8cd7`, branch `codex/c1-06r`. Kullanıcı STATUS/TRACEABILITY
değişiklikleri ve ilgisiz PDF/DOCX/AUDIT çıktıları korunur. Yeni deney kodu
çalıştırılırken çalışma ağacı kirli; registration.json exact kaynak SHA'larını
kaydeder. Yazılım hedefi v1.0.0; görev belgesi r5. Windows11, AMD Ryzen7 250,
RTX5060 Laptop8151MiB, kullanıcı beyanı24GB RAM. Locked Pixi + ayrı
`.venv/c106r`, Python3.12.14, Torch2.10.0+cu128. Float32 model, float64 decoder;
TF32/AMP kapalı, deterministik, CPU thread1, CUBLAS workspace4096:8.

Seed2026100901; n2048; main/local eşli kök seçimi tanı3 ile aynı. Train-only
16800 girdiden hesaplanmış tanı3 normalizasyonu yeniden fit edilmeden kullanıldı.
Dar model136710, geniş535558 parametre. Son katman sıfır; residual başlangıç
q_current. AdamW üç koşuda5000 update, lr.001/wd.01/cosine eta1e-6. L-BFGS
5000 internal iterasyon ve5457 closure; config'te bütün ayarlar kayıtlı.
Toplam20457 gradyan hesabı; maliyetler aynı değildir. Terminal checkpoint
kullanılır; early validation seçimi yoktur. Tam3600 validation değerlendirilir.

Robot/TCP/veri/kilit kaynakları training-freeze.json içindeki122 hash ile;
yeni config/kod/test/audit/ADR ve normalizasyon registration.json ile korunur.
Ağırlıklar ve satır bazlı tahminler data/generated/C1-06R/diagnostic4 altında
LOCAL_ONLY; hücre JSON'ları SHA/yol ve checkpoint'ler pair_id listesi içerir.
ADR-018 Git satır sonu LF olarak sabitlendi; kayıtlı hash checkout'ta korunur.

```powershell
python scripts/c106r_command.py 038-optimization-controls-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r/test_diagnostic4.py tests/c1_06r/test_diagnostic3.py -q --junitxml=experiments/C1-06R/diagnostic4/tests.xml
python scripts/c106r_command.py 039-optimization-controls-matrix -- pixi run --locked .venv/c106r/Scripts/python.exe -m neurokinematics.neural.c106r_diagnostic4
python scripts/c106r_command.py 040-optimization-controls-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_diagnostic4.py
python scripts/c106r_command.py 041-optimization-error-decomposition -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/analyze_c106r_diagnostic4_errors.py
python scripts/c106r_command.py 042-optimization-error-decomposition-fixed -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/analyze_c106r_diagnostic4_errors.py
```

Kampanya UTC10 Ekim06:10:26.388604–06:12:30.849110; Türkiye09:10:26–09:12:30.
Süre124,46s; eğitim dışı ölçüm/reload/hazırlık dahil. Komut038–042 exact
argv/env/zaman/exit ve stdout/stderr SHA kayıtları ../commands altında.
Eski kimliği yeniden çalıştırma engellenir; tekrar için ayrı sürüm gerekir.

## Test ve ham kanıt

- 038: üç yeni başlangıç/kapasite/kayıp gradyanı testi + dört temsil testi;
  toplam7 PASS. Kayıp ölçeği gradyanı çarpıyor, sıfır optimumunu koruyor.
- 039: dört koşu COMPLETE; tüm checkpoint'lerde weights_only güvenli reload,
  dört×(2048+3600)=22592 train/validation satırı EXACT_MATCH.
- 040: kaynak/ham hash, aynı örnek listesi, tam payda/Profil A/B, optimizer
  bütçeleri ve tanı3 referans son tensor/metrik eşliği PASS; frozen122 korundu.
- 041: sonuç sonrası analiz betiğinde robot.limits tuple'ını NumPy dizisi
  gibi indeksleme TypeError verdi. Eğitim veya metrik sonuçları etkilenmedi.
  Başarısız kaynak error-analysis-attempt-001.py ve komut logu korundu.
- 042: açık np.asarray dönüşümü sonrası betimsel analiz COMPLETE; her
  hata bölümü tam paydasına eşit. Yeni eğitim/model seçimi yapılmadı.
- Önceki474 regresyon testi bu tur yeniden çalıştırılmadı; üç-seed yeni
  uzun eğitim NOT_RUN, yeni final NOT_CREATED, eski final raw NOT_READ.

## Sonuç ve yorum

[RESULTS](RESULTS.md) bütün koşulları ve hata ayrımını verir. Validation
her modelde A/B0/3600, main0/3000. Geniş model train Q'yu %46,15 azaltır;
train A3/2048, B0. Local validation medyanı19,65mm/5,55°; referans19,57mm/4,98°.
Bu kapasite artışı eğitim hatasını azaltıyor fakat bu bütçede genellemeyi
iyileştirmiyor. Global kayıp ölçeği tek başına yeterli değil. L-BFGS paketi
bu koşullarda daha kötü. Kök neden tek bileşene indirgenmedi.

Geniş ağ train1886 satırında iki pose eşiği birlikte aşılıyor;19 satır
limit dışı. Limit sorunu çoğunluk hatasını açıklamıyor. Joint4/5 RMSE yüksek;
TCP duyarlılığındaki nedensel katkıları henüz ölçülmedi. Tek seed sonuçları
tüm mimari/optimizer seçeneklerine genellenemez; küçük median değişimleri
ürün başarısı sayılamaz. Profil A/B ve %95 ürün kapısı aynen korunur.

## Sonraki adım

[NEXT_DIAGNOSTIC](NEXT_DIAGNOSTIC.md): aynı train satırlarında eklem/TCP
duyarlılığı, Q/pose hata ilişkisi ve öğrenme gradyanlarını ölç; ardından
tek gerekçeli müdahaleyi ön kaydet. Bu takip NOT_RUN. Yeni uzun paket
hazır değil. Tarihsel C1-06 H2 REJECTED, C1-07/G1 açıklığı korunur.

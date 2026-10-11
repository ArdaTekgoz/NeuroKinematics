# Deney veya uygulama kaydı

Kimlik: RUN-20261009-C106R-ROUND1-AUDIT

Durum: TRAINING COMPLETE / AUDIT PASS / VALIDATION TARGET NOT_MET

Görev ve gereksinim: C1-06R R4, REQ-C03–05; tam kampanya ve seçim denetimi

Tarih ve sorumlu: 9 Ekim 2026; uzun eğitim kullanıcı, denetim ve analiz AI

## Soru ve değişiklik

Kullanıcının tamamladığı 12 koşu hedeflenen başarıyı sağladı mı?
Yeni read-only audit kodu envanter, SHA, contract, epoch/permutation,
tam bütçe, seçim ve checkpoint yeniden çıkarımını denetler. Best/last
train/validation ve q_current tanısı üretildi. Ayrı kayıtlı 24 train
gradyan probu kayıp yön çatışmasını ölçtü. Eğitim değiştirilmedi.

## Tekrar üretim

Başlangıç HEAD: `074f978` (`codex/c1-06r`). Kullanıcıya ait STATUS,
TRACEABILITY ve PDF/DOCX/AUDIT dosyası değişiklikleri korunur. Windows,
RTX 5060 Laptop 8151 MiB, 24 GB RAM, Pixi kilidi + C1-06R exact cu128
overlay; TF32/AMP kapalı, deterministik, CPU thread 1,
CUBLAS_WORKSPACE_CONFIG=:4096:8. Runtime/girdi kimlikleri eski
training-freeze.json ile yeniden doğrulandı. Config hash:
`a97a7f4e8845d2c48d9cd392d5f3f4435672946c508d97ce588a746a91dcc9dd`.
Robot/TCP/data/source
kimlikleri training-freeze ve run contract kayıtlarında sabittir.
Seedler 2026100901/02/03; 2000 epoch/120.000 update/koşu.
Ağırlık ve ham kayıt kimlikleri audit.json manifest alanında bulunur.

```powershell
python scripts/c106r_command.py 020-round1-output-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_round1.py
python scripts/c106r_command.py 021-round1-audit-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r/test_round1_audit.py -q --junitxml=experiments/C1-06R/round1-analysis/audit-tests.xml
python scripts/c106r_command.py 022-round1-output-audit -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/audit_c106r_round1.py
python scripts/c106r_command.py 023-round1-audit-tests -- pixi run --locked .venv/c106r/Scripts/python.exe -m pytest tests/c1_06r/test_round1_audit.py -q --junitxml=experiments/C1-06R/round1-analysis/audit-tests-r2.xml
python scripts/c106r_command.py 024-round1-gradient-probe -- pixi run --locked .venv/c106r/Scripts/python.exe scripts/probe_c106r_round1.py
```

Komutlar immutable isimlidir; yeniden çalıştırmada yeni kayıt kimliği gerekir.
Audit/probe mevcut sonuç dosyasını ezmez. Ham stdout/stderr ve UTC başlangıç/
bitiş zamanları `../commands/020-*`–`024-*` altındadır. Başarılı audit UTC
20:07:47.774851–20:09:05.909797; gradyan tanısı 20:09:55.022492–20:10:04.308804.

## Test ve ham kanıt

- 020: FAIL, TorchVersion weights-only okuma kusuru; kayıt korundu.
- 021: İlk 12 bozuk/eksik envanter ve epoch negatif testi PASS.
- 022: PASS; 85 dosya, 12 koşu, 24.000 epoch, 1.440.000 update,
  aynı seed başlangıç/permutation eşliği; best/last seçim ve kayıt eşliği.
  24 checkpoint × 3600 validation = 86.400 yeniden çıkarım satırı birebir.
  Ayrıca 24 × 16.800 train = 403.200 pose satırı yeniden ölçüldü.
- 023: 13 test PASS; gerçek TorchVersion metadata yükleme kusuru ve
  yalnız bu sınıfa scoped safe-list okuma kontrolü eklendi.
- 024: 24 sabit train gradyan probu PASS; hiçbir optimizer güncellemesi yok.
- Profil A/B: her modelde 0/3600; kayıtlı 960 validation değerlendirmesinde
  A/B sıfır. Main 0/3000, ≥%95 kapısı FAIL. H2-R final NOT_EVALUATED.
- Önceki 471 regresyon bu analiz değişikliği için tekrar çalıştırılmadı.
  İkinci uzun tur NOT_RUN, bağımsız yeni final NOT_CREATED.

## Sonuç ve yorum

Bütünlük ile model başarısı ayrıdır. Eğitim teknik olarak tamamlandı;
%95 hedefini sağlamadı. Gerçek kampanya 10.250,003 saniye: önceki 6–11
saatlik tahmin fazla geldi. Süreyi artırmak bu sonuçlarla tek başına
gerekçelenmez. [Sonuç raporu](RESULTS.md) seed tablolarını, train/validation
ayrışmasını, q_current baseline ve gradyan bulgusunu içerir. Bu tanı
genelleme açığının tek nedenini kanıtlamaz; eski H2 sonucu değişmez.

## Sonraki adım

[Kontrollü tanı planı](NEXT_DIAGNOSTIC.md): mevcut eklem durumuna düzeltme
öğrenme, local/mixed ayrımı, yeterli hassasiyet sonrası kontrollü FK ekleme.
Sonraki training sürümünde Torch sürümü plain string olacak ve production
contract ile gerçek kesinti/devam doğrulanacak. Mevcut launcher'ı yeniden
çalıştırmak gerekmez. Ham çıktı/weights korunur; C1-07/G1 açık kalır.

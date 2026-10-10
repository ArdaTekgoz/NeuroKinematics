# G1 kararı — Core araştırma kapanışı

11 Ekim 2026 · Belge r1 · **PASS / ACCEPTED — RESEARCH CLOSURE**.
C1-01–C1-07 COMPLETE. Direct IK **NO_GO**; ürün hedefi **NOT_MET**;
H2 **REJECTED**. Hybrid **NOT_STARTED**, H1 **NOT_MEASURED**.

Karar, Core raporunun 6. bölümündeki araştırma kapanışı ölçütlerine dayanır:
kritik doğruluk ve tekrar üretim testleri, çalışan zorunlu baselinelar ve
açık hipotez kararı. %95 ürün hedefi sağlanmadı; hiçbir başarı eşiği
düşürülmedi. Olumsuz araştırma kapanışı raporda baştan izin verilen sonuçtur.
ADR-025, C1-06R araştırmasının durdurulmasını ve aday rollerini sabitler.

## Kabul matrisi

| Gereksinim / test | Kanıt | Karar |
|---|---|---|
| G0, robot/TCP, referans/bağımsız FK/Jacobian | [G0](../F0-06/G0_DECISION.md), frozen 4 robot dosyası | Tarihsel PASS; model sözleşmesi değişmedi |
| REQ-C01 / T-C00 | [Baseline kabulü](../C1-01/RUN-20260928-T-C00-acceptance.md) | Beş baseline, 600000 ölçüm kaydı; tarihsel PASS |
| REQ-C03 / T-C07 | [C1-02](../C1-02/acceptance.json) | Veri, split, train-only normalizasyon PASS |
| REQ-C02 / T-C01–02 | [C1-03](../C1-03/stage2/acceptance.json) | FK, gradyan ve mutant kapıları PASS |
| REQ-C03 / T-C03 | [C1-04](../C1-04/stage2/acceptance.json) | Küçük kontrollü öğrenme PASS; direct IK NO_GO |
| REQ-C03–04 / T-C04 | [C1-05](../C1-05/stage2/acceptance.json) | Ana karşılaştırmalar tamamlandı; direct IK NO_GO |
| REQ-C04–05 / T-C05 | [C1-06](../C1-06/stage2/final-001/acceptance.json) | PASS; 21 model × 12000 sorgu, H2 REJECTED |
| C1-06R kapsam kararı | [Durdurma](preparation/C1-06R-closure.json), [negatif rapor](../../docs/research/C1-06_NEGATIVE_RESULTS.md) | CLOSED_WITH_UNMET_PRODUCT_TARGET; yeni final NOT_CREATED |
| REQ-C06 / T-C06 | [Temiz tekrar](closure/clean/complete.json), [witness](closure/clean/witness-result.json) | PASS; 6 checkpoint × 48 sorgu = 288 birebir çıktı ve FK metrikleri |
| Regresyon / yanlış başarı engeli | [JUnit ve loglar](closure/clean/) | 11 handoff + 79 FK + 31 physics + 6 decoder = 127 PASS; 0 skip/fail/error |
| Hash ve izlenebilirlik | [Integrity](closure/integrity.json) | 122 frozen kaynak, 455 C1-06R teslim kaydı, 16 hazırlık dosyası, D3–D9 ve hazırlık kayıtları değişmez |

Tarihsel testlerin tümü bu kapanışta yeniden çalıştırılmadı; seçili kritik
regresyonlar çalıştırıldı. 48 sorguluk T-C06 örneği performans tahmini veya
yeni bağımsız final değildir. main/boundary/singularity × local/wide ilk
8'er validation sorgusu seçildi; başarıya göre örnek seçimi yoktur.

## Gerçek tekrar üretim

Kaynak commit `c7798ef177c831398dc54dec6cbe7f0d0dd2e061`.
Temiz clone başlangıcında `.pixi`/`.venv` yoktu; kilitli kurulum yapıldı,
ortam kopyalanmadı. Windows 11 x64, Python 3.12.14, NumPy 2.5.3,
Torch 2.10.0+cu128, Pixi 0.81.0. CPU witness: thread 1, batch 48.
Decoder testleri CPU/CUDA, RTX 5060 Laptop GPU. Ryzen 7 250 / 24 GB RAM.
Harici artifact root'tan yalnız altı hashli checkpoint okundu.

```powershell
python scripts/c107_command.py 006-clean-reproduction -- python scripts/reproduce_c107.py --checkout temp/c107-clean-20261011 --output experiments/C1-07/closure/clean --artifact-root .
```

Başlangıç 10 Ekim 23:43:04 UTC, bitiş 23:46:42 UTC; İstanbul'da
11 Ekim 02:43–02:46. Toplam yaklaşık 218,55 saniye. Paket cache'i
kullanılabilir; yeniden eğitim NOT_RUN. Son Git status boş.
NaN mutantı testinin NumPy determinant uyarısı logda korunur; test PASS.

## Devir ve açık sınırlar

[Nihai model kartı](MODEL_CARD.md) ve [Hybrid manifesti](HYBRID_HANDOFF.json)
ana FK_TANH'ın üç seed'ini, ikincil local-only RAW'ın üç seed'ini sabitler.
CENTERED ve LOCAL_Z devredilen ana aday değildir. Hiçbir model doğrudan
robot komutu üretmek için kabul edilmedi. Çıktılar yalnız araştırma adaylarıdır.

Yeni C1-06R finali oluşturulmadı; H2-R yeni finalde NOT_EVALUATED.
Eski final raw yeniden analiz edilmedi. Arşivleme opak bayt kopyasıdır.
Collision, fiziksel robot güvenliği, H1, ONNX ve yeni Linux T-C06 NOT_RUN.
Sayısal çözücünün başarısızlığı erişilemezlik kanıtı değildir.

Git kod ve küçük kanıtları taşır. Büyük veriler/checkpointler yerel arşivde;
[arşiv makbuzu](closure/archive-receipt.json) ve [geri yükleme kontrolü](closure/restore-receipt.json)
ile doğrulanır. Uzak arşiv ve ikinci aygıt NOT_CONFIRMED.
Yazılım hedefi v1.0.0; bu belge release/tag oluşturmaz. Foundations ortamının
paket sürümü 0.1.0 ve pixi.lock değiştirilmedi. Belge revizyonu ayrı izlenir.

Sonraki görev H2-01 / REQ-H01 / T-H01. Başlatılmadı; projeye dönüş tarifi
[CORE_RESUME](../../docs/records/CORE_RESUME.md) içindedir. Bu karar teknik
araştırma kabulüdür; hakemli yayın veya harici sertifikasyon değildir.

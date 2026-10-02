# C1-04 ön kayıtlı kabul ve test matrisi

2 Ekim 2026 · Aşama 1 · test sonucu değil. Normatif ayarlar [config.json](config.json) içindedir. Eşikler sonuç sonrası düşürülemez.

| Kapı | Gereksinim ve örnek | Önceden sabitlenmiş karar | Planlanan ham kanıt | Aşama 1 |
|---|---|---|---|---|
| Giriş audit'i | G0, C1-02, C1-03 kabulü; 34 yerel shard SHA; robot/TCP/Torch FK ve normalization | Hash/sayı driftinde dur; girdisiz eğitme | `input-hashes.json`, checker stdout/exit | yalnız hash audit |
| Loader pozitif | 24.000 satır; 12.000 kök, iki mode, değişmez split ve grup; 15.204 train, 3.249 validation etiketli | Alan/sıra/dtype/count, yalnız izinli özellik, NaN sentinel maskesi birebir | pytest JUnit, loader summary JSON | NOT_RUN |
| Loader negatif | Eksik etiket, limit dışı q, NaN/Inf, yanlış feature sırası, yanlış robot/TCP ve split/grup sızıntısı | Her mutasyon beklenen hata sınıfıyla fail-fast; model/optimizer başlatılmaz | mutasyon JUnit ve vaka JSON | NOT_RUN |
| T-C03 gerçek öğrenme | Her model, ilk 64 `main/train/local` etiketli satır; ilk 32 `main/validation/local` izleme; seed `2026100201`, 200 epoch, batch 64 | Her modelde son train loss ≤0,5×epoch0 ve son medyan mutlak q hatası ≤0,75×epoch0; tüm değerler finite; validation kayıtlı | epoch JSONL, başlangıç/son metrik, checkpoint kimliği | NOT_RUN |
| T-C03 yanlış eşleme | Aynı 64 train kökünde q etiketini `pair_id` sıralı bir konum döndür, input/ID sabit | Bağımsız Pinocchio FK Profile B (≤0,001 m ve ≤0,5°) etiket-hedef audit'i eğitime geçmeden reddeder; kontaminasyonlu veride PASS yok | vaka/ilk uyuşmazlık ve JUnit | NOT_RUN |
| E-C01 adalet | Pose-only ve conditioned, aynı etiketli train/validation `pair_id` listesi, üç eşli seed | Altı tam koşu, aynı örnek sırası, loss, optimizer, epoch/step ve etkin batch; iki model aynı seed'de ancak ikisi de patience 20'ye ulaşınca birlikte durur | seed/epoch JSONL, run summary, config/hash | NOT_RUN |
| Checkpoint | En düşük finite genel etiketli validation loss; tam eşitte ilk epoch | Best/last SHA+byte, scaler/robot/veri/config metadata, yükleme ve sabit validation çıkarımı birebir | local weights ve küçük hash manifesti | NOT_RUN |
| Validation geometri | Her model/seed, tüm 3.600 satır; q etiketi metriği yalnız 3.249 etiketli | Ham nonfinite/limit dışı sayısı; geçerli q için Pinocchio FK m/°; mod/family/etiket kırılımı, N/median/P95/P99; geçersizler başarı sayılmaz | per-row validation sonuçları, summary JSON; referans spot cross-check | NOT_RUN |
| Regresyon | C1-02 T-C07 arayüz ve C1-03 FK, seçili Foundations | Eski eşikler sabit; ilgili testler PASS ya da açık FAIL | JUnit ve exact komut/exit/log | NOT_RUN |
| Temiz tekrar | Yeni checkout ve yeni ortamda en az checkpoint yükleme, çıkarım, FK ve hash audit | Başarılıysa yalnız inference/kanıt tekrarını gösterir; ikinci tam eğitim yoksa eğitim determinism iddiası yok | komut/log/JUnit/hash | NOT_RUN |

T-C03 **PASS** yalnız iki modelin küçük öğrenme ölçütleri ve yanlış eşleme/fail-fast kontrolleri geçince. E-C01 üç seed adil karşılaştırması ve FK kırılımları tamamlanmadan **C1-04 COMPLETE** yoktur. Nihai test ve 10.000 sorguluk benchmark C1-06'ya mühürlüdür. Salt train ezberi genelleme kanıtı değildir.

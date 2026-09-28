# ADR-008 · Tarihsel FK kayıtlarının platform kapsamı

Durum: kabul edildi, 26 Eylül 2026; C1-01 / REQ-C01.

## Kanıt ve karar

Windows Foundations kayıtları değişmezdir. F0-06 tarihsel schema mutation testi Windows'ta PASS; Linux'ta eski orientation_error_deg sayısal alanı 0.05991817363546874 yerine 0.05991817364763415 yeniden hesaplanır. Fark 1.216541e-11 derece; rad farkı yaklaşık 2.123265e-13. Konum, geometri A/B ve limit kararları aynı. NumPy 2.5.3 / Pinocchio 4.1.0 iki ortamda aynı; kesin alt runtime işlemi henüz izole edilmedi. Olası kayan nokta platform farkı, doğrulanmış alt işlem nedeni olarak sunulmaz.

Orijinal test, validator eşikleri, kabul toleransları ve eski kayıtlar değiştirilmez. Orijinal Linux koşu 208 PASS / 1 FAIL olarak kalır. Linux taşınabilir regresyon kümesinde yalnız bu tarihsel Windows residual eşleşmesi testi ayrı raporlanır; yerine C1-01 ek testi aynı aday/pose için native runtime residual alanlarını bellekte üretir, bütün kategorik kararların eski kayıtla aynı olduğunu zorunlu tutar, mevcut strict validator ile tekrar doğrular ve eksik solver_status ile 1e-6 derece residual tahrifini reddettiğini kontrol eder. Bu geçici kayıt ölçüm veya tarihsel yeniden üretim kanıtı değildir.

## Sonuç

Yerel native ek test 1 PASS (0.23s). Linux ek test NOT_RUN. Eski Windows byte/residual yeniden üretimi Linux PASS olarak ilan edilmez. C1-01 Linux çıktıları aynı pinned Linux runtime'da doğrulanır. Linux kritik regresyon ancak mevcut diğer 208 test ve ek mutasyon testi PASS olursa kapsamı açıkça belirtilerek kabul edilir. Tam T-C00 ve küçük uçtan uca smoke henüz NOT_RUN.

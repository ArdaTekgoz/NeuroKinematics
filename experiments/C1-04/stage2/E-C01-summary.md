# E-C01 eşli validation özeti

Test ve 10.000 sorguluk benchmark açılmadı. FK metrikleri yalnız limit içi ham q için Pinocchio referansıyla ölçüldü; limit ihlalleri ayrıca sayıldı.

| Seed | Model | En iyi epoch | Val q loss | N geçerli / 3.600 | Limit dışı | Profil A başarı / 3.600 | Konum medyan/P95 (m) | Yönelim medyan/P95 (°) | q MAE medyan (rad) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026100201 | pose_only | 171 | 0.41571 | 3438 | 162 | 0 | 0.582 / 1.242 | 123.6 / 174.5 | 1.397 |
| 2026100201 | conditioned | 199 | 0.170779 | 3331 | 269 | 0 | 0.2061 / 0.9408 | 74.65 / 169.7 | 0.4176 |
| 2026100202 | pose_only | 171 | 0.414716 | 3480 | 120 | 0 | 0.5845 / 1.252 | 121.3 / 173.7 | 1.395 |
| 2026100202 | conditioned | 200 | 0.17443 | 3416 | 184 | 0 | 0.207 / 0.9435 | 78.15 / 169.3 | 0.4359 |
| 2026100203 | pose_only | 165 | 0.414451 | 3460 | 140 | 0 | 0.5706 / 1.255 | 124.1 / 173.8 | 1.397 |
| 2026100203 | conditioned | 199 | 0.173756 | 3412 | 188 | 0 | 0.2103 / 0.9697 | 76.52 / 169.3 | 0.4471 |

Eksi fark conditioned lehinedir; yalnız iki ham q da limit içindeyken eşli FK farkı hesaplanır. Bu seçim yanlılığı yaratabileceği için bütün 3.600 satırın geçerlilik sayıları tabloda korunur.

| Seed | İki model de geçerli N | Conditioned − pose konum farkı medyan (m) | Conditioned − pose yönelim farkı medyan (°) | Conditioned konum kazanım oranı |
|---:|---:|---:|---:|---:|
| 2026100201 | 3238 | -0.2756 | -35.72 | 0.859 |
| 2026100202 | 3325 | -0.2725 | -32.11 | 0.849 |
| 2026100203 | 3310 | -0.2539 | -35.72 | 0.847 |

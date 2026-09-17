# ADR 003 Poz ve eklem temsili

Durum: API tasarım kararı; model gösterimi deneysel

API metre/radyan/saniye, base içinde TCP pozu ve wxyz quaternion kullanır. Eklem sırası robot manifestinden gelir. Sınırlı eklemler limit tabanlı normalize edilir. Quaternion ve sürekli 6D model girdisi aynı API üzerinde ablation olur; q ve delta çıktı da ayrı deneydir.

Gerekçe: Kaynak rapordaki temsiller birbiriyle karışmamalıdır. Quaternion işaret simetrisi ve sınırlı eklemlerde tur bilgisinin korunması gerekir. Tek bir neural gösterim bütün projeye yayılmaz.

Değişirse veri/model şema sürümü artırılır ve bütün adaptör testleri tekrar çalışır. Eski checkpoint sessizce yeni temsille yüklenmez.

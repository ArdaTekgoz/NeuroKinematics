# C1-01 · Ana test adımı için ajan devri

26 Eylül 2026. Durum: IN_PROGRESS / STAGE_2_IMPLEMENTING. Kullanıcı talimatıyla ana smoke/benchmark kodu öncesinde duruldu. Bu dosya yeni ana test kodu değildir.

## Doğrulanmış durum

- Foundations değişmez; Stage1 main/origin main commit33d18e54d6c9480627985d58e6a82557835f7fc0. Stage2 henüz commit/push yok.
- Aktif audited image sha256:8fa7616bb86884e8ce27c0ff86527e1592e519d2a44ab0bd7eb473fbc032b5ff; Ubuntu24.04x86_64/Jazzy, CPU0/1, frozen Pixi lock,691dpkgpaket.
- Linux kritik regresyon211PASS/1deselected; ADR-008 eskiWindowsFKresidualtesti ayrıdır, özgünLinuxFAILkanıtı korunur.
- Eski anaT-C00 INCOMPLETE:4yöntem120000'er kayıt offlinePASS, global115INCOMPLETE. linux-full/ ve attempts/20260926-155030-006/linux-full/ kanıtları,run-binding.json,verified-summary'ler. Bunlar tamkabul veya yeniUDPv4kıyaslama değildir.
- IPC okuyucu processqueueisolation ve tekstdoutreader düzeltildi; warmrestart aynıilk20queryileısıtılır. Eskiwarmupfixpilot120PASSkanıtımevcut.
- Gerçekbozukready:FastDDSRTPS_TRANSPORT_SHMsegmentcreateerrorstdout; linux-ready-diagnostic/ kanıtı. QueuefixDDSresourceproblemını tekbaşınagidermedi.
- Aynıimage kontrollü UDPv4stress500PASS:226SUCCESS/274TIMEOUT;256restart,258launch,5160warmup,worker_start_errornull. /dev/shmönceboş;sonrasıyalnız2lttnggiriş,fastrtpssegment0. rawSHAeb2dd454d02eb46fb45b88507bc0c89aefe051f55359cbccb5842740af3998af; lockSHA12471f7a5619f3efee1dfd737e57e66615b0930209e50b96fe10ec880e43b5c1. Yerelraw/lock/warmupbağkontrolleriPASS. linux-global-restart-stress-udp/stress-gate.json.

## Sonraki yetkilendirilmesi gereken iş

Kullanıcı ajanı değiştirecek ve ana smoke/benchmark kodu için açık onay verecek. Bu onaydan önce o kod yazılmayacak, ana koşu komutu verilmez.

Onaydan sonra görev tanımı/orijinalpastedrequest/config/AGENTS/ADR'leri oku. UDPv4(RMW_IMPLEMENTATION=rmw_fastrtps_cpp,FASTDDS_BUILTIN_TRANSPORTS=UDPv4) ortak runtime politikası ve denetim/kilit/kanıt bağlarını resmileştir; aynı host/image/code/thread/affinity koşullarını beşyöntem için uygula. Frozen baseline solverparam/query/threshold değiştirme. IPCölçümlerini/DDSruntime'ını ayrıştır; DDSyalnıznodebaşlangıçaltyapısı,solverrequestnewlinepipeile iletilir.

Mevcut ana runner/testkodunu gözden geçir; smoke+küçük10/50mspilotyenile, sonra tamT-C00(12000×2×5×5=600000) uygun ortakkoşulda yürüt. Eski480000kanıtı yeniruntimeglobalile sessizcebirleştirme. Hardtimeout/restarttoplamduvarsaatinin maliyetini pilotkanıtından tahmin et;20saatetkinemektavanını indirme, gerçekemekNOT_MEASURED açıkolmalı. Değişmeyenkomutlarıgereksizyereyenidenbuildetme; gerekiyorsanedeninikaydet.

İşlem yalnız kullanıcıPowerShellkomutları+çıktılarıyla; uzaktanDockererişimivarsayılmaz. Yeni hamkanıt ayrıdizin, eskileriüstüneyazma. Linuxaynıruntime'daofflineverify/summarize,all/success/status/subset/start/deadlinegrupları,geometri/deadlineayrı,timingIPC+validationdahil,iterations/seedNOT_AVAILABLEdürüst. Sonuçlar tamkabulöncesiMEASURED_UNVERIFIED.

Kapanış: RUN_REPORT/STATUS/TRACE/task/roadmap tutarlı,gereksinim→değişiklik→test→kanıt; tekrarüretim/kilitvebilinenkısıtlar20saatemekkaydı dahil. YalnızC1-01kapsamıcommit/push,remoteeşleşmesinidoğrula; büyükrawdosyalarıGit'eeklemekyerinehashlidışmanifestkararıgerekir. Kullanıcının ikiWorddosyası kapsam dışı. Corefazını/C1-01'i erkenCOMPLETEişaretleme,sonrakigörevleri başlatma.

## 27 Eylül 2026 · Onay sonrası devam

Kullanıcının “Onaylıyorum” ve devam mesajıyla bu belgedeki onay bekleme adımı tamamlandı. Ana akış hazırlanıyor; yeni worker süre aktarımı düzeltmesi için türetilmiş image build gerekli. Güncel durum ve sonraki komutlar: [çalışma kaydı](RUN-20260927-main-preparation.md), [COMMANDS.md](COMMANDS.md). Önceki maddeler 26 Eylül tarihli devir kaydıdır; yeniden onay istenmeyecek.

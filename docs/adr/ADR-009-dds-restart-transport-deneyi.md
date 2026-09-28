# ADR-009 · C1-01 worker yeniden başlatmalarında DDS transport deneyi

Durum: ÖNERİ / DOĞRULAMA BEKLENİYOR, 26 Eylül 2026.

## Kanıt

Aynı IPCfiximage 8fa7616b... üzerinde global stress121/500 ve tanı121/250INCOMPLETE. Tanı bozukready satırını yakaladı: Fast DDS RTPS_TRANSPORT_SHM Error Failed to create segment fastrtps_071a42563d642699: No such file or directory. /dev/shm251giriş. Shared memory segment oluşturma hatasının stdoutJSONprotokolünü bozduğu doğrudan kanıtlandı. 251giriş ve sıkhardkill kaynakbirikimi hipotezini destekler; işletim sistemi althatanın tamnedeni ayrıcaizoleedilmedi. ÖncekiqueuefixbuDDSprobleminigidermez.

## Kontrollü öneri

Installed Fast DDS2.14.6 resmi envvars dokümanı FASTDDS_BUILTIN_TRANSPORTS=UDPv4 ayarının yalnızUDPv4oluşturduğunu, DEFAULT'ın UDPv4+SHM oluşturduğunu belirtir (use_builtin_transports true durumunda): https://fast-dds.docs.eprosima.com/en/v2.14.6/fastdds/env_vars/env_vars.html . Aynıimage/source/config/query/budgetile mevcuthedefli500kayıtstress çağrısına RMW_IMPLEMENTATION=rmw_fastrtps_cpp ve FASTDDS_BUILTIN_TRANSPORTS=UDPv4 eklenecek. Stressoutput ayrı linux-global-restart-stress-udp/. DDSenv ve/dev/shmönce/sonragirişleri kanıta yazılacak. JSONhataları atlanmayacak,solverdeadlinetolerans değişmeyecek. AnaIPCnewlinepipe kalır.

## Kabul ve etki

500kayıt, enaz100restart,herworker20warmup,fataltransporterror0,independentFK/order/hashPASS gerekir. TestNOT_RUN; başarıiddiasıyok. Başarılıolursa kalıcıtransport/runtimekilidivebeşyöntemdeaynıkoşul gerektiği yeniagentana smoke/benchmarktasarımında ele alınacak. Eski4yöntem480000verifiedkanıt tarihselkoşusuyla korunur; yeniDDSkoşuluylaresidual/timingsonuçlarısessizcebirleştirilmez. Kullanıcıtalimatı: ana smoke/benchmarkkodundanönceonayveajandeğişimi.

## 26 Eylül hedefli doğrulama sonucu
UDPv4stress500/500PASS,256restart,258launch/5160warmup,readyerrornull. Raw/lockSHAkontrolleriPASS;fastrtpsSHMafter0(yalnız2lttnggiriş). Hedeflipolitikadoğrulandı; kalıcıortakruntimekilidi/anaT-C00kabulübekliyor. KullanıcıtalimatıylanextANA testkodundanönceDUR/onay/ajandeğişimi. MAIN_TEST_HANDOFF.md kanıtyollarını içerir.

## 27 Eylül 2026 · Ortak runtime kararı

Durum: **KABUL EDİLDİ / ANA KOŞU DOĞRULAMASI BEKLENİYOR**. Kullanıcı ana test hazırlığına onay verdi. Beş yöntem aynı `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` ve `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` ortamıyla yeniden koşacak. DDS, ROS node başlangıç altyapısıdır; ölçülen istek/yanıt yerel newline JSON pipe kullanır. 500 kayıt stres PASS sonucu uzun koşu garantisi değildir.

`runtime-lock.json`, image/native worker ve Python kaynaklarını, testleri, paket kilidini, CPU 0/1 ve thread ayarlarını bağlar. Prepare → smoke → pilot → full → verify zinciri hashlerle doğrulanır. Kalan bütçeyi iletme kusuru dondurulmuş sözleşmeye uygun düzeltildi; tolerans veya solver ayarı değişmedi. Bunun için mevcut dependency closure korunarak yalnız yerel native worker yeniden derlenecek. Önceki lock ve 480.000 kayıt yeni koşuya taşınmaz. Yeni Linux build ve ana testler NOT_RUN; [çalışma kaydı](../../experiments/C1-01/RUN-20260927-main-preparation.md).

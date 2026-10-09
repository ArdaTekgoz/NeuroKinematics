# C1-06R — Kullanıcı eğitim yönergesi

9 Ekim 2026 · r2 · **READY_FOR_USER_TRAINING / ANA EĞİTİM NOT_RUN**

Bu bilgisayarda ortam kuruldu ve doğrulandı. RTX 5060 Laptop 8 GB, Windows,
24 GB RAM; `.venv/c106r`, PyTorch 2.10.0+cu128. Docker gerekmiyor.
Başlangıç depo HEAD'i `77dae3844d60bbc242855d3874785facf0f1f786`;
yeni uygulamanın asıl byte kimliği [training-freeze.json](training-freeze.json)
ile sabittir. Git teslim kimliği ayrıca `delivery.json` kaydında tutulur.

## Çalıştırılacak tek blok

PowerShell terminaline yapıştır:

```powershell
Set-Location -LiteralPath 'C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main'
powershell.exe -NoProfile -File .\scripts\Start-C106RTraining.ps1
```

Windows ile gelen PowerShell yeterlidir; PowerShell 7 (`pwsh`) kurulumu
gerekmez. Yukarıdaki komut Windows PowerShell ile `-CheckOnly` eklenerek
doğrulandı: [ham kayıt](commands/019-windows-powershell-check-only/stdout.log).

Komut paket/kurulum, kaynak/veri hashleri, GPU ve en az 5 GiB boş disk
kontrolü yapar; ardından dört varyant × üç seed'i sırayla çalıştırır.
Ana eğitim AI tarafından başlatılmadı. Hazır oluş denetimi aynı launcher'ın
`-CheckOnly` seçeneğiyle gerçekten çalıştırıldı; bu seçenek eğitim yapmaz.

## Ne çalışacak, ne kadar sürecek?

- Q, FK, Q_TANH, FK_TANH; seedler 2026100901/02/03. Her model taze ve eşli
  başlangıç ağırlıklarıyla kurulur; eski checkpointten ana eğitime başlanmaz.
- Her koşu 2000 epoch, batch256, 120.000 optimizer güncellemesi; 12 koşuda
  1.440.000 güncelleme. Aynı 15.204 etiketli train satırı kullanılır.
- Her 25 epoch'ta 3.600 validation sorgusunun tamamı bağımsız FK ile ölçülür.
  Checkpoint önce gerçek Profil A başarısı, sonra geçersiz sayısı ve sürekli
  poz hatasıyla seçilir. Test kümesi açılmaz; en iyi seed seçilip diğerleri atılmaz.
- Ölçülen iki-epoch smoke'ta epoch eğitim süreleri Q: 0,53–0,67 s,
  FK: 1,26–1,51 s, Q_TANH: 0,32–0,49 s, FK_TANH: 1,14–1,23 s.
  Buradan hesaplanan toplam eğitim çekirdeği yaklaşık 6 saat; checkpoint,
  validation, ısınma ve başka uygulamalara bağlı farklarla **yaklaşık 6–11 saat**
  ayır. Bu tam kampanya ölçümü veya süre garantisi değildir.
- Smoke'ta PyTorch tensor peak allocation yaklaşık 71 MiB; bu değer CUDA
  context/diğer uygulamaların tüm VRAM'i değildir. Küçük tanıda süreç RAM
  tepe gözlemi yaklaşık 1,2 GiB. 8 GB kart bu smoke'u çalıştırdı. Uzun koşunun
  kesintisiz kaynak profili henüz ölçülmedi. Başlangıçta yaklaşık 299 GiB boş
  disk vardı; launcher güncel boş alanı yeniden denetler.
- Bilgisayar adaptöre bağlı ve uyanık kalsın. Konsolda epoch ve validation
  kayıtları görünür. Tamamlandığında `Completed validation campaign` yazılır.

## Kesinti, hata ve çıktı

Aynı komutu tekrar çalıştırmak tamamlanmış koşuları doğrulayıp atlar,
diğerlerini son tamamlanmış epoch checkpointinden sürdürür. Dört arm için
gerçek CUDA kesinti/devam testinde son ağırlık hashleri kesintisiz koşuyla
birebir eşleşti. Kısmi epoch yeniden çalışır. Kaynak/config değişmiş veya
checkpoint bozuksa otomatik devam reddedilir. İlk checkpoint oluşmadan
kesinti/kurulum hatası varsa hata metnini paylaş; dosya silme veya YAML
düzenleme işi sana bırakılmıyor.

Sonuç kökü:

`C:\Users\Arda TEKGÖZ\Desktop\NeuroKinematics-main\data\generated\C1-06R\round1`

Her `seed-.../ARM/` klasöründe `checkpoint.json`, iki `last-*.pt` slotu,
`epochs.jsonl`; tamamlanınca `best.pt`, `best-validation.json`, `complete.json`
oluşur. Bütün kampanya bitince `campaign-complete.json` ve `SHA256SUMS` yazılır.
Ham weights/çıktılar Git dışındadır. Aynı bilgisayarda olduğumuz için dosya
yüklemen gerekmiyor: bitince **“C1-06R eğitimi bitti”**, hata olursa terminalin
son hata metnini yazman yeterli. Kayıtları ben okuyup doğrulayacağım.

Bu ilk tur validation araştırmasıdır. %95 başarı veya H2-R desteği henüz
ölçülmedi. Sonraki deney/final kararı çıktı denetiminden sonra verilir.

## Ortamı yeniden kurmak gerekirse

Mevcut makinede tekrar kurulum gerekmez. Aynı Windows/Python tabanında
kurulumda gerçekten kullanılan komutlar aşağıdadır; başka işletim sistemi
doğrulanmış sayılmaz. Yeni checkout'ta büyük C1-02 shardları ayrıca hashli
yerel veri kökünden sağlanmalıdır; bunlar normal Git clone ile gelmez.

```powershell
pixi install --locked
pixi run --locked python -m venv --system-site-packages .venv/c106r
pixi run --locked .venv/c106r/Scripts/python.exe -m pip install --ignore-installed --require-hashes --no-deps -r experiments/C1-06R/requirements-win-cu128.lock
pixi run --locked .venv/c106r/Scripts/python.exe -m pip check
```

Yeni/taşınmış ortamda eğitimden önce GPU/FK kabulü tekrar doğrulanır. Mevcut
girdi/ortamı değiştirme ihtiyacı çıkarsa bunu AI hazırlar.

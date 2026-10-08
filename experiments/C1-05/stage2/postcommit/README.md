# Commit sonrası temiz kanıt denetimi

İlk denetim, `8b218b3` checkout'unda `clean/commands/venv/command.json`
bulunamadığı için FAIL verdi. `.gitignore` içindeki genel `venv/` kuralı,
kurulum komutunun üç küçük kanıt dosyasını da dışlamıştı. Gerçek ortam değil,
yalnız bu kanıt klasörü için dar bir istisna eklendi. Frozen manifest ve
log baytları değiştirilmedi. İlk exit 1 kaydı `bundle-audit/` altında korunur.

Düzeltme sonrası `attempt-002/` gerçek argv/exit/stdout/stderr kaydını,
`clean-bundle-audit.json` nihai sonucu ve `closure.json` denetlenen commit'i
taşır. Bu kayıtlar ilk snapshot'tan sonra oluştuğu için onun manifestine
eklenmez; ayrı `SHA256SUMS` ile korunur. Final test ve benchmark açılmadı.

# C1-03 yürütme protokolü r2 — Stage1 r1 üzerine ek

29 Eylül 2026. Sayısal eşikler, örnekler, dtype, loss ve test kapsamı değişmez.
Stage1 dosyaları değiştirilmez; config hash'i f7064ba8…664d6d olarak kalır.

1. ADR-012 runtime düzeltmesi: aynı NumPy 2.5.3'ün Windows wheel'i ve setuptools
   82.0.1 hashli supplemental lock ile **venv içine** `--ignore-installed` kurulur.
   Foundations ortamı korunur. İlk pip check FAIL ve OMP abort saklanır. Paket
   kapısı geçmeden tam test yapılmaz; aynı shell'de erken analytic başlatılmış
   005 denemesi kabul sayılmaz. Komut kaydedicinin yalnız konsol yazdırma encoding
   hatası düzeltildi; subprocess exit0 kurulum kaydı silinmedi.
2. R03 bağımsızlık testinin import harness düzeltmesi: değişmez Foundations
   `kinematics/__init__.py` eager `IndependentFK` export eder. İlk r1 blanket
   custom_fk import yasağı Torch koduna ulaşmadan bu tarihsel paket girişinde
   başarısız oldu (012: 18 PASS, 1 FAIL). Foundations dosyasını değiştirmek yerine
   alt süreç önce bu export'u yükler, **gerçek IndependentFK.forward_kinematics'i
   exception trap ile kapatır**, sonra sonraki custom_fk ve tüm Pinocchio
   importlarını yasaklar; Torch constructor/forward/backward bu sınırda çalışır.
   Böylece bir NumPy FK çağrısını gizleyen test kabul edilemez. Torch kaynak
   AST/source audit'i ayrıca detach/NumPy/reference/no_grad hesabını dışlar.

Bu düzenlemeler başarısız sonucu geçmiş gibi etiketlemez. Yeni analitik/smoke/full
ve ikinci fresh kurulum yalnız r2 altında sıfırdan çalıştırılır. Pinocchio oracle,
F0-02/F0-03 matematik kodu ve kanıtları değişmez; kabul eşikleri gevşetilmez.

3. Artifact audit: aynı sürümlü typing_extensions ilk kurulumda Pixi ortamından
   miras kalmıştı. Exact Stage1 wheel ayrı overlay içine kuruldu (023); 11/11
   URL/SHA/version/path denetimi 024 ile PASS. Temiz kurulum ilk lock için de
   --ignore-installed kullanır; sürüm eşliği artifact eşliği diye sunulmaz.
   Düzeltilmiş exact overlay üstünde yerel smoke/full yeniden çalıştırılır.

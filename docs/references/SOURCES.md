# Kaynak kayıtları ve kullanım sınırları

Erişim tarihi 17 Eylül 2026. Birincil yayın ve resmi yazılım kaynakları kullanılmıştır. Makale kaydı/özeti okunması, tam metin metodoloji replikasyonu anlamına gelmez. Bu paket sistematik literatür taraması veya patent yenilik incelemesi değildir.

## K01 Pinocchio resmi depo

[Birincil kaynak](https://github.com/stack-of-tasks/pinocchio)

**Erişim düzeyi:** Resmi yazılım belgesi/depo.

URDF, FK/Jacobian, geometri bağlantısı; BSD-2-Clause kaydı. Kullanılacak sürüm ayrıca kilitlenecek.

## K02 PyTorch Kinematics resmi depo

[Birincil kaynak](https://github.com/UM-ARM-Lab/pytorch_kinematics)

**Erişim düzeyi:** Resmi yazılım belgesi/depo.

Batched differentiable FK/Jacobian ve URDF; MIT kaydı. API bütün modelleri otomatik destekler varsayımı yok.

## K03 MoveIt Kinematics Configuration

[Birincil kaynak](https://moveit.picknik.ai/main/doc/examples/kinematics_configuration/kinematics_configuration_tutorial.html)

**Erişim düzeyi:** Resmi belge.

KDL ve ortak konfigürasyon. İncelenen doküman rolling/main; uygulama ortamı release ile sabitlenecek.

## K04 MoveIt TRAC-IK

[Birincil kaynak](https://moveit.picknik.ai/main/doc/how_to_guides/trac_ik/trac_ik_tutorial.html)

**Erişim düzeyi:** Resmi belge.

Newton tabanlı bileşen ve SQP; seed/tolerance/timeout nüansları.

## K05 MoveIt pick_ik

[Birincil kaynak](https://moveit.picknik.ai/main/doc/how_to_guides/pick_ik/pick_ik_tutorial.html)

**Erişim düzeyi:** Resmi belge.

Global evolutionary ve local gradient arama; local/global farklı benchmark varyantları.

## K06 Kevin M Lynch ve Frank C Park Modern Robotics Bölüm 6

[Birincil kaynak](https://modernrobotics.northwestern.edu/chapters/chapter6/)

**Erişim düzeyi:** Yazar/üniversite ders kaynağı.

Analitik ve sayısal IK; ileri öğrenme ve implementasyon için temel okuma.

## K07 Raphael Bensadoun Shir Gur Nitsan Blau Lior Wolf Neural Inverse Kinematic

[Birincil kaynak](https://proceedings.mlr.press/v162/bensadoun22a.html)

**Erişim düzeyi:** ICML 2022 yayın kaydı ve özet.

PMLR 162:1787–1797. Eski rapordaki CVPR kaydı ve one-to-many eksikliği düzeltilmiştir.

## K08 Gian Maria Marconi Raffaello Camoriano Lorenzo Rosasco Carlo Ciliberto CRiSP

[Birincil kaynak](https://arxiv.org/abs/2102.12942v3)

**Erişim düzeyi:** 2021 ön baskı kaydı ve özet.

Tam başlık Structured Prediction for CRiSP Inverse Kinematics Learning with Misspecified Robot Models. FK ile yapılandırılmış öğrenme; birebir yeniden uygulama yapılmadı.

## K09 Jan-Gerrit Habekost Connor Gäde Philipp Allgeuer Stefan Wermter CycleIK

[Birincil kaynak](https://arxiv.org/abs/2404.08825v2)

**Erişim düzeyi:** IROS 2024 bilgisi içeren yayın kaydı ve özet.

Tam başlık Inverse Kinematics for Neuro-Robotic Grasping with Humanoid Embodied Agents. Platform uyarlaması; yayın sayıları bizim robotla doğrudan kıyaslanmaz.

## K10 Zeyu Zhang ve Ziyuan Jiao IKDiffuser

[Birincil kaynak](https://arxiv.org/abs/2506.13087v4)

**Erişim düzeyi:** 14 Ocak 2026 v4 ön baskı kaydı ve özet.

Tam güncel başlık IKDiffuser: a Diffusion-based Generative Inverse Kinematics Solver for Kinematic Trees. İlk 2025başlığı değişmiş; under review kaydı korunur.

## K11 Jiahao Yang Shenhao Yan Fan Feng Chengsi Yao Ge Wang Zhixin Mai Yiming Zhao Yatong Han MimicIK

[Birincil kaynak](https://arxiv.org/abs/2606.15148v2)

**Erişim düzeyi:** 16 Haziran 2026 v2 ön baskı kaydı ve özet.

Tam başlık MimicIK: Real-Time Generative Inverse Kinematics from Teleoperation with FK Consistency. Current state/delta/FK yakınlığı; farklı robot/dataset/tolerans.

## K12 Shihui Fang Min Chen Yaran Chen Jia Wang Jinghua Wu Zhihua Zhang Eng Gee Lim AdaKineNet

[Birincil kaynak](https://scholar.xjtlu.edu.cn/en/publications/adakinenet-adaptive-kinematic-neural-network-for-inverse-kinemati/)

**Erişim düzeyi:** Kurumsal hakemli yayın kaydı ve özet.

Robotics and Autonomous Systems 202:105494 Ağustos 2026; DOI 10.1016/j.robot.2026.105494. Tam metin üzerinden performans denetimi yapılmadı.

## K13 Yi Zhou Connelly Barnes Jingwan Lu Jimei Yang Hao Li Rotation Representations

[Birincil kaynak](https://arxiv.org/abs/1812.07035)

**Erişim düzeyi:** Makale kaydı ve özet; CVPR 2019.

Tam başlık On the Continuity of Rotation Representations in Neural Networks. 5D/6D ve düşük boyutlu gösterim tartışması.

## K14 Qt for Python Licenses Used

[Birincil kaynak](https://doc.qt.io/qtforpython-6/licenses.html)

**Erişim düzeyi:** Resmi lisans belgesi.

Kullanılan Qt modülleri ve üçüncü taraf bildirimleri ayrıca incelenir; ticari uygunluk sonucu verilmedi.

## K15 PyTorch torch.onnx

[Birincil kaynak](https://docs.pytorch.org/docs/2.14/onnx.html)

**Erişim düzeyi:** Resmi API belgesi.

İnceleme gününde stable bağlantısı 2.14'e yönlendi. torch.export/dynamo akışı; kurulu sürüm iddiası yok.

## K16 Coal resmi depo

[Birincil kaynak](https://github.com/coal-library/coal)

**Erişim düzeyi:** Resmi yazılım deposu.

Pinocchio ile geometri adaptörü adayı. Kesin paket sürümü ve geometri destek profili H2-04'te belirlenecek.

## K17 Chelsea Finn Pieter Abbeel Sergey Levine MAML

[Birincil kaynak](https://proceedings.mlr.press/v70/finn17a.html)

**Erişim düzeyi:** ICML 2017 yayın kaydı ve özet.

PMLR 70:1126–1135; görev dağılımı ve az gradient adımıyla adaptasyon.

## K18 PyVista resmi belgeleri

[Birincil kaynak](https://docs.pyvista.org/)

**Erişim düzeyi:** Resmi API ve kullanıcı belgesi.

PyVista/VTK görselleştirme adayı. Kullanıcı makinesinde kurulum veya performans doğrulanmadı.

## Eski kaynakların durumu

Eski [9] Bensadoun kaydı K07 ile düzeltilmiştir. Eski [10] F Li ve diğerleri Physics-informed neural networks for inverse kinematics of highly redundant manipulators; [11] Alatty/Yang Ambiguity resolution in learning-based inverse kinematics; [12] Smith/Doe State-conditioned learning for ambiguity resolution in robot kinematics; [13] Agarwal Real-time edge inference for neural robotic control için exact-title aramasında eşleşen güvenilir birincil kayıt doğrulanamadı. Yeni raporda bilimsel dayanak değiller. Yoklukları kesin olarak kanıtlanmış sayılmaz.

Eski [1]–[8] tarihsel kaynak listesi yeni rapora otomatik kopyalanmadı. Yeni metnin ihtiyaç duyduğu teknik dayanaklar yukarıdaki seçili kayıtlarla oluşturuldu. Eski RoboDK/Isaac Sim karşılaştırmasındaki kullanım kolaylığı puanları ve destek sınırlamaları yeniden doğrulanmış sonuç olarak taşınmadı.

## Öncelikli okuma sırası

Foundations için K06,K01,K03–K05; Core için K07,K02,K13,K08,K09,K11; Hybrid için K10,K15,K16; Studio için K14,K18 ve araştırma zamanı K17. AdaKineNet K12 özgünlük tartışmasında yöntem yakınlığı için incelenecek; tam metin edinilirse ayrı okuma notu açılacak.

# C1-01 Stage 1 kaynak ve bağımlılık kaydı

24 Eylül 2026. Bu kayıt kurulu paket kanıtı değildir. Kaynak commitleri `git ls-remote` ile okundu; Stage 2 temiz ortamında checkout ve paket sürümleri tekrar doğrulanacaktır.

| Kimlik | Resmî kaynak / pin | Sürüm | Lisans | Bakım ve arayüz |
|---|---|---|---|---|
| DLS/default | [NeuroKinematics](https://github.com/ArdaTekgoz/NeuroKinematics), `e7d211f42496f803688e2a510daca97e102092dc` | 0.1.0 | `NOT_DECLARED`: depo kökünde lisans dosyası yok; F0 robot snapshot Apache-2.0 lisansı proje geneline taşınmaz | Mevcut Python DLS, değiştirilmeden yerel worker ile çağrılacak. |
| KDL/default | [MoveIt 2](https://github.com/moveit/moveit2), tag `2.15.2`, `093360fef2ea8269c607821b551ffb2d4bc5c9a8` | 2.15.2 | BSD-3-Clause | Bakımı süren MoveIt 2 `kdl_kinematics_plugin/KDLKinematicsPlugin`; C++ plugin. |
| TRAC-IK/speed | [TRACLabs](https://github.com/traclabs/trac_ik), tag `2.0.2`, `d8d54abda51f10554501854a744f21fd5b63daf6` | 2.0.2 | BSD | Jazzy sürümü mevcut; `trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin`; C++ plugin. |
| pick_ik/local | [PickNik](https://github.com/PickNikRobotics/pick_ik), tag `1.1.3`, `c476d3aab7879c54f6167ecdf7a1a7fa769de4de` | 1.1.3 | BSD-3-Clause | PickNik tarafından deprecated; yalnız temel bakım. `mode=local`, C++ plugin. |
| pick_ik/global | Aynı pin | 1.1.3 | BSD-3-Clause | Ayrı kimlik/config; `mode=global`, evrimsel arama ve 2 thread tavanı. |

ROS 2 dağıtımı **Jazzy**, hedef sistem **Ubuntu 24.04 LTS x86_64**. MoveIt kaynak/paket sürümü **2.15.2**, TRAC-IK **2.0.2**, pick_ik **1.1.3** donduruldu. `ros-jazzy-*` Debian paket revizyonları, transitif bağımlılıklar, compiler/C++ runtime ve container/VM kimliği bu Windows makinede kurulu Linux ortamı olmadığı için `NOT_AVAILABLE`; Stage 2 smoke öncesi exact sürüm + SHA-256 lock kaydı zorunlu. Bu çözülmeden tam benchmark yapılmaz. Üçüncü taraf kaynak/binary repo içine kopyalanmaz. Lisans ve notice dosyaları harici kurulumda saklanır; dağıtım gerekirse ayrıca gözden geçirilir.

Resmî kanıt: [ROS Jazzy Ubuntu desteği](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), [MoveIt sürüm/branch politikası](https://github.com/moveit/moveit2), [KDL parametreleri](https://github.com/moveit/moveit2/blob/093360fef2ea8269c607821b551ffb2d4bc5c9a8/moveit_kinematics/kdl_kinematics_plugin/src/kdl_kinematics_parameters.yaml), [TRAC-IK parametreleri](https://github.com/traclabs/trac_ik/blob/d8d54abda51f10554501854a744f21fd5b63daf6/trac_ik_kinematics_plugin/src/trac_ik_kinematics_parameters.yaml), [ROS Index TRAC-IK](https://index.ros.org/p/trac_ik/), [pick_ik kullanım/yerel-küresel mod](https://github.com/PickNikRobotics/pick_ik/blob/c476d3aab7879c54f6167ecdf7a1a7fa769de4de/doc/USAGE.md), [pick_ik deprecation](https://index.ros.org/r/pick_ik/), [pick_ik parametreleri](https://github.com/PickNikRobotics/pick_ik/blob/c476d3aab7879c54f6167ecdf7a1a7fa769de4de/src/pick_ik_parameters.yaml).

Kurulum yöntemi: harici çalışma alanında exact commit checkout + Jazzy paketleri exact Debian revizyonlarıyla; `colcon` Release derlemesi. Repo `pixi.lock` yalnız NeuroKinematics Python katmanını kilitler, ROS paketlerini kilitlemez. Temiz ortamda `ros2 pkg prefix`, `dpkg-query`, `colcon` ve `git rev-parse` çıktıları ayrıca kaydedilecektir. Windows native destek, dört MoveIt plugininin ortak çalışması için doğrulanmadı.

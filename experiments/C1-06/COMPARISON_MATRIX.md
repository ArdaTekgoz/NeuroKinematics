# C1-06 önceden sabit karşılaştırma matrisi

Birincil: E-C05/FK_TANH − E-C03/Q, üç seed eşit ağırlıklı. Profil A, aynı query_id.

| Deney/model | Seed | Best epoch | Bayt | SHA-256 |
|---|---:|---:|---:|---|
| E-C03/Q | 2026100201 | 199 | 1651971 | `bff9c3e01e4085553aa9ee34c3faff53b140f5ab82daf1c52c01abbafaaa76cf` |
| E-C03/FK | 2026100201 | 194 | 1651971 | `052f1f461abfb3e1f69ba4350ebcc745564a9d882cd9133c0e8b613eaa076e79` |
| E-C03/Q | 2026100202 | 200 | 1651971 | `25019bef133f04024f9b5a16835311188649f3e19bdc0d822aaa98648e841a3e` |
| E-C03/FK | 2026100202 | 193 | 1651971 | `b9270b7c11b316496dfa73280efdb25bb31af021928520bada5bb593cfcb5b76` |
| E-C03/Q | 2026100203 | 199 | 1651971 | `21e92f1a1dc6c5901246d876c1aac6bace4c100a05352a13c3e84a800cf39d25` |
| E-C03/FK | 2026100203 | 197 | 1651971 | `690ce01f3766a1e746f30cc63167cc2aef2171f92b606648713703722ace31ab` |
| E-C04/FK | 2026100201 | 6 | 1651971 | `0b719e4c3350e226d6289702613137cfe861022527ce97291560e6846fd8b6db` |
| E-C04/FK_LIMIT | 2026100201 | 6 | 1651971 | `4f675f00ee37bec019e2c0c0f67a00243b437d7eae66ca011a8cfc983e136a1d` |
| E-C04/FK | 2026100202 | 193 | 1651971 | `ad1853c4db74948037e3f3f1ba10b07369afe7b73fbbd7bf97fea1bf3a5186f0` |
| E-C04/FK_LIMIT | 2026100202 | 197 | 1651971 | `75400504bfc2227bd8d54b203e3f1a0cbd33e75dbabd750a8f5bf5ce8f900610` |
| E-C04/FK | 2026100203 | 6 | 1651971 | `3f966d9d8e1fd3c0f23b28189eaed7c8df35514b1e452718d6ec7b7ea41431ae` |
| E-C04/FK_LIMIT | 2026100203 | 6 | 1651971 | `e541695a23ac6e11b669b8f0661dae67e37446546bf7fd08cf957ebb658ebe10` |
| E-C05/FK | 2026100201 | 6 | 1651971 | `074074b9a3e74ac264cf275505820c28900731698f3869687e2ba6216063304a` |
| E-C05/FK_TANH | 2026100201 | 7 | 1651971 | `29d0f2c38ebdd422429fc327e10c4420f77193077e41e4b247a541b5bbf047d9` |
| E-C05/FK | 2026100202 | 193 | 1651971 | `bfd80d9eba0d37e2a4544367952f126d42cbb938f415c944d129d9a0c460f87b` |
| E-C05/FK_TANH | 2026100202 | 186 | 1651971 | `a45b72999ad24e421c4b581859666ae8d4e7bee58b1079c49fc8709e83b763ad` |
| E-C05/FK | 2026100203 | 6 | 1651971 | `c7b2d620dcb4d00f9260a5f275fda0bce7016d42a45126b6a6b4a47163312066` |
| E-C05/FK_TANH | 2026100203 | 8 | 1651971 | `486cbece1ff553aec421ece9101b177f079abbd283bda070830b2fbf52f8f069` |
| E-C01/conditioned | 2026100201 | 199 | 1651459 | `0d931d90fd7081ecf36868b21ccd830fc0a9daa0ff036fb96fff1d53ba8b6a5a` |
| E-C01/conditioned | 2026100202 | 200 | 1651459 | `27bf75601645b4a1e6adf228062a75a2409aca8f1ecad41c1aa2784b4506a00c` |
| E-C01/conditioned | 2026100203 | 199 | 1651459 | `2f05c5ee8c4e6251f8c081aaf6affe31cec8ee8c10d3b5cff9e7f1d34541babf` |

21 checkpointin tamamı 12.000 sorguyu aynı sırada işler; 252.000 farklı model/sorgu sonucu, beş süre geçişiyle 1.260.000 ölçüm satırı. Seedler veya süre tekrarları farklı sorgu sayılmaz. Checkpoint mutlak ve göreli yolları input-hashes/config içindedir.

İkincil keşifsel kıyaslar: E-C03 FK−Q; E-C04 FK_LIMIT−FK; E-C05 FK_TANH−FK. E-C01 conditioned−E-C03 Q yalnız tekrarlı kontrol; validation çıktıları önceki kabulde birebir eş olduğundan iki bağımsız kontrol örneklemi sayılmaz.

E-C03 her seed 200 epoch; E-C04 26/200/26; E-C05 27/200/28. FK_TANH best epochları 7/186/8. Aynı deneyin iki kolu aynı gerçekleşen bütçeyi paylaşır; deneyler arasında süre/epoch farklıdır. FK_TANH−Q bileşik kayıp/başlık etkisidir, tek bileşene nedensel atıf yapılmaz.

DLS/default, KDL/default, TRAC-IK/speed, pick_ik/local ve pick_ik/global: 12.000 aynı sorgu × 10/50 ms × 5 geçiş = yöntem başına 120.000 tarihsel satır. Ubuntu/ROS süreleri ayrı betimsel tablo; Windows neural süreleriyle üstünlük testi yok.

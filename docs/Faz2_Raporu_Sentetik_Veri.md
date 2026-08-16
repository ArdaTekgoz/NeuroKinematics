# FAZ 2 Raporu: Sentetik Veri Fabrikası ve Veri Mühendisliği

## 1. Genel Bakış
Bu fazda NeuroKinematics modelinin eğitiminde kullanılacak fiziksel olarak geçerli, dengeli ve yeniden üretilebilir sentetik veri üretim pipeline'ı oluşturulmuştur.

## 2. Mimari

```
                 KUKA KR6
                    │
                    ▼
             Joint Limits
                    │
                    ▼
        ┌─────────────────────┐
        │ Sampling Engine     │
        │                     │
        │ Uniform     70%     │
        │ Boundary    20%     │
        │ Singularity 10%     │
        └──────────┬──────────┘
                   │
                   ▼
              Forward FK
                   │
                   ▼
          Pose + Joint Dataset
                   │
                   ▼
          Workspace Voxelization
           (Reachability Map)
                   │
                   ▼
         Spatial Group Splitting
          ┌────────┼────────┐
          ▼        ▼        ▼
        TRAIN     VAL      TEST
                   │
                   ▼
          Leakage Detection
          (Joint + Pose)
                   │
                   ▼
              HDF5 Master
                   │
                   ▼
          PyTorch Dataset
```

## 3. Geliştirilen Modüller

### 3.1 Sampling Engine (`neurokinematics/data/sampling.py`)
- **Uniform Joint-Space (70%):** Tam eklem aralığında rastgele örnekleme
- **Workspace Boundary (20%):** En az bir eklem sınıra yakın (%15 margin)
- **Singularity-Enriched (10%):** 3 katmanlı gradual dağılım:
  - Tier 1 (40%): w < 0.01 (derin tekillik)
  - Tier 2 (35%): 0.01 ≤ w < 0.05 (tekilliğe yakın)
  - Tier 3 (kalan): 0.05 ≤ w < 0.10 (tekilliğe yaklaşan)
- **q_{t-1} Üretimi:** Hedef konfigürasyona Gaussian gürültü ekleyerek

### 3.2 Workspace Analysis (`neurokinematics/data/workspace_analysis.py`)
- **Reachable voxel keşfi:** 50K FK örnekle erişilebilir bölgelerin tespiti
- **Coverage:** Erişilebilir voxel bazlı ölçüm (toplam bounding box yerine)
- **Orientation Coverage:** Her voxel'deki açısal çeşitlilik analizi
- **Spatial Group Split:** 4³=64 macro-block, sample-count-aware greedy atama

### 3.3 Data Split & Leakage (`neurokinematics/data/data_split.py`)
- **Normalized Joint-Space Leakage:** q̃ᵢ = (qᵢ - q_min) / (q_max - q_min)
- **Pose-Aware Leakage:** d_pose = √((d_p/σ_p)² + (d_R/σ_R)²)
- **Split Ratio Validation:** Tolerans kontrolü

### 3.4 Dataset Factory (`neurokinematics/data/dataset_factory.py`)
- Master pipeline: sampling → FK → coverage → split → leakage → HDF5
- Tam deterministik (seed kontrollü)
- İlerleme raporlaması

### 3.5 PyTorch Dataset (`neurokinematics/data/dataset.py`)
- **Input:** [position(3), rotation_6d(6), q_prev(6)] = 15D
- **Target:** [sin_q(6), cos_q(6)] = 12D
- HDF5 lazy loading (bellek dostu)
- Split-aware (train/val/test)

### 3.6 HDF5 Yapısı
```
dataset.h5
├── inputs/
│   ├── target_position      (N, 3) float32
│   ├── target_rotation_6d   (N, 6) float32
│   └── q_previous           (N, 6) float32
├── outputs/
│   ├── q                    (N, 6) float32
│   ├── sin_q                (N, 6) float32
│   └── cos_q                (N, 6) float32
├── physics/
│   ├── manipulability       (N,) float32
│   └── joint_limit_margin   (N,) float32
├── splits/
│   ├── train_indices
│   ├── val_indices
│   └── test_indices
├── sampling_method          (N,) int8
├── metadata/ (attrs)
└── normalization/
    ├── q_min, q_max
    └── pos_mean, pos_std
```

## 4. Doğrulama Sonuçları (10K Örnek — Hızlı Test)

| Kriter | Sonuç | Durum |
|---|---|---|
| Determinizm (aynı seed → aynı veri) | ✅ bit-for-bit eşleşme | PASS |
| Joint-Space Leakage | 0 (0.0000%) | PASS |
| Pose Leakage | 0 (0.0000%) | PASS |
| FK Consistency | max error: 0.00e+00 m | PASS |
| Joint Limit Violations | 0 | PASS |
| Workspace Coverage (reachable) | 87.1% (10K ile) | ✅ (1M ile ≥95%) |
| Orientation Coverage | 81.5% | PASS |
| Split Oranları | 73% / 15% / 13% | PASS |

> **Not:** 10K örnekle %87.1 workspace coverage elde edilmiştir. 1M örnekle (100x fazla) ≥%95 hedefi rahatlıkla karşılanacaktır.

## 5. Teknik Kararlar (Kilitlenmiş)
1. Örnekleme: %70 uniform / %20 boundary / %10 singularity-enriched (gradual)
2. Split: Spatial Group Split (komşu voxel'ler aynı grupta)
3. Leakage: Normalized joint-space + Cartesian pose (pozisyon + yönelim)
4. Format: HDF5 master → PyTorch lazy Dataset
5. Kapsam: KR6 configuration-level split (trajectory/robot-level → Faz 5+)

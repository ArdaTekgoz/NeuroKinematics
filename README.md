# NeuroKinematics: Learning-Based Inverse Kinematics

**Learning-Based Inverse Kinematics for Industrial Robotic Manipulators**

## 1. Project Motivation
Inverse kinematics (IK) is a fundamental problem in robotics: determining joint configurations that achieve a desired end-effector pose. For industrial manipulators, IK is traditionally solved using analytical or numerical methods derived from the robot’s kinematic model.

However, classical IK approaches face several limitations:
* Complex derivations for high-DOF manipulators
* Multiple valid solutions (IK ambiguity)
* Sensitivity to modeling errors and singularities
* Difficulty integrating uncertainty or data-driven priors

**NeuroKinematics** explores an alternative perspective: *Can inverse kinematics be learned as a function approximation problem, while still respecting the robot’s geometric structure?* This project investigates machine learning–based IK, progressively integrating physical constraints into a neural formulation.

---

## 2. Experimental Roadmap: The "A-Series"
The project is structured as a stage-based experimental roadmap to avoid "black-box" results and ensure explainable modeling.

| Series | Focus |
| :--- | :--- |
| **A-1** | Pure function approximation (MLP sanity checks) |
| **A-2** | Dataset structure, normalization, stability |
| **A-3** | Pose representations and input encoding |
| **A-4** | Inverse kinematics with geometric consistency |
| **A-5** | (Planned) Physical FK, constraints, real robot alignment |

> **Current Status:** This repository documents progress in **A-4**.

---

## 3. A-4 Series: Geometry-Aware Learning
A-4 focuses on injecting **Forward Kinematics (FK)** structure into the learning process. Joint-space accuracy alone does not guarantee task-space correctness; therefore, a learned IK model must be evaluated by how well its joints reconstruct the desired end-effector pose.

### 4. Problem Formulation
* **Inputs (End-effector pose):** $X = [x, y, z, q_x, q_y, q_z, q_w]$ (Position in meters, orientation as unit quaternion)
* **Outputs (Joint angles):** $Y = [\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6]$ (Radians, KR6-class 6-DOF)
* **Objective:** $f_\theta : \mathbb{R}^7 \to \mathbb{R}^6$ such that $FK(f_\theta(X)) \approx \text{position}(X)$.

### 5. Dataset (Synthetic Baseline)
Used to isolate learning behavior from sensor noise.
* **Input:** Cartesian coordinates and quaternions.
* **Target:** 6-DOF joint angles.

---

## 6. A-4.2 — Joint-Space IK Baseline
Established a stable baseline using only joint-space supervision.
* **Model:** MLP (7 Input, 6 Output), ReLU, Adam Optimizer.
* **Loss:** $L = \text{MSE}(\theta_{pred}, \theta_{gt})$
* **Finding:** Convergence is stable, but Cartesian accuracy is not guaranteed, confirming that joint-space supervision alone cannot enforce geometric correctness.

## 7. A-4.3 — Forward Kinematics–Aware Learning
Introduces task-space consistency.

### A-4.3.1 — FK-Augmented Loss
A differentiable FK function is introduced to penalize Cartesian position error.
$$L = L_{MSE} + L_{FK}$$

### A-4.3.2 — Hard-Sample Weighted FK Loss (Current Checkpoint)
Uniform weighting under-penalizes difficult workspace regions. This stage introduces a dynamic weighting strategy:
* **Weighting:** $w_i = 1 + \alpha \cdot (e_i / \text{mean}(e))$
* **Final Loss:** $L_{total} = w_{mse} \cdot L_{MSE} + w_{fk} \cdot L_{FK\_hard}$

---

## 8. Training Pipeline (A-4.3.2)
The current pipeline is a clean, reproducible research snapshot:
* **Deterministic:** Seed control and explicit train/val split.
* **Modular:** Separation of Dataset, Model, FK, and Loss functions.
* **Monitoring:** Logs include Total Loss, Joint-Space MSE, FK Mean/Max error.

### 9. Validation-Level FK Error Analysis & Statistical Characterization (A-4.3.3)

In this step, the KR6 inverse kinematics model trained in **A-4.3.2** was evaluated using a dedicated validation analysis pipeline, without modifying the training objective.

#### Objectives
* **Quantify** end-effector Cartesian error induced by the learned IK mapping.
* **Validate** FK-consistency beyond joint-space MSE.
* **Characterize** error distribution using robust statistical metrics.

#### Methodology
1.  The trained IK network was evaluated on the held-out validation split.
2.  Predicted joint angles were passed through the same differentiable forward kinematics function used during training.
3.  End-effector position error was computed as Euclidean distance in Cartesian space.
4.  All validation errors were logged and saved for offline analysis.

#### Results (KR6 – Synthetic Dataset)

| Metric | Value |
| :--- | :--- |
| **Mean FK Error** | 0.0416 m |
| **Std FK Error** | 0.0063 m |
| **95th Percentile** | 0.0492 m |
| **99th Percentile** | 0.0530 m |
| **Maximum Error** | 0.1917 m |

> **Note:** These results demonstrate that the model maintains stable Cartesian accuracy, with the vast majority of predictions remaining below 5 cm error, despite being primarily trained using joint-space supervision.

#### Artifacts
All validation outputs are stored in the following directory:
`analysis_a_4_3_3/`
* `fk_errors.npy`: Raw error values.
* `ee_positions.npy`: Calculated end-effector positions.
* `theta_pred.npy`: Predicted joint angles.

*These artifacts are used as the baseline for comparative and ablation studies in the next stage.*

## 10. Comparative & Ablation Studies (A-4.4)

### A-4.4.1 (MSE-only baseline)
A strict baseline model trained using only joint-space **MSE (Mean Squared Error)**, with **no FK supervision** during training. 

* **Objective:** Forward kinematics error is evaluated solely at validation time for analysis purposes. 
* **Significance:** This experiment establishes a lower-bound performance and highlights the necessity of FK-aware loss formulations introduced in **A-4.3.x**. By isolating the effects of joint-space supervision, we can quantify the accuracy gains provided by integrating the differentiable FK layer.

**Artifacts Location:** `analysis/a_4_4_1/` contains the performance distribution and error logs for this baseline.
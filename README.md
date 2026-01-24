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

### A-4.4.2 — FK-only Training (Pure Cartesian Supervision)

In this experiment, joint-space supervision was completely removed, and the network was trained **solely using FK position error** as the loss function. 

* **Methodology:** The model attempted to learn the inverse mapping by minimizing the Euclidean distance between the target and the predicted end-effector position, without any direct guidance on joint angles ($\theta$).
* **Results:** This approach yielded significantly higher FK errors and unstable convergence compared to both the hybrid and MSE-only baselines.
* **Key Finding:** The results confirm that **Cartesian supervision alone is insufficient** for stable inverse kinematics learning. This failure is primarily due to **solution ambiguity** (the "one-to-many" mapping problem), where the network struggles to converge because multiple different joint configurations can result in the same end-effector position.

**Conclusion:** This experiment validates the necessity of a hybrid loss formulation (as seen in A-4.3.x) to provide a unique and stable gradient path during the learning process.

**Artifacts Location:** `analysis/a_4_4_2/`

### A-4.4.3 — FK Weight ($\lambda$) Ablation Study

In this step, we conduct a systematic ablation study on the forward-kinematics (FK) supervision weight ($\lambda_{fk}$) used in the joint-space + FK hybrid loss.

**Objective:** To quantitatively analyze the trade-off between:
* Joint-space accuracy (joint MSE)
* Task-space accuracy (end-effector FK error)

#### Experimental Setup
* **Base Training Pipeline:** Identical to A-4.3.3.
* **Consistency:** Same dataset split, architecture, optimizer, and training schedule.
* **Variable:** Only $\lambda_{fk}$ is varied while keeping all other components fixed.
* **Evaluated Values:** $\lambda_{fk} \in \{5.0, 2.0, 1.0, 0.5, 0.1, 0.0\}$

#### Results Summary

| $\lambda_{fk}$ | FK Mean Error | FK p95 | FK Max | Joint MSE |
| :--- | :--- | :--- | :--- | :--- |
| **5.0** | 0.0251 | 0.0387 | 0.0531 | 5.6017 |
| **2.0** | 0.0353 | 0.0603 | 0.0693 | 5.5289 |
| **1.0** | 0.0416 | 0.0492 | 0.1917 | 5.4431 |
| **0.5** | 0.3146 | 0.4212 | 0.5006 | 4.4626 |
| **0.1** | 1.5708 | 1.7696 | 2.7919 | 3.2332 |
| **0.0** | 5.3487 | 6.3704 | 6.5510 | 1.8709 |

#### Key Observations
* **High $\lambda_{fk}$ ($\ge 2.0$):** Enforces strong task-space consistency but degrades joint-space accuracy.
* **Low $\lambda_{fk}$ ($\le 0.5$):** Leads to severe FK error despite improved joint MSE.
* **Balanced ($\lambda_{fk} \approx 1.0$):** Provides the most balanced trade-off between joint-space and task-space objectives.

#### Conclusion
This ablation confirms that FK supervision is essential, but its contribution must be carefully weighted.

> The observed trade-off motivates the transition to **adaptive or curriculum-based FK weighting**, which is addressed in **A-4.4.4**.

### A-4.4.4 — Curriculum FK Weighting

This step introduces a **curriculum-based FK supervision strategy**, where the FK loss weight ($\lambda_{fk}$) is gradually increased during training instead of being fixed.

#### Motivation
Previous ablation studies (**A-4.4.3**) demonstrated a critical trade-off:
* **High fixed $\lambda_{fk}$:** Enforces task-space accuracy but harms joint-space learning.
* **Low fixed $\lambda_{fk}$:** Leads to severe FK inconsistency.

Curriculum FK weighting aims to combine the advantages of both regimes by dynamically adjusting the weight.

#### Method
We employ a **linear schedule** defined as:

$$
\lambda_{fk}(\text{epoch}) = \left( \frac{\text{epoch}}{\text{total epochs}} \right) \times \lambda_{max}
$$

**Parameters:**
* $\lambda_{max} = 1.0$
* Total epochs = $100$

> **Strategy:** Early epochs focus primarily on joint-space learning to establish a stable baseline, while FK consistency is enforced progressively as training matures.

#### Results (Validation)
The curriculum strategy yielded the following performance metrics:

* **Joint MSE:** `5.36`
* **FK Mean Error:** `0.0419`
* **FK p95:** `0.0515`
* **FK p99:** `0.0601`

#### Key Findings
* **Comparable Accuracy:** Curriculum FK weighting achieves comparable FK accuracy to the best fixed $\lambda_{fk}$ setting.
* **Stability:** Joint-space regression is more stable and significantly less sensitive to early FK noise.
* **Smoother Dynamics:** Training dynamics are smoother, avoiding the abrupt trade-offs observed in fixed-weight setups.

#### Conclusion
Curriculum FK weighting provides a principled and stable alternative to fixed FK supervision.

> **Recommendation:** This approach forms the recommended default training strategy for **NeuroKinematics** moving forward.

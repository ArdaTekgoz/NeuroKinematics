# NeuroKinematics

This repository contains a complete machine learning–based inverse kinematics
project, including data pipelines, model training, evaluation, and a planned
graphical user interface (GUI).

The project is developed as a full-stack robotics and AI portfolio work.

## Results — A-4.2 (Synthetic Baseline)

Training was performed on a synthetically generated dataset using a
forward-kinematics-consistent inverse kinematics pipeline.

**Training setup**
- Optimizer: Adam
- Epochs: 100
- Dataset: Synthetic FK-consistent dataset
- Loss: Joint MSE + FK position loss

**Final performance**
- Mean FK position error: **0.019 m**
- Max FK position error (batch): ~0.085–0.11 m

This baseline demonstrates that the model can achieve centimeter-level
end-effector accuracy on synthetic data before introducing
error-aware weighting mechanisms.

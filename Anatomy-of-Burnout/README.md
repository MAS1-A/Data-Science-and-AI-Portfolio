# 🔥 Anatomy of Burnout: Multi-Task Machine Learning Pipeline on 5 Million Records

## 📌 1. Project Architecture & Executive Summary
This production-grade framework delivers an end-to-end pipeline mapping the non-linear relationship between digital over-exposure and functional cognitive degradation. Handling a massive array of **5,000,000 instances**, the framework successfully optimizes memory states and engineers custom continuous features to power a hybrid multi-task predictive ecosystem.

### 🎯 Core Modeling Targets:
* **Continuous Forecast (Regression):** Predicting the specific, multi-layered scale of `burnout_risk`.
* **Instance Segmentation (Multi-class Classification):** Stratifying profiles into dynamic `productivity_category` brackets (Low, Medium, High).

---

## 🛠️ 2. Key Engineering Pillars & Optimization

### 📦 2.1 Memory Optimization & Guardrails
* **Action:** Slashing raw RAM usage by over 60% (reducing session footprint down to ~1.2 GB) via tactical datatype downcasting (`float32` and `int8`). This anchors absolute kernel stability across gradient boosting routines.

### 🧠 2.2 Predictive Imputation via LightGBM
* **Action:** Implemented programmatic secondary `LGBMRegressor` engines to infer and fill missing behavioral rows (`social_media_hours`, `sleep_hours`), rather than relying on basic median fallbacks. Explicit boundary clipping (`np.clip`) was enforced to completely block artificial data drift.

### 🧪 2.3 Domain Feature Engineering (Synthesis)
* **Focus Density:** $\frac{\text{focus\_sessions}}{\text{distraction\_frequency} + 1}$
* **Digital Toxicity Index:** $\frac{\text{doomscrolling\_duration} \times \text{app\_switch\_frequency}}{100}$
* **Restorative Sleep Score:** $\text{sleep\_hours} \times \text{sleep\_quality}$
* **Meeting Fatigue Burden:** $\text{meeting\_hours} \times \text{stress\_level}$

---

## 🤖 3. Hyperparameter Tuning Framework (Optuna Matrix)
The hyperparameter space was explored using sequential model-based Bayesian Optimization powered by **Optuna**, utilizing a localized persistent SQLite storage engine (`sqlite:///clf_tuning.db`) for run resilience and step checkpoints. Early stopping and pruning callbacks (`LightGBMPruningCallback`) systematically discarded underperforming trial branches.

### 🎯 Optimized Golden Parameters Secured:
* **Classifier:** `learning_rate`: 0.139, `num_leaves`: 56, `max_depth`: 13, `min_child_samples`: 215, `subsample`: 0.927
* **Regressor:** `learning_rate`: 0.038, `num_leaves`: 71, `max_depth`: 19, `min_child_samples`: 186, `subsample`: 0.545

---

## 🏆 4. Model Evaluation & Performance Metrics

### 📊 4.1 Productivity Classifier Performance
* **Overall Metrics:** Accuracy: **79.94%** | One-vs-Rest ROC-AUC Score: **0.9339** | Cross-Entropy Log Loss: **0.4450**
* **Granular F1-Scores:** * `Low Productivity (0)`: **0.77**
  * `Medium Productivity (1)`: **0.72**
  * `High Productivity (2)`: **0.87**

### 📈 4.2 Burnout Risk Regressor Performance
* **R-Squared Score ($R^2$):** **79.45%** (High directional variance capture)
* **Mean Absolute Error (MAE):** **7.96 points** (Extremely tight average error bound on a 100-point scale)
* **Root Mean Squared Error (RMSE):** **10.00 points**
* **Residual Diagnostics:** Error mapping verified normal distribution variance centered precisely around the zero reference coordinates.

---

## 🌌 5. Pipeline Blueprint
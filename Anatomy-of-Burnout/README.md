# 🔥 Anatomy of Burnout: Multi-Task Machine Learning Pipeline on 5M Records

📊 **Production-Grade Ecosystem** | 🧠 **Bayesian Optimization** | 🚀 **Big Data Scalability**

**Author:** Mohamed Abdullah Sabry  
**Role:** AI & Data Science Specialist  

---

## 📌 1. Project Architecture & Executive Summary

This enterprise-grade framework delivers an end-to-end machine learning pipeline mapping the non-linear relationship between digital over-exposure, functional cognitive degradation, and performance optimization. Handling a massive matrix of **5,000,000 instances**, the architecture successfully enforces low-latency memory states and synthesizes high-signal interaction features to power a hybrid multi-task predictive ecosystem.

### 🎯 Core Modeling Targets

| Target Variable | Task Type | Operational Objective | Output Scale / Classes |
| :--- | :--- | :--- | :--- |
| `burnout_risk` | **Regression** | Continuous forecast of employee exhaustion vulnerability | Continuous ($[0, 100]$ Bounds) |
| `productivity_category` | **Multi-class Classification** | Instance segmentation based on multi-dimensional behavioral traits | Categorical (`Low`, `Medium`, `High`) |

---

## 🛠️ 2. Key Engineering Pillars & Optimization

### 📦 2.1 Memory Optimization & Stability Guardrails
Processing a heavy raw operational footprint ($1.2\text{+ GB}$ active RAM session) demands defensive programming. The pipeline integrates runtime memory type downcasting, strictly converting expansive structures into high-density numeric types (`float32` and `int8`). 
> **Result:** Slashed memory overhead by **over 60%**, anchoring absolute kernel stability across dense gradient boosting iterations without data erosion.

### 🧠 2.2 Predictive Imputation via Sub-LightGBM Models
To bypass the distribution skew caused by naive imputation (e.g., median fallback cascades), the pipeline deploys secondary un-tuned `LGBMRegressor` models. Fully observed features are cross-mapped to infer missing behavioral vectors (`social_media_hours`, `sleep_hours`) sequentially.
* Deterministic clipping bounds (`np.clip`) ensure values remain within genuine physiological boundaries.
* This completely insulates downstream models against artificial data distribution shifts.

### 🧪 2.3 Domain Feature Engineering (Non-Linear Synthesis)
Raw behavioral metrics are transformed into high-expressiveness domain interaction variables to expose underlying physiological constraints to the gradient boosting trees:

* **Focus Density Profile:**
$$\text{Focus Density} = \frac{\text{Focus Sessions}}{\text{Distraction Frequency} + 1}$$

* **Digital Toxicity Index:**
$$\text{Digital Toxicity Index} = \frac{\text{Doomscrolling Duration} \times \text{App Switch Frequency}}{100}$$

* **Restorative Sleep Score:**
$$\text{Restorative Sleep Score} = \text{Sleep Hours} \times \text{Sleep Quality}$$

* **Meeting Fatigue Burden:**
$$\text{Meeting Fatigue Burden} = \text{Meeting Hours} \times \text{Stress Level}$$

---

## 🤖 3. Hyperparameter Tuning Framework (Optuna Matrix)

The hyperparameter optimization workflow navigates complex multi-task search boundaries via sequential model-based Bayesian Tuning powered by **Optuna**. Run resilience is managed through a stateful persistent SQLite storage layer, allowing checkpoint recovery.

```
[Optuna Tuning Engine] ──> [SQLite Checkpoint DB] ──> [MedianPruner Callbacks] ──> [Early Stopping (15 Rounds)]
```

* Automated `LightGBMPruningCallback` routines systematically execute early trials drop on underperforming booster branches.
* Deterministic memory deallocation (`gc.collect()`) prevents RAM leaks across thousands of iterations.

### 🎯 Optimized Golden Parameters Secured

#### 📊 Productivity Classifier Configuration
```python
{
    'n_estimators': 604,
    'learning_rate': 0.13917,
    'num_leaves': 56,
    'max_depth': 13,
    'min_child_samples': 215,
    'subsample': 0.9272,
    'colsample_bytree': 0.5501
}
```

#### 📈 Burnout Risk Regressor Configuration
```python
{
    'n_estimators': 669,
    'learning_rate': 0.03890,
    'num_leaves': 71,
    'max_depth': 19,
    'min_child_samples': 186,
    'subsample': 0.5457,
    'colsample_bytree': 0.5043
}
```

---

## 🏆 4. Model Evaluation & Performance Metrics

### 📊 4.1 Productivity Classifier Performance
The final optimized production classifier demonstrates robust predictive generalization across unseen partitions, ensuring symmetric multi-class precision boundaries.

* **Macro Accuracy (Overall):** **79.94%**
* **One-vs-Rest ROC-AUC:** **0.9339** (Exceptional separation capacity)
* **Cross-Entropy Log Loss:** **0.4450** (High class-assignment confidence)

#### 📋 Granular Evaluation Report
| Performance Tier | Precision | Recall | $F_1$-Score | Partition Support |
| :--- | :---: | :---: | :---: | :---: |
| **Low Productivity (0)** | 0.72 | 0.83 | 0.77 | 168,297 |
| **Medium Productivity (1)** | 0.72 | 0.72 | 0.72 | 356,871 |
| **High Productivity (2)** | 0.90 | 0.84 | 0.87 | 474,832 |

---

### 📈 4.2 Burnout Risk Regressor Performance
Continuous error monitoring confirms tight error variance control, proving the model's reliability at identifying high-risk fatigue thresholds.

* **Coefficient of Determination ($R^2$ Score):** **79.45%** (Captures massive non-linear target variance)
* **Mean Absolute Error (MAE):** **7.96 points** (Extremely tight mistake margin on a 100-point scale)
* **Root Mean Squared Error (RMSE):** **10.00 points**
* **Residual Diagnostics:** Dual-panel density mappings confirm error distribution fields display clean, zero-centered Gaussian behavior.

---

## 🌌 5. Pipeline Blueprint

The entire computational lifecycle is executed through a strictly sequential data pipeline:

```text
       [Raw Data Ingestion]
                │
                ▼
  [Memory Tuning & Downcasting]
                │
                ▼
 [Predictive Imputation (LightGBM)]
                │
                ▼
  [Interaction Feature Synthesis]
                │
                ▼
  [Bayesian Optuna Parameter Search]
                │
                ▼
 [Multi-Task Model Convergence (Fit)]
                │
                ▼
   [Executive Dashboard Delivery]
```

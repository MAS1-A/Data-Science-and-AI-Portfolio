# 🫀 Heart Disease Prediction: 10-Model Algorithmic Benchmark & SHAP Diagnostics

📊 **Multi-Model Evaluation** | 🧠 **Explainable AI (XAI)** | 🛡️ **Zero Data-Leakage Pipeline**

---

## 📌 1. Project Overview & Clinical Objective
This predictive framework evaluates and benchmarks **10 different Machine Learning architectures** to establish a highly deterministic system for early heart failure and coronary risk detection. Processing non-linear clinical metrics (e.g., hemodynamics, ST-segment topologies, metabolic profiles), the project prioritizes corporate data integrity, strict preprocessing isolation, and deep global/local explainability.

---

## 🛠️ 2. Pipeline Architecture & Integrity Guardrails

### 🛡️ 2.1 Anti-Leakage Data Engineering
* **The Vulnerability:** Standard workflows often scale feature vectors before splitting data, causing statistical attributes of the test partition to bleed into the training partition (Data Leakage), yielding inflated, artificial validation metrics.
* **The Enforcement:** This pipeline strictly decouples the dataset **before** optimization. The `StandardScaler` fits parameters ($\mu, \sigma$) exclusively on the training partition ($80\%$) and maps transformations downstream to the test partition ($20\%$).

### ⚙️ 2.2 Discrete Feature Encoding
* **Binary Mapping:** Strict deterministic mapping of binary dimensions (`Sex`: M=1/F=0, `ExerciseAngina`: Y=1/N=0) to lock mathematical behavior.
* **Categorical Expansion:** Multi-state categorical vectors (`ChestPainType`, `ST_Slope`, `RestingECG`) are programmatically expanded via One-Hot Encoding with first-column dropping to avoid multi-collinearity traps.

---

## 🏟️ 3. The 10-Model Arena (The Leaderboard)

Ten diverse classifiers—ranging from classical probabilistic baselines to deep multi-layer neural topologies and gradient-boosted decision trees—were trained under standardized cross-validation constraints ($K=5$).

### 🏆 Algorithm Performance Ranking

| Rank | Model Architecture | Test Accuracy | Sensitivity (Recall) | Precision | $F_1$-Score | 5-Fold CV Mean |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **LightGBM Classifier** | **90.76%** | **90.65%** | **93.27%** | **0.92** | **87.60%** |
| 🥈 | XGBoost Classifier | 89.67% | 90.65% | 91.51% | 0.91 | 86.38% |
| 🥉 | Gradient Boosting Tree | 89.13% | 91.59% | 89.91% | 0.91 | 86.51% |
| 4 | Random Forest | 88.59% | 91.58% | 89.09% | 0.90 | 85.69% |
| 5 | Decision Tree Classifier | 88.04% | 88.79% | 90.48% | 0.90 | 82.56% |
| 6 | K-Nearest Neighbors (KNN) | 88.04% | 88.79% | 90.48% | 0.90 | 86.79% |
| 7 | Support Vector Classifier (SVC) | 87.50% | 86.92% | 91.18% | 0.89 | 84.20% |
| 8 | MLP Neural Network | 85.87% | 87.85% | 87.85% | 0.88 | 84.87% |
| 9 | Gaussian Naive Bayes | 85.87% | 84.11% | 90.91% | 0.87 | 85.97% |
| 10 | Logistic Regression | 85.33% | 84.11% | 90.00% | 0.87 | 86.38% |

> 🩺 **Clinical Significance:** In medical diagnostics, **Sensitivity (Recall)** is paramount. LightGBM captures **90.65%** of true positive cardiac cases, dramatically lowering False Negatives (missed diagnoses) while maintaining an exceptional Precision boundary ($93.27\%$).

---

## 🧠 4. Explainable AI (XAI) via SHAP Diagnostics

To eliminate the "Black Box" limitation of tree-based ensembles, cooperative game-theoretic frameworks (**SHAP Values**) were injected to mathematically map feature attribution.

### 🗺️ 4.1 Global Population Impact (Beeswarm Profiling)
* **ST Slope Predominance:** The clinical absence of an upward ST-segment slope (`ST_Slope_Up = False`) emerged as the absolute `#1` global trigger for high heart disease risk categorization.
* **Ischemic Induction:** Induced cardiovascular strain via exercise (`ExerciseAngina = True`) forms a dense clustering matrix heavily skewing the positive log-odds boundary.

### 🩺 4.2 Local Patient Diagnosis (Waterfall Decision Path)
A case study optimization on high-risk patient instances reveals the specific additive breakdown towards final evaluation:
* **Base Reference ($E[f(x)]$):** Standard population log-odds starting coordinate.
* **Risk Drivers:** A combination of a flat ST-segment (`+0.51` shift) and male physiological profiling (`Sex = 1`, `+0.25` shift) rapidly drove the target score up to a critical positive prediction threshold ($f(x) = 1.588$).

---

## 💾 5. Production Artifact Serialization
The highly generalized champion model and its corresponding isolation scaling parameters are locked for micro-service injection:
* Champion Weights: `LightGBM_heart_disease_model.pkl`
* Preprocessing State: `scaler.pkl`
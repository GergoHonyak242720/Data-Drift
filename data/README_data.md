# Data README  Credit Card Fraud Detection 

**Authors:** Levente Staub (243756), Gergő Honyák (242720), Máté Kovásznai (241960)  



## 1. Project Overview

This project simulates a **real-world MLOps scenario** in which a fraud detection model trained on historical data is exposed to production data that has drifted over time. The goal is to:

- Train a baseline fraud classification model on the original Kaggle dataset
- Detect and quantify **data drift** using statistical methods (KS test, PSI, KL divergence)
- Evaluate how drift impacts model performance
- Build a live monitoring dashboard (Streamlit)

The five `drift_*.csv` files represent simulated **production batches**, each representing 30 days of transaction data with varying degrees of distribution shift and fraud rate change.

---

## 2. Dataset Origins

**Source:** [Kaggle  Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Original publisher:** Machine Learning Group, Université Libre de Bruxelles (ULB)

The original dataset contains **284,807 real credit card transactions** collected over **two days in September 2013** by European cardholders. For privacy reasons, the features V1–V28 have been transformed using **Principal Component Analysis (PCA)** the original feature names and meanings are confidential. Only `Time`, `Amount`, and `Class` remain in their original form.

The `drift_*.csv` files are derived from this dataset with modifications to simulate production drift scenarios.

---

## 3. File Inventory

| File | Rows | Fraud Rate | Role | Notes |
|------|------|------------|------|-------|
| `creditcard.csv` | 284,807 | 0.173% | Baseline training data | Original Kaggle dataset |
| `drift_1.csv` | 15,000 | 0.193% | Production batch 1 | Near-baseline, minimal drift |
| `drift_2.csv` | 15,000 | 0.160% | Production batch 2 | Slightly lower fraud rate |
| `drift_3.csv` | 18,000 | 0.522% | Production batch 3 | Elevated fraud rate, larger batch |
| `drift_4.csv` | 15,000 | 0.167% | Production batch 4 | Near-baseline fraud rate |
| `drift_5.csv` | 15,000 | **2.000%** | Production batch 5 | **Strong concept drift — fraud rate ~11.5× baseline** |

---

## 4. Schema Reference

All files share the same 32-column structure (column order may vary between files):

| Column | Type | Description |
|--------|------|-------------|
| `day` | Integer | Simulated production day (0–29). Represents one calendar month of data per file. |
| `Time` | Float | Seconds elapsed between a transaction and the first transaction in the dataset. |
| `Amount` | Float | Transaction amount in EUR. Not scaled; right-skewed. |
| `V1` – `V28` | Float | PCA-transformed features. Original features are confidential. Mean ≈ 0, Std ≈ 1 for most. |
| `Class` | Integer | Target label. **0 = Legitimate**, **1 = Fraudulent**. |


---

## 5. Dataset Statistics

### Original Training Data (`creditcard.csv`)

| Metric | Value |
|--------|-------|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud rate | 0.173% |
| Time span | 48 hours |
| Missing values | None |
| Imbalance ratio | ~1 : 577 |

### Drift Files Summary

| File | Rows | Fraud Cases | Fraud Rate | Days |
|------|------|-------------|------------|------|
| drift_1 | 15,000 | 29 | 0.193% | 0–29 |
| drift_2 | 15,000 | 24 | 0.160% | 0–29 |
| drift_3 | 18,000 | 94 | 0.522% | 0–29 |
| drift_4 | 15,000 | 25 | 0.167% | 0–29 |
| drift_5 | 15,000 | 300 | 2.000% | 0–29 |

---

## 6. Drift File Descriptions

Each file simulates one month of production traffic. They are designed to test different drift scenarios:

### `drift_1.csv`  Stable Production
Fraud rate (0.193%) is nearly identical to the training baseline. Feature distributions are expected to be close to the training data. This file serves as a **negative control**  a well-behaved model should perform well here with minimal detected drift.

### `drift_2.csv`  Slight Underrepresentation of Fraud
Fraud rate dips slightly to 0.160%. Small distribution shifts may be present in some V features. A model should still perform reasonably well, though precision may vary.

### `drift_3.csv` Increased Volume + Rising Fraud Rate
This batch is larger (18,000 rows) and shows a **3× increase in fraud rate** (0.522%). This is a realistic scenario where a fraud campaign begins mid-month. Expect detectable drift in fraud-discriminative features like V4, V11, V14, and V17. Model recall may improve but precision may drop.

### `drift_4.csv`  Near-Baseline, Possible Covariate Shift
Fraud rate (0.167%) is close to the baseline, but feature distributions may have shifted subtly. This tests whether your drift detector can distinguish **covariate drift** (input distribution change) from **concept drift** (label relationship change).

### `drift_5.csv`  Severe Concept Drift 
Fraud rate jumps to **2.0%**  approximately **11.5× the training baseline**. This represents a major fraud wave or a fundamentally different transaction population. Strong degradation in model performance is expected. This file is the key stress-test for your monitoring system and retraining triggers. The `Amount` column is also in a different position, testing data pipeline robustness.

---

## 7. Key EDA Findings

The following findings come from the EDA notebook (`01_eda.ipynb`) and are critical context for model building and drift monitoring:

### Class Imbalance
- The 1:577 fraud-to-legitimate ratio makes **accuracy a misleading metric**
- Required techniques: SMOTE, `class_weight='balanced'`, threshold tuning
- Required metrics: F1-score, AUC-ROC, Precision-Recall AUC

### Time Patterns
- Legitimate transactions follow a **human circadian rhythm** high activity during the day, low at night
- Fraudulent transactions show a **more uniform time distribution**, consistent with automated/bot activity operating 24/7
- The `hour` feature (derived as `Time / 3600`) can be a useful engineered feature

### Amount Patterns
- Transaction amounts are **highly right-skewed** log transformation is recommended before modeling
- Fraudulent transactions tend toward **lower amounts**  consistent with card testing behavior (small test charges before large fraud)
- `Amount` should be **standardized or log-scaled** before training

### Top Discriminative Features
Using both KS statistic and Pearson correlation with the target label, the following features stand out as the **most predictive** of fraud and the **highest-priority features for drift monitoring**:

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | V14 | Strongest KS statistic  distributions diverge sharply between fraud and legit |
| 2 | V4 | High KS + strong positive correlation with fraud |
| 3 | V11 | Top-ranked by both KS and correlation |
| 4 | V17 | Strong negative correlation with fraud |
| 5 | V12 | Consistent performance across both metrics |
| 6 | V10 | Notable discriminative power |

Features with **near-zero discriminative power** (low KS + near-zero correlation) are candidates for removal during feature selection.

### Feature Correlations
V1–V28 are PCA components and are **by construction uncorrelated with each other**. This means multicollinearity is not a concern for these features. However, `Amount` and `Time` are not PCA-transformed and may correlate weakly with some V features.

---


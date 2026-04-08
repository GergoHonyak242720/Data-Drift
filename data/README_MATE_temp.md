# EDA/Dataset Report  Credit Card Fraud Drift Datasets

This notebook describes the conducted **Exploratory Data Analyzis** and the provided **Dataset**. This file provides a 


## 1. The dataset
The school has provided 5 seperate datasets these datasets are derived from the Kaggle Credit Card Fraud dataset where features V1–V28 are the result of **Principal Component Analysis (PCA)** applied to protect cardholder privacy.

**The role of each file:**

| File | Role in Project |
|------|----------------|
| `drift_1` | **Training / Reference dataset** — model is trained here |
| `drift_2` | Production batch — covariate drift in Amount |
| `drift_3` | Production batch — increased volume, higher fraud rate |
| `drift_4` | Production batch — most similar to baseline |
| `drift_5` | Production batch — severe concept drift, 10× fraud rate |
**Dataset structure:**
| Property | drift_1 | drift_2 | drift_3 | drift_4 | drift_5 |
|----------|---------|---------|---------|---------|---------|
| Rows | 15,000 | 15,000 | **18,000** | 15,000 | 15,000 |
| Columns | 32 | 32 | 32 | 32 | 32 |
| Missing Values | **0** | **0** | **0** | **0** | **0** |
| Duplicate Rows | **0** | **0** | **0** | **0** | **0** |
| Data Types | All numeric | All numeric | All numeric | All numeric | All numeric |
| Negative Amount | None | None | None | None | None |
| Day Range | 0–29 | 0–29 | 0–29 | 0–29 | 0–29 |

## 2. Performed data Cleaning

**Step 1 — Missing value check:** `df.isnull().sum()` confirmed zero missing values across all 5 files. No imputation required.

**Step 2 — Duplicate check:** `df.duplicated().sum()` confirmed zero duplicates. No deduplication required.

**Step 3 — Data type validation:** All columns are float or integer as expected. No type casting needed.

**Step 4 — Class label validation:** `df['Class'].unique()` confirmed only `{0, 1}` values across all files. No label correction needed.

**Step 5 — Negative Amount check:** Minimum Amount ≥ $0.01 in all files. No invalid negative transactions.

**Step 6 — Outlier decision:** Outliers detected using the **>3σ rule** (values more than 3 standard deviations from the mean). Decision: **retain all outliers.**

**Why keep outliers in fraud detection?**  
> In most ML tasks, outliers are noise. In fraud detection, they are often the *signal*. A $62,912 transaction is suspicious by nature — removing it would directly harm recall. Instead, `log1p(Amount)` transformation is applied during modelling to reduce right-skew while preserving relative ordering. Tree-based models (Random Forest, XGBoost) are also inherently robust to extreme values, requiring no outlier removal.

## 3. Cross-Dataset Overview 

### 3.1 Class Distribution
| Dataset | Total Rows | Legit (0) | Fraud (1) | Fraud Rate | Change vs drift_1 |
|---------|-----------|-----------|-----------|-----------|------------------|
| drift_1 | 15,000 | 14,971 | 29 | 0.193% | — Baseline — |
| drift_2 | 15,000 | 14,976 | 24 | 0.160% | −17% |
| drift_3 | 18,000 | 17,906 | 94 | 0.522% | **+171%** ⚠️ |
| drift_4 | 15,000 | 14,975 | 25 | 0.167% | −13% |
| drift_5 | 15,000 | 14,700 | 300 | 2.000% | **+937%** 🚨 |

### 3.2 Transaction Amount

| Dataset | Min | Q25 | Median | Mean | Q75 | Max | Std |
|---------|-----|-----|--------|------|-----|-----|-----|
| drift_1 | $0.01 | $10.15 | $33.05 | $142.65 | $105.00 | $62,912 | $718 |
| drift_2 | $0.05 | **$164.90** | **$363.19** | **$639.87** | **$894.45** | $4,942 | $690 |
| drift_3 | $0.01 | $6.62 | $22.28 | $89.97 | $70.24 | $14,243 | $317 |
| drift_4 | $0.01 | $6.33 | $22.43 | $89.45 | $70.02 | $21,699 | $344 |
| drift_5 | $0.01 | $6.66 | $22.60 | $93.53 | $70.67 | $16,857 | $342 |

### 3.3 Fraud VS Legit transactions

| Dataset | Legit Mean | Legit Median | Fraud Mean | Fraud Median | Fraud Max |
|---------|-----------|-------------|-----------|-------------|----------|
| drift_1 | $142.79 | $33.07 | $71.62 | $30.94 | $564.88 |
| drift_2 | $640.07 | $363.22 | $513.51 | $240.48 | $2,072.61 |
| drift_3 | $89.21 | $22.29 | **$234.13** | $16.87 | $14,242.71 |
| drift_4 | $89.05 | $22.43 | **$332.88** | $22.28 | $5,890.58 |
| drift_5 | $89.03 | $22.41 | **$313.90** | $40.28 | $14,958.51 |


# 🏦 Loan Sanction Amount Prediction

> **Predict how much loan a client qualifies for** using applicant demographics,
> financial profiles, and property characteristics.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Installation & Setup](#4-installation--setup)
5. [Pipeline Walkthrough](#5-pipeline-walkthrough)
   - [Data Cleaning](#51-data-cleaning)
   - [Exploratory Data Analysis](#52-exploratory-data-analysis)
   - [Feature Engineering](#53-feature-engineering)
   - [Preprocessing](#54-preprocessing)
   - [Model Training & Selection](#55-model-training--selection)
   - [Statistical Tests for Categorical Features](#56-statistical-tests-for-categorical-features)
6. [Model Results](#6-model-results)
7. [Key Findings](#7-key-findings)
8. [Outputs](#8-outputs)
9. [Next Steps](#9-next-steps)

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline for **regression** — predicting the
exact dollar amount (`Loan Sanction Amount (USD)`) a bank will sanction for a given loan
application.

The pipeline follows the **CRISP-DM** framework:

```
Data Understanding → Data Preparation → Modelling → Evaluation → Deployment
```

**Problem type:** Regression (continuous target)  
**Target variable:** `Loan Sanction Amount (USD)`  
**Evaluation metrics:** RMSE · MAE · R²

---

## 2. Dataset

| Split | Rows | Columns | Notes |
|-------|------|---------|-------|
| `train.csv` | 30,000 | 24 | Includes target variable |
| `test.csv`  | 20,000 | 23 | No target; for final predictions |

### Feature Groups

| Group | Features |
|-------|----------|
| **Applicant Demographics** | Customer ID, Name, Gender, Age |
| **Financial Profile** | Income (USD), Income Stability, Profession, Type of Employment, Current Loan Expenses (USD), Expense Type 1 & 2, Dependents, Credit Score, No. of Defaults, Has Active Credit Card, Co-Applicant |
| **Loan Request** | Loan Amount Request (USD) |
| **Property Details** | Property ID, Property Age, Property Type, Property Location, Property Price, Location |

### Data Quality Issues Discovered

| Issue | Column(s) | Count | Resolution |
|-------|-----------|-------|------------|
| Sentinel value `-999` | Property Price, Co-Applicant, Target | 352 / 168 / 338 | Replace with `NaN` |
| Sentinel value `'?'` | Co-Applicant, Property Price *(test)* | 168 | Replace with `NaN` |
| **Property Age = Income (USD)** | Property Age | 25,150 rows — all | **Drop column** (perfect collinearity) |
| Negative Property Price | Property Price | 352 rows | Replace with `NaN` |
| Target = 0 (rejected loan) | Loan Sanction Amount | 7,865 rows | Keep — valid label |

> **Key finding:** `Property Age` is an exact copy of `Income (USD)` (Pearson r = 1.0).
> This is a data-entry error. Retaining it would cause perfect multicollinearity in linear
> models and wasted splits in tree models. It is dropped entirely.

---

## 3. Project Structure

```
loan-sanction-prediction/
│
├── train.csv                          ← Raw training data (30,000 rows)
├── test.csv                           ← Raw test data (20,000 rows)
│
├── Loan_Sanction_Prediction.ipynb     ← Full Jupyter notebook (end-to-end)
├── Loan_Sanction_Prediction_Report.pdf← PDF report with all charts & statistics
├── predictions.csv                    ← Final test set predictions
│
└── README.md                          ← This file
```

---

## 4. Installation & Setup

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

### Optional (for extended statistical tests)

```bash
pip install scikit-posthocs        # Dunn's post-hoc test
pip install pingouin               # Comprehensive stats (eta-squared, etc.)
```

### Running the Notebook

```bash
jupyter notebook Loan_Sanction_Prediction.ipynb
```

All data files (`train.csv`, `test.csv`) must be in the same directory as the notebook.

---

## 5. Pipeline Walkthrough

### 5.1 Data Cleaning

```python
# Replace sentinel values
train['Property Price'] = pd.to_numeric(train['Property Price'], errors='coerce')
train.loc[train['Property Price'] == -999, 'Property Price'] = np.nan

# Drop the duplicated column
train = train.drop(columns=['Property Age'])   # identical to Income (USD)

# Handle test set '?' values
test['Co-Applicant'] = test['Co-Applicant'].replace('?', np.nan).astype(float)
```

**Why drop `Property Age`?**
A Pearson correlation of **r = 1.0** with `Income (USD)` means the two columns carry
identical information. Keeping both would:
- Make the design matrix X^T X singular → linear regression coefficients become undefined
- Double-count the same signal in tree models

---

### 5.2 Exploratory Data Analysis

Key EDA findings:

| Finding | Implication |
|---------|-------------|
| Target skewness ≈ 2.1 (right-skewed) | Use log-transforms; prefer non-parametric tests |
| 26% of target values are zero | Zero-inflated distribution; consider two-stage model |
| Sanction/Request ratio clusters at 0.70 & 0.75 | Bank applies fixed policy thresholds |
| Income max = $1.78M vs median = $2,222 | Heavy outliers; use RobustScaler, not StandardScaler |
| Credit Score range: 580–896 | Normal credit score range; no anomalies |

**Target distribution:**
```
Mean:   $47,649    Median: $35,209    Std: $48,221
Zeros:  7,865      Skew:   2.1
```

**Strongest correlations with target:**

| Feature | Pearson r |
|---------|-----------|
| Loan Amount Request (USD) | ~0.68 |
| Property Price | ~0.60 |
| Income (USD) | ~0.45 |
| Credit Score | ~0.35 |
| No. of Defaults | ~-0.15 |

---

### 5.3 Feature Engineering

All engineered features were motivated by domain knowledge or statistical reasoning:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `DTI_Ratio` | Current Expenses / Income | **Debt-to-Income** — primary credit risk metric; high DTI → default risk |
| `LTV_Ratio` | Loan Request / Property Price | **Loan-to-Value** — standard mortgage metric; LTV > 80% triggers PMI |
| `Income_per_Dep` | Income / (Dependents + 1) | Disposable income proxy; families with dependents have less free cash |
| `Request_to_Income` | Loan Request / Income | Months of income to repay — affordability signal |
| `Credit_Band` | Bucketed Credit Score [1–5] | Ordinal encoding preserves monotone risk relationship |
| `Has_Defaults` | `I(No. of Defaults > 0)` | Threshold effect: first default is the critical jump |
| `Age_Band` | Bucketed Age [1–5] | Non-linear life-stage risk profile |
| `log_Income` | `log(1 + Income)` | Variance-stabilising transform for heavy right skew |
| `log_Property Price` | `log(1 + Property Price)` | Same — property values span 3 orders of magnitude |
| `log_Loan Request` | `log(1 + Loan Request)` | Normalises the skewed request distribution |
| `Inc_Stab_enc` | Low=0, High=1 | Ordinal encoding for naturally ordered binary |
| `CC_enc` | Unpossessed=0, Inactive=1, Active=2 | Ordinal — Active card signals better credit behaviour |

**Total features after engineering:** 30

---

### 5.4 Preprocessing

#### Categorical Encoding

| Strategy | Applied To | Why |
|----------|-----------|-----|
| **Label Encoding** | Gender, Location, Profession, Type of Employment, Property Location | High-cardinality nominals; tree models handle non-monotone splits |
| **Ordinal Encoding** | Income Stability, Has Active Credit Card | Natural order carries real meaning |
| **Binary flags** | Expense Type 1 & 2, Co-Applicant, Has Defaults | Already binary Yes/No |

#### Missing Value Imputation

**Strategy: Median Imputation** (fit on train only, then applied to test)

The median is preferred over the mean because:
- Robust to outliers — not distorted by the extreme income values
- Optimal under absolute-error loss
- Prevents data leakage: imputer fit only on training data

#### Feature Scaling

**RobustScaler** is used for linear models (not tree models, which are scale-invariant).

| Scaler | Formula | Outlier Sensitivity |
|--------|---------|-------------------|
| StandardScaler | `(x - mean) / std` | **High** — std inflated by extremes |
| **RobustScaler** ✓ | `(x - Q1) / IQR` | **Low** — uses interquartile range |
| MinMaxScaler | `(x - min) / (max - min)` | **Very High** — bounded by extremes |

With Income outliers at $1.78M (800× the median), StandardScaler compresses 99% of values
into a tiny range, making regularisation ineffective.

---

### 5.5 Model Training & Selection

Five model families are tested, progressing from high-bias to low-bias:

```
Linear Regression → Ridge → Lasso → Random Forest → Gradient Boosting
      (high bias)                                        (low bias)
```

**The Bias-Variance Tradeoff:**

```
E[(y - ŷ)²]  =  Bias²(ŷ)  +  Variance(ŷ)  +  σ²
                  (underfitting)  (overfitting)  (irreducible)
```

| Model | Key Hyperparameters | Scaling Needed |
|-------|---------------------|---------------|
| Linear Regression | — | Yes |
| Ridge (L2) | α = 10 | Yes |
| Lasso (L1) | α = 50 | Yes |
| Random Forest | n = 100, max_depth = 10, min_samples_leaf = 5 | No |
| **Gradient Boosting** ★ | n = 100, lr = 0.1, max_depth = 4 | No |

**Train / Validation split:** 80 / 20 (fixed seed = 42)  
**Cross-validation:** 5-fold K-Fold on the best model

---

### 5.6 Statistical Tests for Categorical Features

The following tests are used to measure the statistical significance of categorical
features against the continuous target (`Loan Sanction Amount`):

#### Categorical → Continuous Target

| Test | Use Case | Assumption |
|------|----------|------------|
| **Mann-Whitney U** | Binary feature (2 groups) | Distribution-free (non-parametric) |
| **Kruskal-Wallis H** | Multi-class feature (3+ groups) | Distribution-free |
| **One-Way ANOVA** | Multi-class, normal residuals | Normality + homoscedasticity |
| **Independent t-test** | Binary feature, normal data | Normality |

> Given target skewness ≈ 2.1, **always prefer non-parametric tests** (Kruskal-Wallis,
> Mann-Whitney) unless working with log-transformed values.

**Effect sizes:**

| Test | Effect Size Metric | Scale |
|------|-------------------|-------|
| ANOVA | **Eta-squared (η²)** | 0.01 = small, 0.06 = medium, 0.14 = large |
| Kruskal-Wallis | **Epsilon-squared (ε²)** | Same interpretation as η² |
| Mann-Whitney | **Rank-biserial r** | -1 to +1, 0 = no effect |

#### Categorical → Categorical (Feature vs Feature)

| Test | Use Case | Assumption |
|------|----------|------------|
| **Chi-Square (χ²)** | Association between two categoricals | Expected cell counts ≥ 5 |
| **Fisher's Exact** | 2×2 table, small samples | None |
| **Cramér's V** | Effect size for chi-square | 0 = no association, 1 = perfect |

**Example usage:**

```python
from scipy import stats
import pandas as pd

# Kruskal-Wallis: Does Profession affect Loan Sanction Amount?
groups = [df.loc[df['Profession'] == p, 'Loan Sanction Amount (USD)'].dropna()
          for p in df['Profession'].dropna().unique()]
h_stat, p_value = stats.kruskal(*groups)
print(f"H = {h_stat:.4f}, p = {p_value:.4e}")

# Mann-Whitney U: Does Income Stability affect Loan Sanction Amount?
low  = df.loc[df['Income Stability'] == 'Low',  'Loan Sanction Amount (USD)'].dropna()
high = df.loc[df['Income Stability'] == 'High', 'Loan Sanction Amount (USD)'].dropna()
u_stat, p_value = stats.mannwhitneyu(low, high, alternative='two-sided')
print(f"U = {u_stat:.0f}, p = {p_value:.4e}")

# Chi-Square: Are Profession and Location independent?
ct = pd.crosstab(df['Profession'], df['Location'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"χ² = {chi2:.4f}, df = {dof}, p = {p:.4e}")
```

---

## 6. Model Results

### Validation Set Performance (20% holdout)

| Model | RMSE (USD) | MAE (USD) | R² |
|-------|-----------|----------|-----|
| Linear Regression *(baseline)* | $27,686 | $18,945 | 0.659 |
| Ridge (L2, α=10) | $27,690 | $18,945 | 0.659 |
| Lasso (L1, α=50) | $27,688 | $18,915 | 0.659 |
| Random Forest (n=100, d=10) | $21,967 | $10,346 | 0.786 |
| **Gradient Boosting (n=100, lr=0.1)** ★ | **$21,936** | **$11,630** | **0.786** |

### Why Gradient Boosting Wins

- Linear models plateau together (RMSE ≈ $27,688) → confirmed the bottleneck is
  **non-linearity**, not multicollinearity or overfitting
- GBM reduces RMSE by **~21%** over the linear baseline by fitting residuals sequentially
- Random Forest and GBM perform almost identically — GBM edges ahead via lower bias
  through sequential residual fitting

### Top 10 Predictive Features

| Rank | Feature | Type |
|------|---------|------|
| 1 | `log_Request` | Engineered |
| 2 | `Loan Amount Request (USD)` | Original |
| 3 | `Credit Score` | Original |
| 4 | `Has_CoApplicant` | Engineered |
| 5 | `Inc_Stab_enc` | Engineered |
| 6 | `Age` | Original |
| 7 | `Property Price` | Original |
| 8 | `Request_to_Income` | Engineered |
| 9 | `Income_per_Dep` | Engineered |
| 10 | `log_PropPrice` | Engineered |

> **6 of the top 10 features are engineered** — validating the feature engineering step.

---

## 7. Key Findings

### 1 — Loan Request Amount Is the Strongest Predictor
Banks sanction a fixed fraction of what is requested. The ratio clusters tightly at
**0.70 and 0.75**, strongly suggesting hard policy thresholds rather than continuous
underwriting. Any model predicting sanction amount is effectively learning to assign
applicants to policy tiers.

### 2 — Income and Property Value Matter as Much as Credit Score
Pearson correlations show that income (ability to repay) and property value (collateral)
have comparable or stronger linear association with sanction amount than credit score.

### 3 — The First Default Is the Critical Threshold
The binary `Has_Defaults` flag outranks the raw `No. of Defaults` count in feature
importance — confirming that the jump from 0 → 1 default matters far more than
marginal additional defaults.

### 4 — Income Stability Is a Key Underwriting Signal
High-stability applicants receive median sanctions 15–20% higher. Lenders rationally
discount volatile income streams since future repayment ability is less certain.

### 5 — The Zero Loans Are Meaningful
7,865 sanctioned amounts are exactly zero — rejected applications. A production-grade
pipeline should model this with a **two-stage approach**:
- **Stage 1:** Classify reject vs approve (logistic regression / GBM)
- **Stage 2:** Predict amount only for approved applications

---

## 8. Outputs

| File | Description |
|------|-------------|
| `Loan_Sanction_Prediction.ipynb` | Full runnable notebook with markdown explanations |
| `Loan_Sanction_Prediction_Report.pdf` | Comprehensive PDF report — EDA, statistics, model results |
| `predictions.csv` | Final test set predictions (`Customer ID` + `Predicted_Loan_Sanction_USD`) |

**Prediction summary (test set):**

```
Min:  $0          Mean: $47,227
Max:  $317,016    Std:  ~$48,000
```

---

## 9. Next Steps

## Statistical Reference

### Metrics Used

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **RMSE** | `√(Σ(y - ŷ)² / n)` | Penalises large errors quadratically; same units as target |
| **MAE** | `Σ|y - ŷ| / n` | Average absolute error; robust to outliers |
| **R²** | `1 - SS_res / SS_tot` | Proportion of variance explained; 1.0 = perfect |

### Why Non-Parametric Tests?

The target variable violates normality (skew ≈ 2.1, zero-inflated). Non-parametric tests
make no distributional assumptions — they rank-transform the data before testing, making
them robust to:
- Non-normal distributions
- Heavy tails and outliers
- Zero-inflated responses

---

*Built with Python 3 · scikit-learn · pandas · matplotlib · seaborn · scipy · reportlab*

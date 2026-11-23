### Supervised Machine Learning & Financial Risk Modeling
**Invoice Delay Regression & Credit Default Classification**  
<sub>`Python` • `Scikit-Learn` • `MSc in AI & Deep Learning (UNIWA)`</sub>

---

This project contains the machine learning pipelines, empirical evaluations, and statistical proofs developed for the **Fundamentals of Artificial Intelligence & Machine Learning (AIDL01)** course in the **MSc in Artificial Intelligence and Deep Learning** at the **University of West Attica**.

It investigates two foundational supervised learning domains in financial analytics:
1. **Supervised Regression:** Forecasting Accounts Receivable settlement delays (`DaysToSettle`) to optimize working capital and corporate cash flow.
2. **Supervised Classification:** Assessing borrower credit default risk (`loan_status`) under class imbalance.

---

## 📑 Project Structure
```text
├── Financial_Risk_and_Delay_Modeling.ipynb  # Full pipeline: EDA, feature engineering, modeling & cross-validation
├── Fotis_Papadopoulos_AIDL01_Machine_Learning_Report.pdf  # Technical report with mathematical proofs
└── README.md                                # Findings and performance summary
```

---

## 1. Regression Analysis: Invoice Payment Delay (`DaysToSettle`)

### 🎯 Business Objective & Data Leakage Prevention
Corporate cash flow depends on predictable receivables. Using the **IBM Watson Analytics Accounts Receivable dataset** (2,466 invoices), this task predicts the exact number of days required to settle an invoice (`DaysToSettle`).
- **Data Leakage Safeguard:** Future-looking attributes (`SettledDate`, `DaysLate`) were intentionally excluded during preprocessing, ensuring predictions rely strictly on information available at the exact moment of invoice issuance.
- **Feature Engineering:** Derived `PaymentTerms` by computing the difference between `DueDate` and `InvoiceDate`.

### 🔬 Models & Cross-Validation
- **Linear Regression (Baseline):** Standard multiple linear regression.
- **Decision Tree Regressor:** Compared unpruned baseline vs. a 5-fold `GridSearchCV` tuned model pruned to `max_depth=3`.
- **Random Forest Regressor:** Compared default unconstrained forest vs. a tuned ensemble (`n_estimators=150`, `max_depth=3`).

### 📊 Regression Performance Comparison (Test Set)
| Model | MAE (Days) | RMSE (Days) | $R^2$ Score | Overfitting Status |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 8.44 | 10.60 | 0.1947 | Well-generalized (Train/Test aligned) |
| **Default Decision Tree** | 10.97 | 14.54 | -0.5181 | **Severe Overfitting** (Unpruned) |
| **Tuned Decision Tree (`max_depth=3`)** | 8.44 | 10.62 | 0.1923 | Resolved (Pruned) |
| **Default Random Forest** | 8.91 | 11.23 | 0.0956 | Moderate Overfitting |
| **Tuned Random Forest (`n_estimators=150`)** | **8.43** | **10.57** | **0.1993** | **Best Performance & Lowest Error** |

> **Key Takeaway:** Unpruned tree-based models memorized training noise. Constraining tree depth (`max_depth=3`) eliminated overfitting and produced the lowest prediction error ($MAE = 8.43$ days).

---

## 2. Classification Analysis: Credit Default Risk (`loan_status`)

### 🎯 Objective & Class Imbalance Challenge
Using the **Kaggle Credit Risk Dataset** (5,000 borrower records, 12 features), models predict the probability of loan default (`loan_status = 1`).
- **Class Imbalance:** 77.5% Safe Loans vs. 22.5% Defaults.
- **Evaluation Strategy:** Due to the skew, accuracy alone is misleading (a naive model predicting all zeros achieves 77.5% accuracy). Evaluation prioritized **Precision, Recall, F1-Score, and ROC-AUC**.

### 🛠️ Data Preprocessing Pipeline
1. **Missing Data Imputation:** Median imputation applied to `person_emp_length` and `loan_int_rate` to protect against extreme outliers.
2. **Categorical Encoding:** One-hot encoding with `drop_first=True` to eliminate dummy variable trap / multicollinearity.
3. **Feature Scaling:** `StandardScaler` fitted exclusively on training data to prevent data leakage in distance/gradient algorithms.

### 📊 Classification Performance Comparison (Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.8533 | 0.7878 | 0.5346 | 0.6370 | 0.8530 |
| **K-Nearest Neighbors ($K=5$)** | 0.8693 | 0.8171 | 0.5900 | 0.6849 | 0.8252 |
| **Default Random Forest** | 0.8873 | 0.9004 | 0.6731 | 0.7702 | 0.8812 |
| **Tuned Random Forest (`max_depth=7`)** | **0.8960** | **0.9116** | **0.6288** | **0.7443** | **0.8910** |

> **Key Takeaway:** The **Tuned Random Forest Classifier** emerged as the top performer, achieving **91.2% Precision** and an **ROC-AUC of 0.8910**, ensuring high reliability for institutional credit risk assessments.

---

## 🧮 Mathematical & Theoretical Foundations (Report Appendix)
The accompanying technical report contains complete mathematical derivations and statistical proofs:
- **Z-Score Normalization:** Proof that standardized variables strictly have zero mean ($\mathbb{E}[Z]=0$) and unit variance ($\text{Var}(Z)=1$), and proof that correlation equals the covariance of standardized variables.
- **Logistic Regression Loss & Gradients:** Full derivation of Binary Cross-Entropy loss optimization via gradient descent.
- **Determinism in L-BFGS Solver:** Mathematical rationale for `random_state` in full-batch Logistic Regression, exploring second-order Taylor approximations (Newton-Raphson) and handling singular Hessian matrices.
- **Gini Impurity Probability Model:** Classical urn model derivation showing why $I_G = 1 - \sum p_i^2$ represents the exact misclassification probability, with mathematical proof of maximum impurity at 0.5.
- **ROC-AUC Geometry:** Geometric and coordinate proofs establishing the theoretical boundary values ($AUC = 1.0$ for a perfect classifier, $AUC = 0.5$ for random guessing).

---

## 💻 Tech Stack
- **Languages & Frameworks:** Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn.
- **Techniques:** Cross-Validation (5-fold), Hyperparameter GridSearch, Ensemble Learning, Monotonic Transformations, Feature Scaling.

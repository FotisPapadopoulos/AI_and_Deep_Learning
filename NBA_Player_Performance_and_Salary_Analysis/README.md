### NBA Player Performance & Salary Analysis (2010–2025)
**Econometric & Exploratory Data Analysis of Athletic Output vs. Contract Compensation**  
<sub>`Python` • `Pandas` • `Seaborn` • `MSc in AI & Deep Learning (UNIWA)`</sub>

---

This repository contains the exploratory data analysis, econometric modeling, and mathematical statistical evaluations developed for the **Platforms for AI and Python Programming (AIDL_03)** course in the **MSc in Artificial Intelligence and Deep Learning** at the **University of West Attica**.

The study investigates the economic relationship between athletic on-court production (scoring volume, shooting efficiency, positional roles) and financial compensation (salaries) in the National Basketball Association over a 15-year period (2010–2025).

---

## 📑 Project Structure
```text
├── Papadopoulos_Fotis_aidl_0114.ipynb        # Complete Python EDA pipeline: cleaning, stats & visualizations
├── Fotis_Papadopoulos_NBA_Salary_Report.pdf  # Comprehensive academic research report with statistical proofs
└── README.md                                 # Executive summary of methodology & discoveries
```

---

## 📊 Dataset & Integrity Pipelines
- **Data Sourcing:** Merged longitudinal data combining **Basketball-Reference** (box scores, minutes, per-game metrics) and **HoopsHype** (historical contract valuations and annual earnings).
- **Dataset Scale:** 7,298 player-season records across 31 performance and financial variables.
- **Data Quality Safeguards:**
  - Evaluated missing shooting percentages (e.g., zero 3-point attempts resulting in valid mathematical undefined states).
  - Resolved mid-season trade consolidations (preventing duplicate stat counting).
  - Addressed nominal duplicate player names (e.g., Chris Johnson in 2013, Tony Mitchell in 2014).

---

## 🔍 Key Research Questions & Empirical Findings

### 1. Positional Salary Distributions & The "Aging Centers" Paradox
- **Extreme Right Skew:** Player compensation exhibits severe positive skewness ($2.41$) and heavy-tailed leptokurtic excess kurtosis ($6.56$), driven by supermax contracts ($CV = 1.32$).
- **Career Stage Dynamics:** Sub-grouping across career stages (**Young** <25, **Prime** 25–30, **Veteran** >30) uncovered an unexpected positional divergence:
  - While Point Guards and Forwards maintain or expand their median earnings into their veteran years, **Centers (C) suffer a sharp decline in median salary and contract spread post-age 30**, reflecting team reluctance to invest heavily in aging big men facing athletic decay.

### 2. Scoring Profiles & Scoring Rate (PTS / Minute)
- Engineered a categorical **Scoring Profile** based on playing time (`Bench`, `Role Player`, `Star`).
- Verified that Star players score at a substantially faster rate per minute ($0.53\text{ PTS/MP}$) compared to Role Players ($0.39$) and Bench players ($0.33$).
- Tight 95% confidence intervals confirmed that high playing time is directly linked to superior scoring rate across all five positions.

### 3. Shooting Volume vs. Shooting Efficiency: The Compensation Driver
- **Shooting Volume ($FGA$):** Demonstrates a strong positive linear relationship with salary ($r = 0.65, R^2 = 0.42$). Teams heavily compensate offensive facilitators who shoulder primary shot creation duties.
- **Shooting Efficiency ($eFG\%$):** Displays near-zero linear correlation ($r = 0.08$) with contract compensation, forming a vertical funnel pattern. High efficiency on low volume does not translate to salary; elite earners maintain stable 50–60% $eFG\%$ while absorbing massive volume.

### 4. The Diminishing Returns Hypothesis (Points per Million Dollars)
- Formulated and tested the financial metric $\text{Points per Million Dollars} = \frac{\text{PTS}}{\text{Salary (in Millions)}}$.
- **Empirical Confirmation:**
  - **Low Tier (<$2M):** Median efficiency of $4.0\text{ PTS/\$M}$ (high per-dollar value driven by rookie-scale and minimum contracts).
  - **Mid Tier ($2M–$10M):** Median efficiency drops to $2.0\text{ PTS/\$M}$.
  - **High Tier (>$10M):** Median efficiency collapses to $0.8\text{ PTS/\$M}$.
- **Economic Insight:** Confirms the **"Superstar Premium"**. Under salary cap constraints, teams willingly overpay for top-tier individual scoring because scarce, championship-caliber scoring output is non-additive.

---

## 🧮 Mathematical & Statistical Appendix Highlights
The accompanying academic paper details formal derivations and proofs:
- **Sample Mean Variance Scaling:** Analytical proof of $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$ and step-by-step derivation of Z-interval bounds.
- **Skewness & Kurtosis Moments:** Mathematical proof of zero skewness for symmetric densities, standard normal kurtosis derivation ($\kappa = 3$), and sample bias-correction multipliers.
- **OLS Regression Variance & Confidence Bands:** Mathematical proof for the standard error of prediction and the derivation of the "hourglass" flaring effect at distribution extremities.

---

## 💻 Tech Stack
- **Environment & Tools:** Python 3.11+, Jupyter Notebook.
- **Data & Analytics Libraries:** Pandas, NumPy, SciPy.
- **Visualization Suite:** Seaborn, Matplotlib.

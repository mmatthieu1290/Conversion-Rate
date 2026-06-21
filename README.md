# Conversion Rate Prediction

Binary classification to predict whether a user will subscribe to a newsletter, based on browsing behavior and demographic features. Includes systematic comparison of multiple models and business recommendations derived from model interpretation.

---

## Results

**Best model:** XGBoost (without `source` feature)
**F1-score on test set:** 0.78
**Most predictive feature:** `total_pages_visited`

---

## Dataset

284,580 observations — imbalanced dataset (3.2% positive class).

| Feature | Description |
|---|---|
| `country` | User country (China, US, UK, Germany) |
| `age` | User age |
| `new_user` | 1 if first visit, 0 otherwise |
| `source` | Traffic source (Ads, SEO, Direct) |
| `total_pages_visited` | Number of pages visited in the session |
| `converted` | Target — 1 if subscribed, 0 otherwise |

---

## Methodology

1. **EDA** — conversion rates by country, source, and new/returning users; logistic curves for age and pages visited
2. **Preprocessing** — OneHotEncoding (country, source), StandardScaler (age, total_pages_visited), stratified train/test split
3. **Feature selection** — four feature combinations tested (all features, without source, without age, without new_user)
4. **Model comparison** — Logistic Regression, KNN, XGBoost with GridSearchCV (5-fold CV, optimized on F1)
5. **Evaluation** — F1-score, confusion matrices, ROC curves across all model/feature combinations
6. **Business recommendations** — derived from LR coefficients and conversion rate analysis

---

## Key Findings

- `total_pages_visited` is by far the strongest predictor of conversion
- Returning users convert at higher rates than new users
- Chinese users show very low conversion rates (~0.1% vs ~4-6% for other countries)
- Traffic source has minimal impact on conversion probability

**Recommendations to the newspaper:** prioritize engagement strategies that encourage users to visit more pages and return to the site; do not invest in promotion in China.

---

## Models Compared

| Model | Feature set | F1 test |
|---|---|---|
| XGBoost | Without source | 0.78 |
| XGBoost | All features | 0.78 |
| Logistic Regression | All features | 0.78 |
| KNN | All features | 0.78 |

---

## Files

| File | Description |
|---|---|
| `Version_finale_projet3_Matthieu_Marechal.ipynb` | Full notebook |
| `conversion_data_train.csv` | Labeled dataset (284,580 rows) |
| `conversion_data_test.csv` | Unlabeled dataset for prediction |
| `conversion_data_test_predictions_MARECHAL.csv` | Final predictions |

---

## Installation

```bash
git clone https://github.com/mmatthieu1290/Conversion-Rate
cd Conversion-Rate
pip install pandas numpy scikit-learn xgboost plotly seaborn
```

Then open and run `Version_finale_projet3_Matthieu_Marechal.ipynb` in Jupyter.

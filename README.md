# 🏡 House Prices — Advanced Regression Techniques

Predicting residential house sale prices in Ames, Iowa from 79 descriptive features — a full machine learning pipeline built for the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=flat)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

> Machine Learning project — Master's in AI & Data, Université Paris Dauphine-PSL

---

## 🎯 Results

| Model | CV RMSLE ↓ |
|---|---|
| ⭐ **Stacking (Ridge meta-model)** | **0.1063 ± 0.0031** |
| ElasticNet (tuned) | 0.1084 |
| Lasso (tuned) | 0.1085 |
| Ridge (tuned) | 0.1105 |
| CatBoost | 0.1123 |
| XGBoost | 0.1125 |
| Gradient Boosting | 0.1152 |
| LightGBM | 0.1217 |

Stacking six base learners reduced error by ~2% over the best single model and divided the variance across folds by almost 3.

## 🔧 Pipeline

1. **EDA**: target distribution (skewness 1.88 → 0.12 after `log1p`), Q-Q plots, missing values, ANOVA on categorical features, correlation heatmap
2. **Cleaning**: domain-aware imputation (`"None"` for absent features, neighborhood median for `LotFrontage`), removal of documented outliers
3. **Ordinal encoding** for quality scales (`Po` → `Ex`)
4. **Feature engineering**: `TotalSF`, `TotalBath`, `HouseAge`, `QualxSF` interaction, binary indicators, `log1p` on 35 skewed features
5. **Preprocessing pipeline**: `ColumnTransformer` with `RobustScaler` + `OneHotEncoder` (243 final features)
6. **Benchmark** of 7 models with 7-fold cross-validation
7. **Hyperparameter tuning** with `GridSearchCV` (Ridge, Lasso, ElasticNet)
8. **Stacking**: Ridge, Lasso, ElasticNet, Gradient Boosting, LightGBM, CatBoost → Ridge meta-model
9. **Kaggle submission** generated from the final model

## 💡 Key insights

- `OverallQual`, `GrLivArea` and the engineered `TotalSF` / `QualxSF` are the strongest price drivers.
- Regularized linear models outperformed gradient boosting on this dataset once the features were properly transformed.
- Combining linear and tree-based models in a stack gave the most accurate and most stable predictions.

## 🚀 Run it

```bash
git clone https://github.com/ranimmhiri/house-prices-kaggle.git
cd house-prices-kaggle
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), place them in a `data/` folder, then open `house_prices.ipynb`.

## 📁 Structure

```
house-prices-kaggle/
├── house_prices.ipynb   # full analysis & modeling
├── requirements.txt
└── data/                # Kaggle CSVs (not versioned)
```

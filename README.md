# House Price Prediction

An end-to-end machine learning project that predicts house sale prices from the Ames Housing dataset (Kaggle: *House Prices - Advanced Regression Techniques*). The project covers exploratory data analysis, leakage-free preprocessing with scikit-learn pipelines, a Random Forest baseline, hyperparameter tuning, and log-target modeling.

## Highlights

- Leakage-safe preprocessing: all imputation statistics (mode, median) are learned on the training split only, inside a single `Pipeline`.
- Domain-aware missing value handling (e.g. `NaN` in `PoolQC` means "no pool", not "unknown").
- Custom scikit-learn transformers, `ColumnTransformer` with explicit column groups, and a column-coverage check so no feature is silently dropped.
- Evaluation with RMSE on log(price), the standard metric for this dataset, plus MAE and R².
- 5-fold cross-validation and `RandomizedSearchCV` run on the full pipeline.

## Project Workflow

1. **EDA**: structure, missing values, target distribution (right-skewed), outliers, correlations, categorical cardinality.
2. **Data cleaning decisions**
   - Categorical `NaN` meaning "feature absent" -> `"None"` (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`, `BsmtQual`, `BsmtCond`, `BsmtFinType1`, `MasVnrType`).
   - `BsmtExposure` / `BsmtFinType2`: `"None"` if the house has no basement, otherwise the training-set mode.
   - `MasVnrArea`, `GarageYrBlt`: `NaN` -> `0`.
   - `Electrical`: training-set mode.
   - `LotFrontage`: median of the house's neighborhood (global median as fallback).
3. **Feature engineering**: `HasGarage`, `HasPool`, `HasMiscFeature`, `HasBasement`; `MSSubClass` cast to string (it is categorical); `Id` dropped.
4. **Encoding**
   - Ordinal encoding with explicit category order for quality-type columns (`Po < Fa < TA < Gd < Ex`), `BsmtExposure`, `BsmtFinType1/2`, `GarageFinish`.
   - One-hot encoding (`handle_unknown="ignore"`) for nominal columns.
5. **Modeling**: Random Forest baseline, then log-transformed target via `TransformedTargetRegressor`.
6. **Evaluation & tuning**: hold-out validation, 5-fold CV, `RandomizedSearchCV`.

## Pipeline Architecture

```
HouseFeatureAdder
      |
BasementMissingHandler
      |
LotFrontageImputer
      |
ColumnTransformer
   |- qual / expo / fintype / gfin  -> constant imputer + OrdinalEncoder
   |- nominal                       -> constant imputer + OneHotEncoder
   |- electrical                    -> most-frequent imputer + OneHotEncoder
   |- zero (MasVnrArea, GarageYrBlt)-> constant 0
   |- numeric                       -> median imputer
   |- binary (Has* flags)           -> passthrough
      |
RandomForestRegressor
      |
TransformedTargetRegressor (log1p / expm1)
```

The final preprocessed matrix has 259 features with no NaN or infinite values.

## Results

Validation split: 80/20, `random_state=42`. Metric: RMSE on log(price) (lower is better).

| Model | CV RMSE (log), 5-fold | Validation RMSE (log) |
|---|---|---|
| Random Forest baseline (300 trees) | 0.14581 ± 0.02041 | 0.14929 |
| Random Forest, tuned (RandomizedSearchCV) | 0.14393 | 0.15011 |
| Random Forest + log target | — | **0.14443** |

Baseline validation scores: MAE ≈ 17,102, R² ≈ 0.896.

**Takeaways**
- Hyperparameter tuning of the Random Forest gave no meaningful gain; the CV improvement is well inside the fold-to-fold standard deviation.
- Log-transforming the target gave the largest improvement.

## Tech Stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, Jupyter.

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook
```

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place them in the `data/` folder.

## Repository Structure

```
.
├── data/                # train.csv, test.csv (not tracked)
├── notebooks/           # EDA and modeling notebook(s)
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Gradient boosting models (GradientBoosting, XGBoost, LightGBM)
- [ ] Linear models (Ridge, Lasso) with scaling and skewness correction
- [ ] Ensembling with `VotingRegressor` / `StackingRegressor`
- [ ] Error analysis on large-residual houses and outliers
- [ ] Better handling of `GarageYrBlt` (e.g. garage age instead of filling with 0)

## License

MIT (change as needed).

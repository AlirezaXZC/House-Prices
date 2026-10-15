# House Price Prediction

An end-to-end machine learning project that predicts house sale prices
using the Ames Housing dataset from Kaggle's [House Prices: Advanced
Regression
Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition.

The project covers exploratory data analysis (EDA), leakage-safe
preprocessing with scikit-learn pipelines, Random Forest modeling,
hyperparameter tuning, and log-target modeling.

## Highlights

-   **Leakage-safe preprocessing:** imputation statistics are learned
    from training data within the preprocessing pipeline.
-   **Domain-aware missing-value handling:** for example, a missing
    `PoolQC` value represents the absence of a pool, rather than an
    unknown pool quality.
-   **Custom transformers:** dedicated handling for house features,
    basement-related missing values, and `LotFrontage`.
-   **Explicit feature groups:** `ColumnTransformer` separates ordinal,
    nominal, numeric, zero-imputed, and binary features.
-   **Feature coverage check:** verifies that expected features are
    accounted for during preprocessing.
-   **Evaluation:** RMSE on log-transformed sale prices, alongside MAE
    and R².
-   **Model validation:** hold-out validation, 5-fold cross-validation,
    and `RandomizedSearchCV`.

## Project Workflow

1.  **EDA:** inspect dataset structure, missing values, target
    distribution, outliers, correlations, and categorical cardinality.
2.  **Data-cleaning decisions:**
    -   Categorical missing values that indicate an absent feature are
        filled with `"None"` (including `PoolQC`, `MiscFeature`,
        `Alley`, `Fence`, `FireplaceQu`, garage-related categorical
        columns, `MasVnrType`, and basement-quality/finish columns).
    -   `BsmtExposure` and `BsmtFinType2` receive special handling:
        `"None"` when the house has no basement; otherwise, remaining
        missing values are imputed using the mode learned from training
        data.
    -   `MasVnrArea` and `GarageYrBlt`: missing values are replaced with
        `0`.
    -   `Electrical`: missing values are replaced with the training-set
        mode.
    -   `LotFrontage`: imputed using the neighborhood median, with the
        global training-set median as a fallback.
3.  **Feature engineering:** add `HasGarage`, `HasPool`,
    `HasMiscFeature`, and `HasBasement`; cast `MSSubClass` to a
    categorical string; drop `Id` from model features.
4.  **Encoding:**
    -   Ordinal encoding with explicit category order for quality and
        other ordinal features. For quality columns, the order includes
        `None < Po < Fa < TA < Gd < Ex`.
    -   One-hot encoding for nominal categorical features, with
        `handle_unknown="ignore"`.
5.  **Modeling:** train a Random Forest baseline and evaluate a model
    using a log-transformed target via `TransformedTargetRegressor`.
6.  **Evaluation and tuning:** compare hold-out validation and 5-fold
    cross-validation results; run `RandomizedSearchCV` on the full
    estimator pipeline.

## Pipeline Architecture

The preprocessing pipeline is wrapped by `TransformedTargetRegressor`
for the log-target experiment.

``` text
TransformedTargetRegressor
  target transform: log1p / inverse transform: expm1
  └── Pipeline
      ├── HouseFeatureAdder
      ├── BasementMissingHandler
      ├── LotFrontageImputer
      ├── ColumnTransformer
      │   ├── Ordinal features -> imputation + OrdinalEncoder
      │   ├── Nominal features -> imputation + OneHotEncoder
      │   ├── Electrical -> most-frequent imputation + OneHotEncoder
      │   ├── Zero-imputed features (MasVnrArea, GarageYrBlt) -> constant 0
      │   ├── Other numeric features -> median imputation
      │   └── Binary indicators (Has* flags) -> passthrough
      └── RandomForestRegressor
```

The final preprocessed matrix has **259 features**, with no NaN or
infinite values in the checked output.

## Results

Validation split: 80/20, `random_state=42`. The primary comparison
metric is RMSE on log-transformed sale price; lower is better.

  Experiment                                         5-fold CV RMSE (log)   Validation RMSE (log)
  ------------------------------------------------ ---------------------- -----------------------
  Random Forest baseline (300 trees)                    0.14581 ± 0.02041                 0.14929
  Random Forest hyperparameter-tuning experiment                  0.14393                 0.15011
  Random Forest with log-transformed target                           ---             **0.14443**

Baseline validation metrics: MAE ≈ 17,102 and R² ≈ 0.896.

### Takeaways

-   Hyperparameter tuning did not produce a meaningful improvement on
    the reported validation split. The CV difference should be
    interpreted cautiously given fold-to-fold variation.
-   In the reported experiments, transforming the target to log space
    produced the best validation RMSE.
-   **Model-selection note:** the tuned hyperparameters and the
    log-target experiment are separate experiments in the current
    results. The final log-target model should only be described as
    "tuned" if the selected parameters are explicitly applied to that
    estimator and the result is re-evaluated.

## Tech Stack

Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, and Jupyter
Notebook.

## Getting Started

Clone the repository:

``` bash
git clone https://github.com/AlirezaXZC/House-Prices.git
cd House-Prices
```

Install the core dependencies:

``` bash
pip install numpy pandas scikit-learn seaborn matplotlib jupyter
```

Open the notebook:

``` bash
jupyter notebook
```

Download `train.csv` and `test.csv` from the [Kaggle competition data
page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place them in the `1- Data/` directory. The data files are not
tracked in this repository.

## Repository Structure

``` text
.
├── 1- Data/                  # Local Kaggle data files (not tracked by Git)
├── 2- NoteBooks/
│   └── 01_Exploration.ipynb  # EDA and modeling notebook
├── .gitignore
├── LICENSE
└── README.md
```

## Roadmap

-   [ ] Evaluate gradient boosting models (Gradient Boosting, XGBoost,
    LightGBM).
-   [ ] Compare linear models (Ridge, Lasso) with appropriate scaling
    and skewness handling.
-   [ ] Explore ensembling with `VotingRegressor` and
    `StackingRegressor`.
-   [ ] Perform error analysis on houses with large residuals and on
    outliers.
-   [ ] Investigate alternative `GarageYrBlt` handling, such as garage
    age or a dedicated missingness indicator.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

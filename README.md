# Crop Yield Prediction with Artificial Neural Networks

Tabular regression on a smart-agriculture dataset (500 farms, 22 columns) to predict crop **yield** from soil, weather, farming practice, and vegetation features. The project covers data cleaning, leakage-safe preprocessing, a baseline ANN, a deeper ANN with early stopping, and a benchmark against classical machine learning models.

**Key result:** none of the models outperformed a simple "predict the mean" baseline (all test R² ≤ 0). Increasing model complexity brought no meaningful improvement.

## Key Objectives

- **EDA & Data Cleaning:** Detect and fix data quality issues (inconsistent formats, empty strings, impossible values).
- **Leakage-Safe Preprocessing:** Split first, then fit imputers, encoders, and scalers on training data only.
- **Baseline ANN:** A small two-hidden-layer network.
- **Modified ANN:** A deeper network with early stopping.
- **Benchmarking:** Compare the ANNs against Linear Regression, SVR, Random Forest, Gradient Boosting, and XGBoost to see whether the limitation is specific to neural networks.
- **Evaluation:** R², RMSE, MAE, and MAPE on a held-out test set.

## Dataset

500 rows and 22 columns. Target: `yield`.

| Type | Columns |
|------|---------|
| Categorical | `region` (5), `crop_type` (5), `fertilizer_type`, `irrigation_type`, `crop_disease_status` |
| Numerical | soil moisture, soil pH, temperature, rainfall, humidity, sunlight hours, pesticide usage, latitude, longitude, NDVI index, total days |
| Dates | `sowing_date`, `harvest_date` (converted to `total_days`) |
| Dropped identifiers | `farm_id`, `sensor_id`, `timestamp` |

**Data quality issues found and handled**

| Issue | Handling |
|-------|----------|
| `soil_pH` stored as text (some values use a comma as decimal, e.g. `6,42`) | Replace comma with dot, convert to numeric |
| `latitude` stored as text (2 empty strings) | Convert to missing values, then impute |
| `total_days` has a negative value (1 row) | Set to missing, recompute from `harvest_date - sowing_date` |
| Missing values in `sunlight_hours`, `irrigation_type`, `crop_disease_status` | Impute after splitting (see below) |

The correlation between every numerical feature and `yield` is below 0.1 (absolute value).

## Preprocessing Pipeline

- **Split:** 70% train / 10% validation / 20% test (350 / 50 / 100 rows).
- **Imputation (fitted on training data only):**
  - `sunlight_hours`: median
  - `latitude`: median per region (global median as fallback)
  - `irrigation_type`, `crop_disease_status`: mode
- **Encoding:**
  - One-Hot for nominal features (`region`, `crop_type`, `fertilizer_type`, `irrigation_type`)
  - Ordinal for `crop_disease_status` (None < Mild < Moderate < Severe)
- **Scaling:** `StandardScaler` for numerical features
- Final input dimension: 28 features

## Models

### 1. Baseline ANN
- Dense 84 (ReLU) → Dense 56 (ReLU) → Dense 1 (linear)
- 7,253 parameters, 90 epochs, batch size 32

### 2. Modified ANN
- Dense 128 (ReLU) → Dense 64 (ReLU) → Dense 32 (ReLU) → Dense 1
- Early stopping on validation loss (patience 15, best weights restored), up to 90 epochs

Both use Adam (learning rate 1e-3) and MSE loss.

## Results

Test set (100 samples):

| Model | R² | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| Baseline ANN | -0.1428 | 12,562.88 | 10,484.81 | 29.09% |
| **Modified ANN** | **-0.1358** | **12,524.63** | 10,630.14 | **28.34%** |

**Benchmark against classical ML**

| Model | R² | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| Linear Regression | -0.0853 | 12,243.02 | 10,934.44 | 31.40% |
| SVR (RBF) | -0.0042 | 11,776.28 | 10,514.08 | 30.54% |
| Random Forest | -0.0725 | 12,170.17 | 10,655.09 | 30.13% |
| Gradient Boosting | -0.1747 | 12,737.28 | 10,932.46 | 30.20% |
| XGBoost | -0.3515 | 13,661.89 | 11,490.01 | 31.97% |

## Key Findings

- **Baseline ANN:** training and validation loss converged together, so there was no overfitting. But R² was negative on train (-0.13), validation (-0.26), and test (-0.14), which indicates the model could not capture meaningful patterns.
- **Modified ANN:** a deeper network with early stopping changed almost nothing. R², RMSE, and MAPE improved only marginally, and MAE got slightly worse. The improvement is not meaningful in practice.
- **Benchmark:** SVR, Linear Regression, and Random Forest scored slightly better than the ANNs on RMSE, while Gradient Boosting and XGBoost were worse. All models have negative R², so no model learned a useful relationship.
- **Possible explanation:** the numerical features have very weak correlation with yield (all below 0.1), and models of very different types performed similarly. This suggests the available features carry limited predictive information, although the small sample size (500 rows) may also contribute.

## Conclusion

This project walked through a complete tabular regression workflow: cleaning inconsistent data, preprocessing without data leakage, building and extending an ANN, and benchmarking it against classical machine learning models. The modified ANN, with more hidden layers and early stopping, was selected as the final model because it was marginally better than the baseline on R², RMSE, and MAPE.

At the same time, the results show that the predictive performance is limited overall. All models, including the ANNs and the classical benchmarks, reached a test R² at or below zero, meaning none could explain yield better than simply predicting the average. Because models of very different types behaved similarly, the results point to limited signal in the available features (all correlations with yield are below 0.1) rather than to a lack of model capacity, although this was not tested directly and the small sample size (500 rows) may also play a role.

Future work could explore more informative features, additional data, or feature selection to see whether performance can be improved.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Machine Learning:** scikit-learn, XGBoost
- **Data Handling & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Environment:** Kaggle Notebooks
dataset (Parquet file).
2. Run all cells in order. The notebook cleans the data, trains both ANNs, and prints the benchmark comparison.

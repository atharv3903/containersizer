# Stage 2 model report (generated)

Training: 245 steady training configurations, excluding configs whose median repeat is a stall (grouped 5-fold CV by parameter tuple, 40 random-search candidates per model, target log(rt)). Fit time 16 s.

Leaked validation configs excluded from the clean sets: validation_0005, validation_0006, validation_0009, validation_0014, validation_0032, validation_0047

## Clean steady validation (n = 41), 95% bootstrap CIs

| Model | n | R² | RMSE (ms) | MAPE | P90-APE | R² (log) |
|---|---|---|---|---|---|---|
| svr | 41 | 0.982 [0.970, 0.998] | 6.72 [1.72, 10.23] | 4.5% | 9.8% [5.1%, 15.4%] | 0.993 |
| xgboost | 41 | 0.997 [0.987, 1.000] | 2.75 [0.57, 4.65] | 2.5% | 3.3% [2.1%, 6.3%] | 0.992 |
| random_forest | 41 | 0.997 [0.988, 1.000] | 2.74 [0.71, 4.52] | 2.1% | 2.0% [1.5%, 3.8%] | 0.993 |
| mlp | 41 | 0.860 [0.764, 0.983] | 18.70 [3.78, 28.36] | 12.0% | 33.4% [17.3%, 41.1%] | 0.972 |
| ridge | 41 | 0.803 [0.742, 0.955] | 22.16 [6.51, 32.32] | 12.0% | 16.0% [14.2%, 44.3%] | 0.961 |

## Same metrics on all 46 steady validation configs (includes the 5 leaked ones; the original evaluation's set)

| Model | n | R² | RMSE (ms) | MAPE | P90-APE | R² (log) |
|---|---|---|---|---|---|---|
| svr | 46 | 0.983 | 6.42 | 4.3% | 8.9% | 0.994 |
| xgboost | 46 | 0.997 | 2.69 | 2.4% | 3.4% | 0.992 |
| random_forest | 46 | 0.997 | 2.68 | 2.0% | 2.1% | 0.994 |
| mlp | 46 | 0.872 | 17.70 | 11.4% | 33.0% | 0.974 |
| ridge | 46 | 0.816 | 21.17 | 11.7% | 16.7% | 0.964 |

## Clean validation, every label (n = 44, includes 3 non-steady)

| Model | n | R² | RMSE (ms) | MAPE | P90-APE | R² (log) |
|---|---|---|---|---|---|---|
| svr | 44 | -0.058 | 851.65 | 11.0% | 15.2% | 0.388 |
| xgboost | 44 | -0.061 | 853.05 | 9.1% | 5.6% | 0.320 |
| random_forest | 44 | -0.062 | 853.41 | 8.7% | 3.7% | 0.303 |
| mlp | 44 | -0.056 | 851.13 | 17.9% | 39.5% | 0.431 |
| ridge | 44 | -0.062 | 853.29 | 17.9% | 43.9% | 0.285 |

## Grouped CV out-of-fold on training (n = 245)

| Model | n | R² | RMSE (ms) | MAPE | P90-APE | R² (log) |
|---|---|---|---|---|---|---|
| svr | 245 | 0.981 | 4.90 | 4.3% | 8.5% | 0.996 |
| xgboost | 245 | 1.000 | 0.75 | 1.7% | 3.3% | 0.999 |
| random_forest | 245 | 0.999 | 1.11 | 1.8% | 3.8% | 0.999 |
| mlp | 245 | 0.908 | 10.87 | 15.9% | 32.1% | 0.948 |
| ridge | 245 | 0.877 | 12.62 | 9.4% | 15.2% | 0.980 |

## Sensitivity: training WITH the 1 config(s) whose median is a stall (clean steady validation)

| Model | n | R² | RMSE (ms) | MAPE | P90-APE | R² (log) |
|---|---|---|---|---|---|---|
| svr | 41 | 0.983 | 6.43 | 5.4% | 13.7% | 0.992 |
| xgboost | 41 | 0.958 | 10.24 | 6.1% | 21.1% | 0.986 |
| random_forest | 41 | 0.997 | 2.88 | 4.4% | 3.4% | 0.985 |
| mlp | 41 | 0.966 | 9.26 | 13.1% | 24.7% | 0.971 |
| ridge | 41 | 0.773 | 23.79 | 11.1% | 20.7% | 0.963 |

Best by P90-APE on clean steady validation: **random_forest**

## Best hyperparameters

- svr: `{"svr__C": 243.0159, "svr__epsilon": 0.05404, "svr__gamma": 0.0078}`
- xgboost: `{"colsample_bytree": 0.7878, "learning_rate": 0.04716, "max_depth": 2, "min_child_weight": 3, "n_estimators": 264, "reg_lambda": 0.00121, "subsample": 0.90335}`
- random_forest: `{"max_depth": 6, "max_features": 1.0, "min_samples_leaf": 1, "n_estimators": 188}`
- mlp: `{"mlpregressor__activation": "relu", "mlpregressor__alpha": 0.0214, "mlpregressor__hidden_layer_sizes": [16], "mlpregressor__learning_rate_init": 0.00906}`
- ridge: `{"ridge__alpha": 0.00014}`

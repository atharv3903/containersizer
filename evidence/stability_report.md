# Stage 3 stability report (generated)

CPU calibration (steady no-write training configs): demand = 3.33 ms + 24.4 ns × cpuLoad. Thresholds: θ_cpu = 1.0 (theory), θ_io = 0.60 (lowest disk utilisation at which a training config was not steady).

Training: 250 configs, 4 not steady. Held-out: 66 validation + grid configs whose tuple is not in training, 4 not steady.

## Held-out configurations

| Method | caught (TP) | missed (FN) | false alarms (FP) | correct safe (TN) | recall | false-alarm rate | precision |
|---|---|---|---|---|---|---|---|
| gate | 4 | 0 | 2 | 60 | 100% | 3.2% | 67% |
| logistic | 3 | 1 | 1 | 61 | 75% | 1.6% | 75% |
| random_forest_clf | 1 | 3 | 0 | 62 | 25% | 0.0% | 100% |

## Grouped 5-fold CV over all 325 configurations (9 not steady)

| Method | caught (TP) | missed (FN) | false alarms (FP) | correct safe (TN) | recall | false-alarm rate | precision |
|---|---|---|---|---|---|---|---|
| gate | 9 | 0 | 21 | 295 | 100% | 6.6% | 30% |
| logistic | 7 | 2 | 15 | 301 | 78% | 4.7% | 32% |
| random_forest_clf | 2 | 7 | 2 | 314 | 22% | 0.6% | 50% |

## Every non-steady configuration and the gate's view (fitted on training only)

| config | label | rt ms | completed/offered | ρ_cpu | ρ_io | margin (≥ 1 = unsafe) |
|---|---|---|---|---|---|---|
| train_0036 | near_saturation | 787 | 0.93 | 0.28 | 1.00 | 1.67 |
| train_0050 | overload | 1390 | 0.76 | 0.35 | 1.20 | 2.00 |
| train_0166 | near_saturation | 1302 | 0.81 | 0.35 | 1.20 | 2.00 |
| train_0207 | near_saturation | 774 | 0.86 | 0.42 | 0.60 | 1.00 |
| validation_0014 | near_saturation | 456 | 0.93 | 0.09 | 0.60 | 1.00 |
| validation_0018 | overload | 1623 | 0.79 | 0.55 | 1.00 | 1.67 |
| validation_0019 | overload | 2143 | 0.78 | 0.09 | 1.20 | 2.00 |
| validation_0044 | overload | 5019 | 0.60 | 0.42 | 1.20 | 2.00 |
| grid_slice_0025 | overload | 19578 | 0.28 | 1.25 | 0.00 | 1.25 |

## Gate false alarms over all configurations: 21 steady configs with margin ≥ 1

33% of them had at least one stalled repeat, against 0.3% of the other steady configurations: the gate's 'false alarms' are where steady state is fragile.

| config | rt ms | ρ_cpu | ρ_io | margin | stalled repeats |
|---|---|---|---|---|---|
| train_0002 | 6.5 | 0.06 | 1.00 | 1.67 | 0 |
| train_0009 | 28.7 | 0.55 | 1.00 | 1.67 | 1 |
| train_0022 | 16.9 | 0.31 | 1.00 | 1.67 | 0 |
| train_0041 | 16.8 | 0.08 | 1.00 | 1.67 | 0 |
| train_0046 | 28.5 | 0.55 | 1.00 | 1.67 | 0 |
| train_0061 | 25.7 | 0.42 | 0.60 | 1.00 | 0 |
| train_0071 | 28.5 | 0.28 | 1.00 | 1.67 | 0 |
| train_0117 | 17.7 | 0.04 | 1.00 | 1.67 | 0 |
| train_0119 | 194.0 | 0.09 | 0.60 | 1.00 | 2 |
| train_0121 | 20.5 | 0.15 | 1.00 | 1.67 | 1 |
| train_0130 | 124.4 | 0.31 | 1.00 | 1.67 | 0 |
| train_0173 | 31.1 | 0.07 | 1.00 | 1.67 | 0 |
| train_0191 | 16.8 | 0.62 | 1.00 | 1.67 | 0 |
| train_0205 | 5.5 | 0.58 | 1.00 | 1.67 | 1 |
| train_0206 | 29.1 | 0.28 | 1.00 | 1.67 | 0 |
| train_0225 | 28.6 | 0.28 | 1.00 | 1.67 | 1 |
| train_0229 | 20.1 | 0.31 | 1.00 | 1.67 | 1 |
| train_0245 | 31.2 | 0.55 | 1.00 | 1.67 | 0 |
| validation_0031 | 28.8 | 0.14 | 1.00 | 1.67 | 0 |
| validation_0037 | 34.9 | 0.08 | 1.00 | 1.67 | 1 |
| grid_slice_0021 | 25.0 | 0.15 | 1.00 | 1.67 | 0 |

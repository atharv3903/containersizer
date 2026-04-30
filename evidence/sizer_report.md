# Stage 4 sizer report (generated)

Every prediction below is out of fold: grouped, stratified 5-fold CV over all 325 measured configurations (gate + random forest refitted per fold). Ground truth = the measurement (steady and mean response time ≤ SLA). SLAs tested: 10, 20, 30, 50, 100, 200, 500 ms. Guard band (final model) = +3.8% (P90 out-of-fold relative error).

## 1. Per-configuration decisions (325 configurations × 7 SLAs = 2,275 decisions)

| Method | accepted | bad accepts (would break the SLA) | needless rejects | truly OK |
|---|---|---|---|---|
| regressor_only | 1829 | 60 (3.3%) | 2 (0.1%) | 1771 |
| sizer | 1647 | 2 (0.1%) | 126 (7.1%) | 1771 |

Per SLA:

| SLA ms | method | accepted | bad accepts | needless rejects | truly OK |
|---|---|---|---|---|---|
| 10 | sizer | 87 | 0 | 2 | 89 |
| 10 | regressor_only | 94 | 5 | 0 | 89 |
| 20 | sizer | 191 | 1 | 6 | 196 |
| 20 | regressor_only | 200 | 6 | 2 | 196 |
| 30 | sizer | 245 | 1 | 38 | 282 |
| 30 | regressor_only | 295 | 13 | 0 | 282 |
| 50 | sizer | 267 | 0 | 19 | 286 |
| 50 | regressor_only | 295 | 9 | 0 | 286 |
| 100 | sizer | 267 | 0 | 19 | 286 |
| 100 | regressor_only | 295 | 9 | 0 | 286 |
| 200 | sizer | 295 | 0 | 21 | 316 |
| 200 | regressor_only | 325 | 9 | 0 | 316 |
| 500 | sizer | 295 | 0 | 21 | 316 |
| 500 | regressor_only | 325 | 9 | 0 | 316 |

## 2. Sizing decisions on measured workloads

For each workload measured under 2+ containers and each SLA that at least one measured container meets: the method picks the cheapest container it believes meets the SLA, and the pick is checked against its measurement.

| Method | cases | pick meets SLA | pick violates SLA | no pick | cheapest correct pick | mean extra cost when met |
|---|---|---|---|---|---|---|
| regressor_only | 268 | 257 (95.9%) | 11 (4.1%) | 0 | 257 (95.9%) | 0.0% |
| sizer | 268 | 252 (94.0%) | 0 (0.0%) | 16 | 227 (84.7%) | 2.3% |

## 3. Example recommendations (sizer fitted on the 250 training configurations)

| Workload | SLA | Recommendation | Measured containers for this workload |
|---|---|---|---|
| 10 req/s, size 50, cpuLoad 5,000,000 | 300 ms | 2 CPU, 512 MiB, 10 MB/s (predicted 121.8 ms, bound 126.5 ms, margin 0.63) | 1 CPU/512/20: 19578 ms overload |
| 30 req/s, size 200, cpuLoad 100,000 | 50 ms | no feasible container | 0.5 CPU/1024/50: 1390 ms overload; 0.5 CPU/1024/50: 1302 ms near_saturation; 2 CPU/512/50: 2143 ms overload |
| 20 req/s, size 100, cpuLoad 1,000,000 | 50 ms | 1 CPU, 512 MiB, 50 MB/s (predicted 29.0 ms, bound 30.1 ms, margin 0.67) | 2 CPU/1024/20: 28 ms steady; 2 CPU/512/20: 29 ms steady; 1 CPU/512/20: 1623 ms overload |
| 5 req/s, size 10, cpuLoad 500,000 | 25 ms | 0.5 CPU, 512 MiB, 10 MB/s (predicted 16.6 ms, bound 17.2 ms, margin 0.16) | 1 CPU/1024/20: 17 ms steady; 0.5 CPU/1024/50: 16 ms steady; 0.5 CPU/1024/50: 17 ms steady; 2 CPU/512/50: 16 ms steady; 0.5 CPU/1024/50: 16 ms steady; 1 CPU/512/20: 16 ms steady; 1 CPU/512/20: 16 ms steady |
| 50 req/s, size 10, cpuLoad 100,000 | 10 ms | 0.5 CPU, 512 MiB, 10 MB/s (predicted 5.2 ms, bound 5.4 ms, margin 0.58) | 2 CPU/512/10: 5 ms steady; 0.5 CPU/1024/20: 5 ms steady; 1 CPU/1024/10: 5 ms steady; 0.5 CPU/1024/50: 5 ms steady; 2 CPU/512/50: 5 ms steady |

## 4. Capacity planning: max req/s within a 50 ms SLA (512 MiB, 20 MB/s, no writes)

| cpus | cpuLoad 100k | 500k | 1M |
|---|---|---|---|
| 0.5 | 50 | 32 | 18 |
| 1 | 50 | 50 | 36 |
| 2 | 50 | 50 | 50 |

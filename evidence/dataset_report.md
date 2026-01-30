# Dataset report (generated)

Generated 2026-11-20T18:05:49Z by `csizer report` 0.1.0 (Python 3.12.3, pandas 2.2.3). Do not edit by hand.

Data: `data/v1`: 975 runs, 325 configurations.

## Checks

| Status | Check | Result |
|---|---|---|
| OK | `manifest` | all 990 files match their SHA-256 |
| OK | `parts_union` | raw_results.csv = train + validation + grid_slices (975 runs) |
| OK | `configs_match` | every planned configuration was run, in file order |
| OK | `repeats` | all 325 configurations have repeats 1, 2, 3 |
| OK | `on_grid` | every parameter value is one of the design levels |
| OK | `logs_agree` | all 975 CSV rows equal their raw log on 7 fields and the request URI |
| WARN | `errors` | 2 of 975 runs had errors (1 answered nothing); the configuration is labelled overload and the dead repeat is excluded from the median |
| WARN | `repeat_agreement` | repeats agree closely for most configs (median CV 0.4%); 17 stall repeat(s) in 13 configs; 4 configs have 2+ stalls so their median is a stall |
| WARN | `duplicates` | 325 configurations, 291 distinct parameter tuples; 28 tuples measured more than once; 6 validation configs also appear in training |
| INFO | `labels` | stability labels: steady 316, overload 5, near_saturation 4 |
| INFO | `zero_write_sizes` | size 10/25/50 write nothing (PHP loop bound < 1): grid-slice RT 16.3-16.5 ms (spread 0.2 ms); size 100 (1 MB) 16.9 ms; size 200 (2 MB) 25.0 ms |
| INFO | `one_at_a_time_effects` | at the midpoint workload (arr 10, size 50, cpuLoad 500k) memory changes RT by 0.1 ms, device_write by 0.5 ms, cpus by 0.5 ms |
| INFO | `cpu_cost_per_rand` | one rand() call costs 24.5 ns (fit on 146 steady, no-write configs with cpus >= 1 and cpuLoad <= 1M; intercept 3.43 ms, R^2 0.9860) |
| INFO | `run_order_drift` | median residual by quarter of the training run order varies by 0.23 ms (no visible drift if < 1 ms) |
| INFO | `demo_reproducibility` | 15 later re-runs of 5 configurations (8 matching training config_ids) are within 0.4 ms of the original medians |
| INFO | `design_vs_outcome` | the design predicted 60 of 300 train+validation configs near saturation (rho > 0.85); 8 were measured non-steady. With the measured 24.5 ns per rand() and the real writes, 24 configs exceed 0.85 |

## Coverage of design levels (configurations per level)

### train

| param | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|---|
| arr | 1: 59 | 2: 47 | 5: 37 | 10: 35 | 20: 30 | 30: 17 | 40: 19 | 50: 6 |
| cpus | 0.5: 89 | 1.0: 84 | 2.0: 77 |  |  |  |  |  |
| memory | 512: 127 | 1024: 123 |  |  |  |  |  |  |
| device_write | 10: 91 | 20: 84 | 50: 75 |  |  |  |  |  |
| size | 10: 64 | 25: 47 | 50: 50 | 100: 43 | 200: 46 |  |  |  |
| cpuLoad | 100000: 78 | 500000: 74 | 1000000: 76 | 5000000: 22 |  |  |  |  |

### validation

| param | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|---|
| arr | 1: 12 | 2: 8 | 5: 6 | 10: 8 | 20: 6 | 30: 5 | 40: 5 | 50: 0 |
| cpus | 0.5: 16 | 1.0: 14 | 2.0: 20 |  |  |  |  |  |
| memory | 512: 25 | 1024: 25 |  |  |  |  |  |  |
| device_write | 10: 12 | 20: 21 | 50: 17 |  |  |  |  |  |
| size | 10: 8 | 25: 14 | 50: 12 | 100: 8 | 200: 8 |  |  |  |
| cpuLoad | 100000: 15 | 500000: 16 | 1000000: 12 | 5000000: 7 |  |  |  |  |

## Response time by stability label (configuration medians, ms)

| label | n | min | p50 | p90 | max |
|---|---|---|---|---|---|
| near_saturation | 4 | 456.4 | 780.8 | 1147.9 | 1302.5 |
| overload | 5 | 1389.5 | 2143.2 | 13754.4 | 19577.9 |
| steady | 316 | 5.1 | 16.8 | 31.1 | 194.0 |

## Details

### `manifest`

```json
{
 "files": 990
}
```

### `parts_union`

```json
{
 "runs_by_type": {
  "train": 750,
  "validation": 150,
  "grid_slice": 75
 }
}
```

### `configs_match`

```json
{
 "train": 250,
 "validation": 50,
 "grid_slice": 25
}
```

### `logs_agree`

```json
{
 "fields": [
  "reply_time_mean_ms",
  "throughput_rps",
  "reply_time_max_ms",
  "reply_time_stddev_ms",
  "errors",
  "conns_attempted",
  "arr"
 ]
}
```

### `errors`

```json
{
 "runs": [
  {
   "run": "validation_0019_rep2",
   "errors": 58,
   "replies": 392,
   "conn_refused": 0,
   "conn_reset": 58
  },
  {
   "run": "validation_0019_rep3",
   "errors": 450,
   "replies": 0,
   "conn_refused": 450,
   "conn_reset": 0
  }
 ]
}
```

### `repeat_agreement`

```json
{
 "cv_quantiles": {
  "0.5": 0.0035,
  "0.9": 0.0212,
  "0.95": 0.0804,
  "0.99": 1.5351
 },
 "stall_runs": [
  [
   "train_0009",
   3,
   125.8
  ],
  [
   "train_0036",
   3,
   1027.1
  ],
  [
   "train_0050",
   3,
   4910.9
  ],
  [
   "train_0106",
   1,
   179.4
  ],
  [
   "train_0119",
   2,
   308.7
  ],
  [
   "train_0119",
   3,
   194.0
  ],
  [
   "train_0121",
   3,
   654.6
  ],
  [
   "train_0205",
   1,
   1028.7
  ],
  [
   "train_0207",
   1,
   774.5
  ],
  [
   "train_0207",
   3,
   1754.7
  ],
  [
   "train_0225",
   3,
   919.0
  ],
  [
   "train_0229",
   3,
   763.0
  ],
  [
   "validation_0014",
   2,
   456.4
  ],
  [
   "validation_0014",
   3,
   680.5
  ],
  [
   "validation_0018",
   1,
   1622.9
  ],
  [
   "validation_0018",
   3,
   2538.0
  ],
  [
   "validation_0037",
   1,
   411.6
  ]
 ],
 "stall_repeat_position": {
  "1": 5,
  "2": 2,
  "3": 10
 },
 "stall_configs_by_writes_per_request": {
  "1": 7,
  "2": 6
 },
 "configs_with_2plus_stalls": [
  "train_0119",
  "train_0207",
  "validation_0014",
  "validation_0018"
 ]
}
```

### `duplicates`

```json
{
 "validation_in_train": [
  "validation_0005",
  "validation_0006",
  "validation_0009",
  "validation_0014",
  "validation_0032",
  "validation_0047"
 ],
 "duplicate_tuples": [
  [
   "1",
   "0.5",
   "512",
   "50",
   "200",
   "1000000",
   "['train_0045', 'train_0110']"
  ],
  [
   "1",
   "1.0",
   "512",
   "10",
   "10",
   "500000",
   "['train_0028', 'train_0107']"
  ],
  [
   "1",
   "1.0",
   "512",
   "10",
   "100",
   "500000",
   "['train_0052', 'train_0147']"
  ],
  [
   "1",
   "1.0",
   "512",
   "20",
   "50",
   "500000",
   "['train_0105', 'grid_slice_0001']"
  ],
  [
   "1",
   "2.0",
   "1024",
   "50",
   "200",
   "500000",
   "['validation_0020', 'validation_0028']"
  ],
  [
   "2",
   "0.5",
   "512",
   "20",
   "25",
   "500000",
   "['train_0076', 'validation_0047']"
  ],
  [
   "2",
   "1.0",
   "1024",
   "10",
   "10",
   "100000",
   "['train_0081', 'train_0247']"
  ],
  [
   "2",
   "1.0",
   "1024",
   "10",
   "10",
   "1000000",
   "['train_0108', 'train_0227']"
  ],
  [
   "2",
   "2.0",
   "1024",
   "20",
   "100",
   "5000000",
   "['train_0013', 'validation_0032']"
  ],
  [
   "5",
   "0.5",
   "1024",
   "20",
   "200",
   "500000",
   "['train_0155', 'train_0200']"
  ],
  [
   "5",
   "0.5",
   "1024",
   "50",
   "25",
   "500000",
   "['train_0058', 'validation_0009']"
  ],
  [
   "5",
   "0.5",
   "1024",
   "50",
   "200",
   "500000",
   "['train_0021', 'train_0216', 'validation_0005']"
  ],
  [
   "5",
   "1.0",
   "512",
   "20",
   "50",
   "500000",
   "['validation_0026', 'grid_slice_0003']"
  ],
  [
   "10",
   "0.5",
   "512",
   "10",
   "100",
   "1000000",
   "['train_0009', 'train_0046']"
  ],
  [
   "10",
   "0.5",
   "512",
   "50",
   "100",
   "500000",
   "['train_0055', 'train_0142']"
  ],
  [
   "10",
   "1.0",
   "512",
   "20",
   "10",
   "500000",
   "['train_0152', 'grid_slice_0017']"
  ],
  [
   "10",
   "1.0",
   "512",
   "20",
   "50",
   "500000",
   "['grid_slice_0004', 'grid_slice_0010', 'grid_slice_0012', 'grid_slice_0015', 'grid_slice_0019', 'grid_slice_0023']"
  ],
  [
   "10",
   "1.0",
   "512",
   "20",
   "200",
   "500000",
   "['train_0121', 'grid_slice_0021']"
  ],
  [
   "20",
   "0.5",
   "512",
   "20",
   "50",
   "100000",
   "['train_0027', 'train_0080']"
  ],
  [
   "20",
   "1.0",
   "1024",
   "10",
   "50",
   "500000",
   "['validation_0017', 'validation_0036']"
  ],
  [
   "20",
   "2.0",
   "512",
   "20",
   "50",
   "100000",
   "['train_0137', 'train_0208']"
  ],
  [
   "30",
   "0.5",
   "1024",
   "50",
   "200",
   "100000",
   "['train_0050', 'train_0166']"
  ],
  [
   "30",
   "1.0",
   "1024",
   "50",
   "10",
   "100000",
   "['train_0123', 'train_0212']"
  ],
  [
   "30",
   "2.0",
   "512",
   "50",
   "100",
   "100000",
   "['train_0119', 'validation_0014']"
  ],
  [
   "30",
   "2.0",
   "1024",
   "50",
   "10",
   "1000000",
   "['train_0006', 'train_0082']"
  ],
  [
   "40",
   "0.5",
   "1024",
   "10",
   "25",
   "100000",
   "['train_0068', 'validation_0006']"
  ],
  [
   "40",
   "1.0",
   "1024",
   "10",
   "25",
   "100000",
   "['train_0015', 'train_0159']"
  ],
  [
   "40",
   "2.0",
   "512",
   "50",
   "50",
   "500000",
   "['train_0005', 'train_0101', 'train_0177']"
  ]
 ]
}
```

### `labels`

```json
{
 "by_type": {
  "grid_slice": {
   "near_saturation": 0,
   "overload": 1,
   "steady": 24
  },
  "train": {
   "near_saturation": 3,
   "overload": 1,
   "steady": 246
  },
  "validation": {
   "near_saturation": 1,
   "overload": 3,
   "steady": 46
  }
 },
 "non_steady": [
  [
   "train_0036",
   "near_saturation",
   0.93,
   787.1,
   0
  ],
  [
   "train_0050",
   "overload",
   0.763,
   1389.5,
   0
  ],
  [
   "train_0166",
   "near_saturation",
   0.813,
   1302.5,
   0
  ],
  [
   "train_0207",
   "near_saturation",
   0.86,
   774.5,
   0
  ],
  [
   "validation_0014",
   "near_saturation",
   0.933,
   456.4,
   0
  ],
  [
   "validation_0018",
   "overload",
   0.785,
   1622.9,
   0
  ],
  [
   "validation_0019",
   "overload",
   0.778,
   2143.15,
   1
  ],
  [
   "validation_0044",
   "overload",
   0.597,
   5019.2,
   0
  ],
  [
   "grid_slice_0025",
   "overload",
   0.28,
   19577.9,
   0
  ]
 ]
}
```

### `zero_write_sizes`

```json
{
 "grid_rt_by_size": {
  "10": 16.3,
  "25": 16.5,
  "50": 16.4,
  "100": 16.9,
  "200": 25.0
 }
}
```

### `one_at_a_time_effects`

```json
{
 "grid_rt": {
  "memory": {
   "512": 16.7,
   "1024": 16.6
  },
  "device_write": {
   "10": 16.5,
   "20": 16.5,
   "50": 17.0
  },
  "cpus": {
   "0.5": 16.8,
   "1.0": 16.8,
   "2.0": 16.3
  },
  "arr": {
   "1": 16.7,
   "2": 16.5,
   "5": 16.4,
   "10": 16.3,
   "20": 16.5,
   "30": 16.0,
   "40": 14.1,
   "50": 13.3
  },
  "cpuLoad": {
   "100000": 5.8,
   "500000": 16.8,
   "1000000": 28.2,
   "5000000": 19577.9
  }
 }
}
```

### `cpu_cost_per_rand`

```json
{
 "ns_per_rand": 24.487705799238242,
 "intercept_ms": 3.4294684543555713,
 "r2": 0.9859683810175854,
 "n": 146
}
```

### `run_order_drift`

```json
{
 "median_residual_ms_by_quarter": {
  "Q1": 0.57,
  "Q2": 0.52,
  "Q3": 0.49,
  "Q4": 0.34
 },
 "n": 161
}
```

### `demo_reproducibility`

```json
{
 "rows": [
  [
   "demo_0001",
   "train_0001",
   5.8,
   5.7,
   0.1
  ],
  [
   "demo_0002",
   "train_0002",
   6.7,
   6.5,
   0.2
  ],
  [
   "demo_0003",
   "train_0003",
   16.1,
   16.2,
   -0.1
  ],
  [
   "demo_0004",
   "train_0006",
   25.0,
   25.0,
   0.0
  ],
  [
   "demo_0004",
   "train_0082",
   25.0,
   25.0,
   0.0
  ],
  [
   "demo_0005",
   "train_0005",
   14.5,
   14.3,
   0.2
  ],
  [
   "demo_0005",
   "train_0101",
   14.5,
   14.1,
   0.4
  ],
  [
   "demo_0005",
   "train_0177",
   14.5,
   14.6,
   -0.1
  ],
  [
   "demo_0001",
   "train_0001",
   5.7,
   5.7,
   0.0
  ],
  [
   "demo_0002",
   "train_0002",
   6.7,
   6.5,
   0.2
  ],
  [
   "demo_0003",
   "train_0003",
   16.1,
   16.2,
   -0.1
  ],
  [
   "demo_0004",
   "train_0006",
   25.0,
   25.0,
   0.0
  ],
  [
   "demo_0004",
   "train_0082",
   25.0,
   25.0,
   0.0
  ],
  [
   "demo_0005",
   "train_0005",
   14.3,
   14.3,
   0.0
  ],
  [
   "demo_0005",
   "train_0101",
   14.3,
   14.1,
   0.2
  ],
  [
   "demo_0005",
   "train_0177",
   14.3,
   14.6,
   -0.3
  ],
  [
   "demo_0001",
   "train_0001",
   5.7,
   5.7,
   0.0
  ],
  [
   "demo_0002",
   "train_0002",
   6.7,
   6.5,
   0.2
  ],
  [
   "demo_0003",
   "train_0003",
   16.3,
   16.2,
   0.1
  ],
  [
   "demo_0004",
   "train_0006",
   24.8,
   25.0,
   -0.2
  ],
  [
   "demo_0004",
   "train_0082",
   24.8,
   25.0,
   -0.2
  ],
  [
   "demo_0005",
   "train_0005",
   14.3,
   14.3,
   0.0
  ],
  [
   "demo_0005",
   "train_0101",
   14.3,
   14.1,
   0.2
  ],
  [
   "demo_0005",
   "train_0177",
   14.3,
   14.6,
   -0.3
  ]
 ]
}
```

### `design_vs_outcome`

```json
{
 "planned_near_saturation": 60,
 "measured_non_steady": 8,
 "corrected_rho_above_soft": 24,
 "non_steady_with_rho": [
  [
   "train_0036",
   "near_saturation",
   1.0,
   1.0
  ],
  [
   "train_0050",
   "overload",
   1.2,
   1.2
  ],
  [
   "train_0166",
   "near_saturation",
   1.2,
   1.2
  ],
  [
   "train_0207",
   "near_saturation",
   0.75,
   0.6
  ],
  [
   "validation_0014",
   "near_saturation",
   0.6,
   0.6
  ],
  [
   "validation_0018",
   "overload",
   1.0,
   1.0
  ],
  [
   "validation_0019",
   "overload",
   1.2,
   1.2
  ],
  [
   "validation_0044",
   "overload",
   1.2,
   1.2
  ]
 ],
 "rho_design_quantiles": [
  0.25,
  1.0,
  1.2
 ],
 "rho_corrected_quantiles": [
  0.122,
  0.5,
  1.2
 ]
}
```


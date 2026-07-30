# Calibration Set C — Audit Log

**Protocol:** Label-free random split of eval nodes. Labels are NEVER used to construct C.

- `use_label_clean_calibration = False`
- `rho = 0.3` (|C| / |V_eval|)

| Dataset | Seed | |C| | |T| | Anom in C | Anom in T | pi_C | pi_T |
|---------|------|-----|-----|-----------|-----------|------|------|
| cora | 0 | 292 | 684 | 13 | 31 | 0.0445 | 0.0453 |
| cora | 1 | 292 | 684 | 15 | 31 | 0.0514 | 0.0453 |
| cora | 2 | 292 | 684 | 12 | 34 | 0.0411 | 0.0497 |
| cora | 3 | 292 | 684 | 18 | 30 | 0.0616 | 0.0439 |
| cora | 4 | 292 | 684 | 8 | 39 | 0.0274 | 0.0570 |
| citeseer | 0 | 359 | 839 | 16 | 41 | 0.0446 | 0.0489 |
| citeseer | 1 | 359 | 839 | 14 | 39 | 0.0390 | 0.0465 |
| citeseer | 2 | 359 | 839 | 14 | 38 | 0.0390 | 0.0453 |
| citeseer | 3 | 359 | 839 | 15 | 31 | 0.0418 | 0.0369 |
| citeseer | 4 | 359 | 839 | 21 | 40 | 0.0585 | 0.0477 |
| pubmed | 0 | 2129 | 4970 | 96 | 270 | 0.0451 | 0.0543 |
| pubmed | 1 | 2129 | 4970 | 115 | 237 | 0.0540 | 0.0477 |
| pubmed | 2 | 2129 | 4970 | 107 | 265 | 0.0503 | 0.0533 |
| pubmed | 3 | 2129 | 4970 | 116 | 253 | 0.0545 | 0.0509 |
| pubmed | 4 | 2129 | 4970 | 111 | 262 | 0.0521 | 0.0527 |
| acm | 0 | 326 | 763 | 25 | 28 | 0.0767 | 0.0367 |
| acm | 1 | 326 | 763 | 22 | 35 | 0.0675 | 0.0459 |
| acm | 2 | 326 | 763 | 13 | 33 | 0.0399 | 0.0433 |
| acm | 3 | 326 | 763 | 18 | 38 | 0.0552 | 0.0498 |
| acm | 4 | 326 | 763 | 12 | 41 | 0.0368 | 0.0537 |
| blogcatalog | 0 | 561 | 1311 | 33 | 67 | 0.0588 | 0.0511 |
| blogcatalog | 1 | 561 | 1311 | 28 | 75 | 0.0499 | 0.0572 |
| blogcatalog | 2 | 561 | 1311 | 26 | 65 | 0.0463 | 0.0496 |
| blogcatalog | 3 | 561 | 1311 | 25 | 61 | 0.0446 | 0.0465 |
| blogcatalog | 4 | 561 | 1311 | 25 | 60 | 0.0446 | 0.0458 |
| flickr | 0 | 818 | 1909 | 45 | 99 | 0.0550 | 0.0519 |
| flickr | 1 | 818 | 1909 | 33 | 97 | 0.0403 | 0.0508 |
| flickr | 2 | 818 | 1909 | 42 | 108 | 0.0513 | 0.0566 |
| flickr | 3 | 818 | 1909 | 45 | 89 | 0.0550 | 0.0466 |
| flickr | 4 | 818 | 1909 | 48 | 87 | 0.0587 | 0.0456 |

## Code path

1. `partition.make_partition()` shuffles all node indices without labels/scores
2. Train/val/eval split by configured fractions
3. Eval nodes split: first `rho` fraction → C, remainder → T
4. Anomalies are **not** excluded unless `use_label_clean_calibration=True`

See `src/crc_gad/partition.py` lines 45–75.
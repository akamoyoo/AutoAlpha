# Factor Evaluation Workflow

## Directory Convention
- `factors/factors.example`: factor template and API reference.
- `factors/inbox`: new factors from other AI (one `.py` per factor).
- `factors/passed`: factors that pass quality + collinearity checks.
- `factors/rejected`: eliminated factors.
- `factors/reports`: JSON reports with metrics and rejection reasons.
- `factors/cache`: cached factor values for reuse in collinearity checks.

## Factor File Contract
Each factor file should define:
- `compute_factor(df: pd.DataFrame) -> pd.Series` (required)
- `FACTOR_NAME` (optional)
- `FACTOR_DESCRIPTION` (optional)
- `REQUIRED_COLUMNS` (optional)

Returned series must:
- be numeric,
- have same length as `df`,
- align with `df.index`,
- avoid future leakage.

## Evaluate One Factor
```bash
python factors/evaluate_factor.py --factor factors/inbox/<factor_file>.py
```

## Batch Evaluate All Inbox Factors
```bash
python factors/evaluate_factor.py --batch-inbox
```

Consider that any of the computation may be parallelizable.

## Default Data/Target
- Data: `Data/BTCUSDT/klines_1s_monthly_filled_all_with_logret_1h.parquet`
- Target: `log_return`

If target column is not found, evaluator will try fallback names:
- `log_return`, `logret_1h`, `target`, `y`

## Factor Cache (avoid repeated compute)
During collinearity checks, each passed factor would normally be recomputed on the full dataset.
Now evaluator can cache each factor series and reuse it in later runs.

Cache path format:
- `factors/cache/<data_fingerprint>/<target_col>/<factor_stem>__<factor_source_fingerprint>.parquet`

Cache invalidation behavior:
- Data file changed (`path + size + mtime` fingerprint changed) -> new cache namespace
- Target column changed -> different cache namespace
- Factor code changed (source hash changed) -> new cache file

Useful flags:
- `--cache-dir factors/cache`: set cache location
- `--no-cache`: disable cache read/write
- `--rebuild-cache`: force recompute and overwrite cache files


Build cache for all passed factors (recommended after large factor updates):
```bash
python factors/evaluate_factor.py --build-cache-from-passed
```

## Decision Rules (default thresholds)
- Coverage >= `0.8`
- Unique ratio >= `0.001`
- Std >= `1e-8`
- Predictive correlation gate:
  - `|pearson| >= 0.005` OR `|spearman| >= 0.005`
- Collinearity with passed factors:
  - reject if any `|corr| > 0.9`

You can override thresholds by CLI flags (see `--help`).

## Output
For every evaluation:
- factor file is moved to `passed/` or `rejected/`
- a JSON report is saved under `factors/reports/`
- report includes `cache_stats` for collinearity stage (`hits`, `misses`, `writes`)

# age_days Pipeline Update

## What changed

1. **Data preparation now guarantees `age_days` in MuData obs**
   - `scripts/build_splicevi_mudata.py` now:
     - normalizes metadata column names (`strip()`),
     - validates duplicate metadata headers,
     - requires/derives `age_days` from metadata (`age_days`, or fallback from `age_numeric`, `age_day`, `age`, `age_in_days`),
     - parses `age_days` to numeric (`float32`),
     - keeps backward compatibility by mirroring to `age_numeric` when missing.

2. **Evaluation uses `age_days` for regression by default**
   - `eval_splicevi.py` now:
     - adds `--age_target_col` (default: `age_days`),
     - runs **linear regression** (not ridge) on latent embeddings,
     - reports clearly defined metrics: **R², MAE, RMSE**,
     - supports fallback to `age_numeric` if `age_days` is unavailable.

3. **UMAP coloring includes `age_days`**
   - `eval_splicevi.py` appends `age_days` to UMAP color keys when available.
   - `scripts/run_staged_eval.sh` now prioritizes `age_days` in auto-selected UMAP keys.

4. **Split defaults and docs updated to `age_days`**
   - `scripts/create_test_split.py` default stratification age column changed from `age_numeric` to `age_days`.
   - `README.md` and `docs/DATA_SPLITTING.md` updated accordingly.

## Key code snippets

```python
# build_splicevi_mudata.py
age_days = pd.to_numeric(obs["age_days"], errors="coerce")
obs["age_days"] = age_days.astype(np.float32)
if "age_numeric" not in obs.columns:
    obs["age_numeric"] = obs["age_days"]
```

```python
# eval_splicevi.py
linreg = LinearRegression().fit(X_tr, y_tr)
y_pred = linreg.predict(X_ev)
r2_age = linreg.score(X_ev, y_ev)
mae_age = mean_absolute_error(y_ev, y_pred)
rmse_age = np.sqrt(mean_squared_error(y_ev, y_pred))
```

```python
# eval_splicevi.py CLI
parser.add_argument("--age_target_col", type=str, default="age_days", ...)
```

## How to run the updated pipeline

```bash
# Build MuData (age_days is now enforced/derived in obs)
python scripts/build_splicevi_mudata.py \
  --expr-matrix data/Tasic2018_MO_VIS_core.individual.expr.mat.txt \
  --splicing-matrix data/MO_VIS_core.individual.cass.mat.txt \
  --metadata-csvs data/MO_sample_metadata.csv data/VIS_sample_metadata.csv \
  --output-h5mu data/processed/splicevi_custom_input.h5mu
```

```bash
# Split (default age stratification now uses age_days)
python scripts/create_test_split.py \
  --train-path data/processed/splicevi_custom_input.h5mu \
  --output-train-path data/processed/splicevi_custom_input_train70.h5mu \
  --output-test-path data/processed/splicevi_custom_input_test30.h5mu \
  --stratify-age-col age_days \
  --stratify-celltype-col class
```

```bash
# Evaluate with age_days regression target
python eval_splicevi.py \
  --train_mdata_path data/processed/splicevi_custom_input_train70.h5mu \
  --test_mdata_path data/processed/splicevi_custom_input_test30.h5mu \
  --model_dir models/custom_baseline_run \
  --fig_dir logs/eval_runs/latest/figures \
  --age_target_col age_days
```

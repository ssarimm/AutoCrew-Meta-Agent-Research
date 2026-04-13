# AutoCrew Bug Fixes — Copy-Paste Guide

## Files Changed (10 files)

Replace each file in your project with the corresponding file from this folder.

### 1. `requirements.txt` (root)
- **Added:** `httpx` (needed by `groq_limits.py`)
- **Added:** `Pillow` (needed by `report_generator.py` for image sizing)

### 2. `src/main.py`
- **Fixed:** Dataset 1 description now specifies `header=None` (CSV has no header row)
- **Fixed:** Dataset 3 description now says `fillna` instead of `dropna` (preserves ~30K rows)
- **Fixed:** Dataset 2 description mentions class imbalance and `class_weight='balanced'`

### 3. `src/manual/modeling_crew.py`
- **Fixed:** Dataset-aware loading via `_build_loading_code()` helper:
  - Dataset 1: `header=None`, `replace('?')`, `dropna()`
  - Dataset 2: drops `id` column, uses `class_weight='balanced'`
  - Dataset 3: uses `fillna(median)` instead of `dropna()`, targets `SeriousDlqin2yrs`
- **Fixed:** All metrics consistently use `average='weighted'`

### 4. `src/manual/mrm_crew.py`
- **Fixed:** Same dataset-aware preprocessing as modeling_crew
- **Fixed:** Stress test uses same RF params per dataset (e.g. `class_weight='balanced'` for fraud)
- **Fixed:** No more hardcoded target column detection with `_pd.read_csv(nrows=0)`

### 5. `src/meta_agent/instruction_writer.py`
- **Fixed:** Fallback instructions are now dataset-aware (header, fillna, class_weight)
- **Fixed:** All metric prints use `average='weighted'` consistently
- **Fixed:** Added `_get_load_code()`, `_get_target_code()`, `_get_rf_params()` helpers
- **Fixed:** LLM system prompt now includes dataset-specific rules

### 6. `src/meta_agent/orchestrator.py`
- **Fixed:** `_extract_target_hint()` now uses dataset path for reliable detection
- **Fixed:** `_get_dataset_hint()` injects dataset-specific loading hints into all code tasks
- **Fixed:** Always injects `average='weighted'` reminder for training/eval tasks
- **Fixed:** Removed reliance on NL parsing alone for target column

### 7. `src/crew_factory/auto_runner.py`
- **Fixed:** Replaced hardcoded `time.sleep(60)` with configurable `PHASE_WAIT_SEC`
  - Default: 30s (was 60s × 3 = 180s wasted)
  - Set `AUTOCREW_PHASE_WAIT=0` env var to disable waits entirely
  - Set `AUTOCREW_PHASE_WAIT=60` to restore old behavior

### 8. `src/tools/code_execution.py`
- **Fixed:** Temp file `temp_script.py` is now cleaned up after execution (in `finally` block)
- **Fixed:** Added `pos_label` error hint when target has string labels like '+'/'-'

### 9. `src/tools/eda_tool.py`
- **Fixed:** Uses `header=None` for `credit_card_approval.csv`
- **Fixed:** Checks for '?' placeholder values and warns about them
- **Fixed:** Temp file `temp_eda.py` is now cleaned up after execution

### 10. `src/evaluation/generate_charts.py`
- **Fixed:** EDA charts now try to load REAL data instead of always using synthetic
- **Fixed:** Falls back to synthetic data gracefully if CSV is not found
- **Fixed:** Correct column labels (no more fake "Gender Distribution" for anonymized data)

---

## Summary of Root Causes Fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| No `header=None` for approval CSV | Loses first data row, column names are data values | Dataset-aware loading |
| `dropna()` on cs-training.csv | Drops ~30K rows (20% of data) | Use `fillna(median)` instead |
| No `class_weight` for fraud dataset | 99% accuracy by predicting all-legitimate | `class_weight='balanced'` |
| F1 `binary` vs `weighted` mismatch | Manual and Auto paths report incomparable metrics | Force `average='weighted'` everywhere |
| Hardcoded 60s × 3 sleeps | 180s wasted per Auto run | Configurable via env var, default 30s |
| Temp files not cleaned up | Leaks code on disk | `finally` block cleanup |
| Synthetic EDA charts | Misleading labels in research report | Load real data when available |
| Missing pip dependencies | `httpx` ImportError, `Pillow` ImportError | Added to requirements.txt |
| `pos_label` error not hinted | Agent gets stuck retrying with wrong params | Added error hint |

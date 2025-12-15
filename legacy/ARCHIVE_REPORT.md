# Archive & Cleanup Report

## Removed functions from active modules
- `utils/s1_global_sync.py`:
  - Removed `s1_run_npz` CLI/helper wrapper and its loader/save dependencies (unused by `test.py` or `nbar.stages.s1`).

## Archived files
- `utils/util.py` → `legacy/archived/utils/util.py` (legacy helpers and visualization utilities, unused by regression path).
- `utils/matrix.py` → `legacy/archived/utils/matrix.py` (ESR/AOCC metrics and helpers, used only by historical analysis scripts).
- `baselines/DWF.py` → `legacy/archived/baselines/DWF.py` (baseline experiment script).
- `baselines/stage1.py` → `legacy/archived/baselines/stage1.py` (legacy CLI wrapper for S1).
- `data/main.py` → `legacy/archived/data/main.py` (legacy dataset helper stub).

## Removed empty directories
- `baselines/`
- `data/`

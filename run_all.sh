#!/usr/bin/env bash
# Regenerates every file in results/ from data/specimens.csv (about 12 minutes on 4 cores).
set -euo pipefail
cd "$(dirname "$0")/code"
export TQDM_DISABLE=1
for s in 01_descriptive_and_validation_stats 02_cross_validation 03_shap_and_input_ablation \
         04_bayesian_optimization 05_failure_mode_statistics 06_configuration_level_statistics 08_extra_models; do
  echo "=== $s"; python "$s.py"
done
python 07_check_reported_values.py

#!/usr/bin/env bash
set -e

python -m src.experiments.build_samples --config configs/baseline.yaml --dataset squad
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset squad
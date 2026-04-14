#!/usr/bin/env bash
set -e

python -m src.experiments.build_samples --config configs/baseline.yaml --dataset nq
python -m src.experiments.run_baseline --config configs/baseline.yaml --dataset nq
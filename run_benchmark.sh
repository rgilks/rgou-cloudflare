#!/bin/bash
set -e

# Go to the Rust AI core directory
cd worker/rust_ai_core

echo "[run_benchmark.sh] Building Rust benchmark binary..."
cargo build --release --bin ai_benchmark

echo "[run_benchmark.sh] Running benchmark..."
./target/release/ai_benchmark ai_benchmark_config.json ai_benchmark_results.csv

echo "[run_benchmark.sh] Plotting results..."
python3 ../../ml/scripts/plot_ai_benchmark.py

# Copy the plot to the project root for convenience
cp ai_benchmark_results.png ../../ai_benchmark_results.png

echo "[run_benchmark.sh] Latest results:"
tail -n +2 ai_benchmark_results.csv | column -t -s, | tee /dev/tty

echo "[run_benchmark.sh] If you trained a new model, remember to update ml/ml_experiments_log.csv with the results above." 
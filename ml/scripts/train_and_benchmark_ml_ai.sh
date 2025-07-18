#!/bin/bash

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_version>"
  exit 1
fi

MODEL_VERSION=$1
WEIGHTS_FILE=ml/data/weights/ml_ai_weights_${MODEL_VERSION}.json

# Train the model against EMM-6
caffeinate -i python3 ml/scripts/train_ml_ai.py \
  --num-games 2000 \
  --epochs 20 \
  --dropout 0.3 \
  --use-rust-ai \
  --output $WEIGHTS_FILE \
  --rust-ai-depth 6

# Register the model for benchmarking
python3 ml/scripts/add_ml_model_to_benchmark.py ml-$MODEL_VERSION $WEIGHTS_FILE expectiminimax 6

# Run the benchmark
cd worker/rust_ai_core
./target/release/ai_benchmark
cd ../..

# Update the benchmark graph
python3 ml/scripts/plot_ai_benchmark.py

echo "All done! Model ml-$MODEL_VERSION trained, benchmarked, and graphed against EMM-6." 
#!/usr/bin/env python3
import json
import sys
import os

MODELS_PATH = os.path.join(os.path.dirname(__file__), '../..', 'worker/rust_ai_core/ml_models.json')
BENCHMARK_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../..', 'worker/rust_ai_core/ai_benchmark_config.json')

def add_model(model_name, weights_path):
    with open(MODELS_PATH, 'r+') as f:
        models = json.load(f)
        if model_name in models:
            print(f"Model {model_name} already exists in ml_models.json.")
        else:
            models[model_name] = weights_path
            f.seek(0)
            json.dump(models, f, indent=2)
            f.truncate()
            print(f"Added {model_name} to ml_models.json.")

def add_benchmark_matchup(model_name, opponent_type='expectiminimax', opponent_depth=3, games=100):
    with open(BENCHMARK_CONFIG_PATH, 'r+') as f:
        config = json.load(f)
        matchups = config.get('matchups', [])
        matchup_name = f"{model_name.upper()} vs EMM-{opponent_depth}"
        for m in matchups:
            if m['name'] == matchup_name:
                print(f"Matchup {matchup_name} already exists in ai_benchmark_config.json.")
                return
        new_matchup = {
            "name": matchup_name,
            "player1": {"type": model_name},
            "player2": {"type": opponent_type, "depth": opponent_depth},
            "games": games
        }
        matchups.append(new_matchup)
        config['matchups'] = matchups
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()
        print(f"Added matchup {matchup_name} to ai_benchmark_config.json.")

def main():
    if len(sys.argv) < 3:
        print("Usage: add_ml_model_to_benchmark.py <model_name> <weights_path> [opponent_type] [opponent_depth]")
        sys.exit(1)
    model_name = sys.argv[1]
    weights_path = sys.argv[2]
    opponent_type = sys.argv[3] if len(sys.argv) > 3 else 'expectiminimax'
    opponent_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    add_model(model_name, weights_path)
    add_benchmark_matchup(model_name, opponent_type, opponent_depth)

if __name__ == "__main__":
    main() 
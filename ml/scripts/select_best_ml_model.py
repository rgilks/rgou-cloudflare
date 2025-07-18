#!/usr/bin/env python3
import csv
import os

RESULTS_PATH = os.path.join(os.path.dirname(__file__), '../..', 'worker/rust_ai_core/ai_benchmark_results.csv')

best_model = None
best_win_rate = -1.0

with open(RESULTS_PATH, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['matchup'].startswith('ML') and row['player2'] == 'EMM-3':
            model_name = row['player1'].replace('ML-', '')
            win_rate = float(row['win_rate_p1'])
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_model = model_name

if best_model:
    print(f"Best ML model vs EMM-3: {best_model} (win rate: {best_win_rate:.3f})")
else:
    print("No ML model found in benchmark results.") 
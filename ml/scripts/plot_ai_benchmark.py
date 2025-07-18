import matplotlib.pyplot as plt
import csv

matchups = []
win_rates = []

with open('ai_benchmark_results.csv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        matchups.append(row['matchup'])
        win_rates.append(float(row['win_rate_p1']))

plt.figure(figsize=(10, 6))
plt.plot(matchups, win_rates, marker='o')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Win Rate (Player 1)')
plt.xlabel('Matchup')
plt.title('AI Benchmark: Win Rates for All Matchups')
plt.tight_layout()
plt.grid(True)
plt.show() 
import matplotlib.pyplot as plt
import csv

matchups = []
win_rates = []

with open('ai_benchmark_results.csv') as f:
    lines = f.readlines()
    reader = csv.reader(lines[1:], delimiter='\t')
    for row in reader:
        if len(row) < 4:
            continue
        matchups.append(row[0].strip())
        win_rates.append(float(row[3].strip()))

plt.figure(figsize=(10, 6))
plt.plot(matchups, win_rates, marker='o')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Win Rate (Player 1)')
plt.xlabel('Matchup')
plt.title('AI Benchmark: Win Rates for All Matchups')
plt.tight_layout()
plt.grid(True)
plt.savefig('ai_benchmark_results.png', bbox_inches='tight')
plt.show() 
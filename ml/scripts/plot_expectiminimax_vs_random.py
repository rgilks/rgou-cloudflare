import matplotlib.pyplot as plt
import csv

depths = []
win_rates = []
avg_times = []

with open('expectiminimax_vs_random.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        depths.append(int(row['depth']))
        win_rates.append(float(row['win_rate']))
        avg_times.append(float(row['avg_time_ms']))

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Depth')
ax1.set_ylabel('Win Rate', color=color)
ax1.plot(depths, win_rates, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Avg Time per Move (ms)', color=color)
ax2.plot(depths, avg_times, marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Expectiminimax vs Random: Win Rate and Time per Move by Depth')
plt.show() 
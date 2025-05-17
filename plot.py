import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("dimension_results_1.csv")
df1 = df1.sort_values("embedding_dim")
df2 = pd.read_csv("dimension_results_2.csv")
df2 = df2.sort_values("embedding_dim")

plt.figure()
plt.plot(df1["embedding_dim"], df1["best_map@50"], marker='o', linestyle='-', label='BCE')
plt.plot(df2["embedding_dim"], df2["best_map@50"], marker='o', linestyle='-', label='BPR')
plt.legend()
plt.title("MAP@50 vs Embedding Dimension")
plt.xlabel("Embedding Dimension (d)")
plt.ylabel("MAP@50")
plt.grid(True)
plt.savefig("map_curve.png")
plt.show()
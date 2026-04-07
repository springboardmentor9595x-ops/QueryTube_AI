import pandas as pd

m1 = pd.read_csv("data/minilm_metrics.csv")
m2 = pd.read_csv("data/mpnet_metrics.csv")
m3 = pd.read_csv("data/multiqa_metrics.csv")

final = pd.concat([m1, m2, m3])
print("\nFINAL MODEL COMPARISON:\n")
print(final)



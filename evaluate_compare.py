import sys
import pandas as pd

gt = pd.read_csv(sys.argv[1])
pred = pd.read_csv(sys.argv[2])

merge_cols = ["part","measure","pitch","staff","voice"]

df = gt.merge(pred, on=merge_cols)

df["correct"] = df["ground_truth"] == df["predicted_fingering"]

accuracy = df["correct"].mean()

print("Total notes:", len(df))
print("Exact match accuracy:", round(accuracy*100,2), "%")

# Confusion matrix
cm = pd.crosstab(df["ground_truth"], df["predicted_fingering"])
print("\nConfusion Matrix:\n", cm)

import pandas as pd
import os

# -----------------------------
# Step 1: Base Directory
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

print("📁 Base directory:", base_dir)

# -----------------------------
# Step 2: Function to find metrics
# -----------------------------
def find_metrics_file(folder_name):
    folder_path = os.path.join(base_dir, "..", folder_name)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ Folder not found: {folder_path}")

    for file in os.listdir(folder_path):
        if "metrics" in file.lower():
            full_path = os.path.join(folder_path, file)
            print(f"✅ Found: {full_path}")
            return pd.read_csv(full_path)

    raise FileNotFoundError(f"❌ No metrics file in {folder_path}")

# -----------------------------
# Step 3: Load all models
# -----------------------------
minilm = find_metrics_file("minilm_csv")
mpnet = find_metrics_file("mpnet_csv")
multiqa = find_metrics_file("multiqa_csv")

print("\n✅ All files loaded successfully!")

# -----------------------------
# Step 4: Add Metric
# -----------------------------
minilm["Metric"] = "Cosine"
mpnet["Metric"] = "Cosine"
multiqa["Metric"] = "Cosine"

# -----------------------------
# Step 5: Combine
# -----------------------------
comparison_df = pd.concat([minilm, mpnet, multiqa], ignore_index=True)

comparison_df = comparison_df[["Model", "Metric", "Top1", "Top3", "Top5", "AvgRank"]]

# -----------------------------
# Step 6: Display
# -----------------------------
print("\n📊 FINAL COMPARISON TABLE")
print(comparison_df)

# -----------------------------
# Step 7: Save
# -----------------------------
output_path = os.path.join(base_dir, "..", "final_comparison.csv")
comparison_df.to_csv(output_path, index=False)

print(f"\n💾 Saved at: {output_path}")

# -----------------------------
# Step 8: Best Model
# -----------------------------
best_model = comparison_df.sort_values(by="Top3", ascending=False).iloc[0]

print("\n🏆 BEST MODEL")
print(f"Model        : {best_model['Model']}")
print(f"Metric       : {best_model['Metric']}")
print(f"Top-1 Recall : {best_model['Top1']:.2f}")
print(f"Top-3 Recall : {best_model['Top3']:.2f}")
print(f"Top-5 Recall : {best_model['Top5']:.2f}")
print(f"Avg Rank     : {best_model['AvgRank']:.2f}")
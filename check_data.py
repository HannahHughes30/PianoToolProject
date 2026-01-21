"""Quick status check"""
import pandas as pd
from pathlib import Path

csv_path = Path('data/training_data/fingering_data.csv')
if not csv_path.exists():
    print("❌ No data. Run: python src/annotation/extract_all_fingering.py")
    exit()

df = pd.read_csv(csv_path)
annotated = (df['fingering'] != '') & df['fingering'].notna()

print(f"Total: {len(df)} notes")
print(f"Annotated: {annotated.sum()} ({annotated.sum()/len(df)*100:.1f}%)")
print(f"Missing: {(~annotated).sum()}")

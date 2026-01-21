"""Debug why model predicts all finger 1"""
import pandas as pd
from pathlib import Path

# Check training data
df = pd.read_csv('data/training_data/features.csv')

print("="*60)
print("TRAINING DATA ANALYSIS")
print("="*60)

# Check fingering distribution
print("\nFingering Distribution in Training Data:")
print(df['fingering'].value_counts().sort_index())

# Check prev_fingering values
print("\nPrev Fingering Distribution:")
print(df['prev_fingering'].value_counts().sort_index().head(10))

# Check next_fingering values  
print("\nNext Fingering Distribution:")
print(df['next_fingering'].value_counts().sort_index().head(10))

# Check how many notes have context
has_prev = (df['prev_fingering'] > 0).sum()
has_next = (df['next_fingering'] > 0).sum()

print(f"\nNotes with prev_fingering > 0: {has_prev}/{len(df)} ({has_prev/len(df)*100:.1f}%)")
print(f"Notes with next_fingering > 0: {has_next}/{len(df)} ({has_next/len(df)*100:.1f}%)")

# Sample some rows
print("\nSample Training Data (first 20 rows):")
print(df[['pitch', 'fingering', 'prev_fingering', 'next_fingering', 'is_step_motion']].head(20))

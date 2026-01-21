"""
Combine all training data sources
"""
import pandas as pd
from pathlib import Path

print("="*60)
print("COMBINING ALL TRAINING DATA")
print("="*60)

# Load original data
original = pd.read_csv('data/training_data/fingering_data.csv')
print(f"\n✓ Original data: {len(original)} notes")

# Load week2 batch
week2_path = Path('data/annotations/week2_batch.csv')
if week2_path.exists():
    week2 = pd.read_csv(week2_path)
    print(f"✓ Week 2 batch: {len(week2)} notes")
    
    # Combine
    combined = pd.concat([original, week2], ignore_index=True)
else:
    print("⚠️ No week2_batch.csv found")
    combined = original

# Remove duplicates (just in case)
before = len(combined)
combined = combined.drop_duplicates()
after = len(combined)
if before != after:
    print(f"✓ Removed {before - after} duplicates")

# Save combined dataset
combined.to_csv('data/training_data/fingering_data_combined.csv', index=False)

print(f"\n{'='*60}")
print(f"TOTAL TRAINING DATA: {len(combined)} notes")
print(f"{'='*60}")

# Show fingering distribution
print("\nFingering distribution:")
print(combined['fingering'].value_counts().sort_index())

print(f"\n✓ Saved to: data/training_data/fingering_data_combined.csv")

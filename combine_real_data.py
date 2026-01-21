import pandas as pd
from pathlib import Path

print("="*60)
print("COMBINING TRAINING DATA (FINGERED ONLY)")
print("="*60)

# Load original - ONLY fingered notes
original = pd.read_csv('data/training_data/fingering_data.csv')
original_fingered = original[original['fingering'].notna()].copy()
print(f"\n✓ Original fingered notes: {len(original_fingered)}")

# Load week2 batch - ONLY fingered notes
week2_path = Path('data/annotations/week2_batch.csv')
if week2_path.exists():
    week2 = pd.read_csv(week2_path)
    week2_fingered = week2[week2['fingering'] != ''].copy()
    print(f"✓ Week 2 fingered notes: {len(week2_fingered)}")
    
    combined = pd.concat([original_fingered, week2_fingered], ignore_index=True)
else:
    print("⚠️ No week2_batch.csv found")
    combined = original_fingered

# Remove duplicates
before = len(combined)
combined = combined.drop_duplicates()
after = len(combined)
if before > after:
    print(f"✓ Removed {before - after} duplicates")

combined.to_csv('data/training_data/fingering_data_combined.csv', index=False)

print(f"\n{'='*60}")
print(f"TOTAL FINGERED NOTES: {len(combined)}")
print(f"{'='*60}")

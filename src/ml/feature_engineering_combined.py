"""
Feature engineering for combined dataset
"""
import pandas as pd
import sys
sys.path.append('src/ml')
from feature_engineering_v2 import extract_features

print("="*60)
print("FEATURE ENGINEERING - COMBINED DATA")
print("="*60)

# Load combined data
df = pd.read_csv('data/training_data/fingering_data_combined.csv')
print(f"\n✓ Loaded {len(df)} notes")

# Extract features
print("\n⏳ Extracting features...")
df_features, feature_names = extract_features(df)

# Save
output_path = 'data/training_data/features_combined.csv'
df_features.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"✓ Features saved: {output_path}")
print(f"  Total samples: {len(df_features)}")
print(f"  Total features: {len(feature_names)}")
print(f"{'='*60}")

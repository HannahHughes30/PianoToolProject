"""Quick check of annotation status"""
import pandas as pd
from pathlib import Path

csv_path = Path('data/training_data/all_fingering_annotations.csv')

if not csv_path.exists():
    print("❌ No annotations found!")
    print("Run: python src/annotation/extract_all_fingering.py")
    exit()

df = pd.read_csv(csv_path)

print("="*60)
print("ANNOTATION STATUS")
print("="*60)

# Overall stats
annotated = (df['fingering'] != '') & df['fingering'].notna()
total = len(df)
ann_count = annotated.sum()
coverage = (ann_count / total * 100)

print(f"\nTotal Notes: {total}")
print(f"Annotated: {ann_count} ({coverage:.1f}%)")
print(f"Missing: {total - ann_count}")

# Per-file breakdown
print(f"\nPer-File Coverage:")
for file in df['file'].unique():
    file_df = df[df['file'] == file]
    file_ann = ((file_df['fingering'] != '') & file_df['fingering'].notna()).sum()
    file_total = len(file_df)
    file_cov = (file_ann / file_total * 100)
    
    bar = "█" * int(file_cov / 2)
    print(f"  {file[:35]:35s} {bar:50s} {file_cov:5.1f}% ({file_ann}/{file_total})")

# Fingering distribution (for annotated notes)
if ann_count > 0:
    print(f"\nFingering Distribution:")
    df_ann = df[annotated].copy()
    df_ann['fingering'] = pd.to_numeric(df_ann['fingering'], errors='coerce')
    print(df_ann['fingering'].value_counts().sort_index())

print(f"\n{'='*60}")
print(f"Ready for training!")
print(f"{'='*60}")

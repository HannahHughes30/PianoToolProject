import sys
sys.path.append('src')
sys.path.append('src/annotation')
from extract_all_fingering import extract_from_mxl
import pandas as pd
from pathlib import Path

print("="*60)
print("EXTRACTING WEEK 2 BATCH")
print("="*60)

week2_dir = Path('data/training_pieces/week2_batch')
all_notes = []

for file in week2_dir.glob('*'):
    if file.suffix in ['.mxl', '.xml']:
        notes = extract_from_mxl(file)
        all_notes.extend(notes)
    elif file.suffix == '.mscz':
        print(f"\n⚠️ Skipping {file.name} - MuseScore format (.mscz)")
        print("   Open in MuseScore and export as MusicXML first!")

if all_notes:
    df = pd.DataFrame(all_notes)
    output_path = 'data/annotations/week2_batch.csv'
    df.to_csv(output_path, index=False)
    
    total = len(df)
    fingered = sum(1 for _, row in df.iterrows() if row['fingering'])
    
    print(f"\n{'='*60}")
    print(f"✓ SAVED: {output_path}")
    print(f"  Total notes: {total}")
    print(f"  Fingered: {fingered} ({fingered/total*100:.1f}%)")
    print(f"{'='*60}")
else:
    print("\n⚠️ No notes extracted!")
    print("Make sure files are .mxl or .xml format")

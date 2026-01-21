import sys
sys.path.append('src/ml')
import pandas as pd
import joblib
from feature_engineering_v2 import extract_features
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = []
    
    for part in root.findall('.//part'):
        for measure in part.findall('measure'):
            measure_num = measure.get('number', '0')
            for note_elem in measure.findall('note'):
                if note_elem.find('rest') is not None:
                    continue
                pitch_elem = note_elem.find('pitch')
                if pitch_elem is None:
                    continue
                
                step = pitch_elem.find('step').text
                octave = pitch_elem.find('octave').text
                alter_elem = pitch_elem.find('alter')
                
                if alter_elem is not None:
                    alter = int(alter_elem.text)
                    pitch = step + ('#' if alter == 1 else 'b') + octave
                else:
                    pitch = step + octave
                
                annotations.append({
                    'file': xml_path.stem,
                    'measure': measure_num,
                    'pitch': pitch,
                    'fingering': '',
                    'part': part.get('id', 'P1')
                })
    
    return annotations

# Extract notes
notes = extract_from_xml(Path('data/test_pieces/mozart_sonfacile.xml'))
df = pd.DataFrame(notes)
print(f"Extracted {len(df)} notes from Mozart\n")

# Extract V2 features
print("Extracting V2 features...")
df_features, _ = extract_features(df)

# Drop non-numeric
text_cols = ['file', 'pitch', 'part', 'measure', 'fingering']
X = df_features.drop(columns=[col for col in text_cols if col in df_features.columns], errors='ignore')

# Try to load model - check what type it is
model_data = joblib.load('models/fingering_model_xgb_v2.pkl')
print(f"\nModel type: {type(model_data)}")

if isinstance(model_data, dict):
    print("Model is a dict with keys:", model_data.keys())
    model = model_data['model']  # Extract actual model
else:
    model = model_data

# Predict
y_pred = model.predict(X)

# Show results
print(f"\nV2 Model Predictions on Mozart:")
print(f"{'='*60}")
print("\nFirst 30 notes:")
for i in range(min(30, len(df))):
    print(f"  {df.iloc[i]['pitch']:4s} → Finger {y_pred[i]}")

print(f"\n{'='*60}")
print("Fingering distribution:")
print(pd.Series(y_pred).value_counts().sort_index())
print(f"{'='*60}")

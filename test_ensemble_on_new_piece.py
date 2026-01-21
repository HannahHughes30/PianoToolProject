"""
Test ensemble model on completely new, unseen pieces
"""
import sys
sys.path.append('src')
sys.path.append('src/ml')
import pandas as pd
import joblib
from feature_engineering_ensemble import extract_ensemble_features
import xml.etree.ElementTree as ET
from pathlib import Path

def extract_from_xml(xml_path):
    """
    Extract notes from plain XML file (not zipped .mxl)
    """
    print(f"\nProcessing: {xml_path.name}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []
    
    annotations = []
    parts = root.findall('.//part')
    
    for part_idx, part in enumerate(parts):
        part_id = part.get('id', f'P{part_idx+1}')
        
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
                
                # Check for fingering
                fingering = ''
                notations = note_elem.find('notations')
                if notations is not None:
                    technical = notations.find('technical')
                    if technical is not None:
                        fingering_elem = technical.find('fingering')
                        if fingering_elem is not None:
                            fingering = fingering_elem.text
                
                annotations.append({
                    'file': xml_path.stem,
                    'measure': measure_num,
                    'pitch': pitch,
                    'fingering': fingering,
                    'part': part_id
                })
    
    print(f"  ✓ {len(annotations)} notes")
    return annotations

def test_on_new_piece(file_path):
    """
    Predict fingering on new piece and show results
    """
    print("="*60)
    print(f"TESTING ENSEMBLE ON: {file_path}")
    print("="*60)
    
    path = Path(file_path)
    
    # Extract notes
    print("\n1. Extracting notes from piece...")
    
    if path.suffix == '.mxl':
        from annotation.extract_all_fingering import extract_from_mxl
        notes = extract_from_mxl(path)
    elif path.suffix == '.xml':
        notes = extract_from_xml(path)
    else:
        print(f"  ✗ Unsupported file type: {path.suffix}")
        return
    
    if not notes:
        print("  ✗ No notes found!")
        return
    
    df = pd.DataFrame(notes)
    print(f"   ✓ Found {len(df)} notes")
    
    # Extract ensemble features
    print("\n2. Extracting ensemble features...")
    df_features, _ = extract_ensemble_features(df)
    
    # Drop non-numeric columns
    text_cols = ['file', 'pitch', 'part', 'measure', 'fingering']
    X = df_features.drop(columns=[col for col in text_cols if col in df_features.columns], errors='ignore')
    
    print(f"   ✓ {X.shape[1]} features extracted")
    
    # Load ensemble model
    print("\n3. Loading ensemble model (97.2% accuracy)...")
    model = joblib.load('models/xgboost_ensemble.pkl')
    
    # Predict
    print("\n4. Predicting fingerings...")
    y_pred_labels = model.predict(X)
    y_pred = y_pred_labels + 1  # Convert 0-4 to 1-5
    
    # Add predictions to dataframe
    df_features['predicted_fingering'] = y_pred
    
    # Show results
    print(f"\n{'='*60}")
    print("PREDICTIONS:")
    print(f"{'='*60}")
    
    # Group by measure
    result_df = df_features[['measure', 'pitch', 'predicted_fingering']].copy()
    
    print("\nFirst 30 notes:")
    print(result_df.head(30).to_string(index=False))
    
    # Fingering distribution
    print(f"\n{'='*60}")
    print("Fingering distribution:")
    print(result_df['predicted_fingering'].value_counts().sort_index())
    
    # Save predictions
    output_path = f"data/predictions/test_{path.stem}_ensemble.csv"
    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions: {output_path}")
    
    # Check if ground truth exists
    if 'fingering' in df_features.columns:
        df_with_truth = df_features[df_features['fingering'].notna()].copy()
        df_with_truth = df_with_truth[df_with_truth['fingering'] != ''].copy()
        
        if len(df_with_truth) > 0:
            y_true = df_with_truth['fingering'].astype(int)
            y_pred_truth = df_with_truth['predicted_fingering']
            
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true, y_pred_truth)
            
            print(f"\n{'='*60}")
            print(f"🎯 VALIDATION ACCURACY: {accuracy:.1%}")
            print(f"   (on {len(y_true)} ground-truth fingerings)")
            print(f"{'='*60}")
    else:
        print("\n(No ground truth fingerings in this file)")
    
    return result_df

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_ensemble_on_new_piece.py <path_to_xml_or_mxl>")
        sys.exit(1)
    
    test_on_new_piece(sys.argv[1])

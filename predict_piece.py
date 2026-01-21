"""
FINGER A COMPLETE PIECE
Takes any PDF → predicts fingering for all notes
"""
import subprocess
from pathlib import Path
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile

def note_to_midi(pitch):
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    if '#' in pitch:
        return (int(pitch[2]) + 1) * 12 + note_map[pitch[0]] + 1
    elif 'b' in pitch:
        return (int(pitch[2]) + 1) * 12 + note_map[pitch[0]] - 1
    return (int(pitch[1]) + 1) * 12 + note_map[pitch[0]]

def extract_notes_from_mxl(mxl_path):
    """Extract all notes from MusicXML file"""
    with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
        content = zip_ref.read('score.xml')
    
    root = ET.fromstring(content)
    notes_data = []
    
    for part in root.findall('.//part'):
        part_id = part.get('id', 'P1')
        
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
                
                notes_data.append({
                    'measure': measure_num,
                    'pitch': pitch,
                    'part': part_id
                })
    
    return pd.DataFrame(notes_data)

def predict_fingering_for_piece(notes_df, model, features):
    """Predict fingering for entire piece"""
    
    # Calculate features
    notes_df['midi'] = notes_df['pitch'].apply(note_to_midi)
    
    # Group by part (separate hands)
    results = []
    
    for part_id in notes_df['part'].unique():
        part_df = notes_df[notes_df['part'] == part_id].copy().reset_index(drop=True)
        
        # Calculate intervals
        part_df['interval'] = part_df['midi'].diff().fillna(0)
        part_df['abs_interval'] = part_df['interval'].abs()
        
        # Position
        part_df['measure_position'] = part_df.groupby('measure').cumcount()
        part_df['measure_length'] = part_df.groupby('measure')['measure_position'].transform('max') + 1
        part_df['measure_position_norm'] = part_df['measure_position'] / part_df['measure_length']
        
        # Hand
        part_df['hand_encoding'] = 1 if part_id == 'P2' else 0
        
        # Pedagogy features
        part_df['is_black_key'] = part_df['pitch'].str.contains('#|b').astype(int)
        part_df['is_step_motion'] = (part_df['abs_interval'] <= 2).astype(int)
        part_df['is_large_jump'] = (part_df['abs_interval'] > 7).astype(int)
        part_df['is_high_register'] = (part_df['midi'] > 72).astype(int)
        part_df['is_low_register'] = (part_df['midi'] < 60).astype(int)
        part_df['is_ascending'] = (part_df['interval'] > 0).astype(int)
        part_df['is_descending'] = (part_df['interval'] < 0).astype(int)
        
        # First pass: predict without context
        part_df['prev_fingering'] = 0
        part_df['next_fingering'] = 0
        
        predicted_fingers = []
        
        for idx, row in part_df.iterrows():
            # Get previous fingering
            if idx > 0:
                row['prev_fingering'] = predicted_fingers[-1]
            
            feature_vector = [row[f] for f in features]
            finger = model.predict([feature_vector])[0]
            predicted_fingers.append(int(finger))
        
        part_df['predicted_fingering'] = predicted_fingers
        results.append(part_df)
    
    return pd.concat(results, ignore_index=True)

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_piece.py <pdf_file>")
        print("\nExample: python predict_piece.py data/test_sheets/piano3_cabinsong.pdf")
        exit()
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        exit()
    
    print("="*60)
    print("AUTOMATIC FINGERING PREDICTION")
    print("="*60)
    print(f"\nInput: {pdf_path.name}")
    
    # Step 1: Run OMR
    print("\nStep 1: Running OMR...")
    result = subprocess.run(
        ['python', 'src/omr/audiveris_wrapper.py', str(pdf_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("❌ OMR failed")
        print(result.stderr)
        exit()
    
    # Find output MXL file
    output_dir = Path('data/omr_output')
    mxl_files = list(output_dir.glob(f"*{pdf_path.stem}*.mxl"))
    
    if not mxl_files:
        print("❌ No MXL output found")
        exit()
    
    mxl_path = mxl_files[0]
    print(f"✓ OMR complete: {mxl_path.name}")
    
    # Step 2: Extract notes
    print("\nStep 2: Extracting notes...")
    notes_df = extract_notes_from_mxl(mxl_path)
    print(f"✓ Found {len(notes_df)} notes")
    
    # Step 3: Load model
    print("\nStep 3: Loading model...")
    model_path = Path('models/fingering_model.pkl')
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
        model = saved['model']
        feature_names = saved['features']
    print(f"✓ Model loaded (79.7% accuracy)")
    
    # Step 4: Predict fingering
    print("\nStep 4: Predicting fingering...")
    predictions = predict_fingering_for_piece(notes_df, model, feature_names)
    
    # Show results
    print(f"\n{'='*60}")
    print("PREDICTIONS")
    print(f"{'='*60}")
    
    for part_id in predictions['part'].unique():
        part_data = predictions[predictions['part'] == part_id]
        hand = "LEFT" if part_id == "P2" else "RIGHT"
        
        print(f"\n{hand} HAND ({part_id}):")
        print(f"{'Measure':<10} {'Note':<6} {'Fingering'}")
        print("-" * 30)
        
        for _, row in part_data.head(30).iterrows():
            print(f"{row['measure']:<10} {row['pitch']:<6} {row['predicted_fingering']}")
        
        if len(part_data) > 30:
            print(f"... ({len(part_data) - 30} more notes)")
    
    # Save predictions
    output_path = Path('data/predictions') / f"{pdf_path.stem}_predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    predictions.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved to: {output_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

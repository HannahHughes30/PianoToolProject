"""
FEATURE ENGINEERING WITH PEDAGOGY
- Fixed prev/next fingering to use actual previous/next notes
"""
import pandas as pd
from pathlib import Path

def note_to_midi(pitch):
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    if '#' in pitch:
        return (int(pitch[2]) + 1) * 12 + note_map[pitch[0]] + 1
    elif 'b' in pitch:
        return (int(pitch[2]) + 1) * 12 + note_map[pitch[0]] - 1
    return (int(pitch[1]) + 1) * 12 + note_map[pitch[0]]

def add_pedagogy_features(df):
    """Add piano pedagogy knowledge as features"""
    
    # Black key (avoid thumb on black keys)
    df['is_black_key'] = df['pitch'].str.contains('#|b').fillna(False).astype(int)
    
    # Step motion (common for sequential fingering)
    df['is_step_motion'] = (df['abs_interval'] <= 2).astype(int)
    
    # Large jump (usually finger 1 or 5)
    df['is_large_jump'] = (df['abs_interval'] > 7).astype(int)
    
    # Octave register (high notes use 4-5, low use 1-2)
    df['octave'] = df['midi'] // 12
    df['is_high_register'] = (df['midi'] > 72).astype(int)
    df['is_low_register'] = (df['midi'] < 60).astype(int)
    
    # Direction (ascending/descending)
    df['is_ascending'] = (df['interval'] > 0).astype(int)
    df['is_descending'] = (df['interval'] < 0).astype(int)
    
    return df

def extract_features(df):
    print("Extracting features with pedagogy rules...")
    
    df['midi'] = df['pitch'].apply(note_to_midi)
    df['fingering'] = pd.to_numeric(df['fingering'], errors='coerce')
    df['has_annotation'] = df['fingering'].between(1, 5, inclusive='both')
    
    # Intervals
    df['interval'] = df.groupby(['file', 'part'])['midi'].diff().fillna(0)
    df['abs_interval'] = df['interval'].abs()
    
    # Position
    df['measure_position'] = df.groupby(['file', 'measure']).cumcount()
    df['measure_length'] = df.groupby(['file', 'measure'])['measure_position'].transform('max') + 1
    df['measure_position_norm'] = df['measure_position'] / df['measure_length']
    
    # Hand
    df['hand_encoding'] = (df['part'] == 'P2').astype(int)
    
    # FIXED: Context fingering - shift to get ACTUAL previous/next notes
    df['prev_fingering'] = df.groupby(['file', 'part'])['fingering'].shift(1).fillna(0)
    df['next_fingering'] = df.groupby(['file', 'part'])['fingering'].shift(-1).fillna(0)
    
    # Add pedagogy features
    df = add_pedagogy_features(df)
    
    features = [
        'midi', 'interval', 'abs_interval', 'measure_position_norm',
        'hand_encoding', 'prev_fingering', 'next_fingering',
        'is_black_key', 'is_step_motion', 'is_large_jump',
        'is_high_register', 'is_low_register', 'is_ascending', 'is_descending'
    ]
    
    return df, features

if __name__ == '__main__':
    csv_path = Path('data/training_data/fingering_data.csv')
    if not csv_path.exists():
        print("❌ Run: python src/annotation/extract_all_fingering.py")
        exit()
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} notes")
    
    df_features, feature_names = extract_features(df)
    
    # Check fix worked
    print("\nSample to verify prev/next fingering:")
    sample = df_features[['pitch', 'fingering', 'prev_fingering', 'next_fingering']].head(10)
    print(sample)
    
    output_path = Path('data/training_data/features.csv')
    df_features.to_csv(output_path, index=False)
    
    print(f"\n✓ Features saved")
    print(f"\nNext: python src/ml/train_model.py")

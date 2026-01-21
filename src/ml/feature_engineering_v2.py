"""
FEATURE ENGINEERING V2 WITH ENHANCED PEDAGOGY
- Fixed prev/next fingering to use actual previous/next notes
- Added 5 new features for better accuracy
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

def add_v2_features(df):
    """Add new V2 features for better accuracy"""
    
    print("\nAdding V2 features...")
    
    # Feature 15: Finger span (same as abs_interval but explicit)
    df['finger_span'] = df['abs_interval'].copy()
    
    # Feature 16: Hand position (rolling average of recent 5 pitches)
    df['hand_position'] = df.groupby(['file', 'part'])['midi'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Feature 17: Scale pattern detection (3+ consecutive steps in same direction)
    df['in_scale_pattern'] = 0
    
    for i in range(2, len(df)):
        # Check if we have at least 2 previous notes in same part
        if i >= 2:
            current_file = df.loc[i, 'file']
            current_part = df.loc[i, 'part']
            
            # Check if same file and part
            if (df.loc[i-1, 'file'] == current_file and 
                df.loc[i-2, 'file'] == current_file and
                df.loc[i-1, 'part'] == current_part and 
                df.loc[i-2, 'part'] == current_part):
                
                # Get last 3 intervals
                int1 = df.loc[i-2, 'interval']
                int2 = df.loc[i-1, 'interval']
                int3 = df.loc[i, 'interval']
                
                # Check if all same direction and stepwise
                if not pd.isna(int1) and not pd.isna(int2) and not pd.isna(int3):
                    all_up = (0 < int1 <= 2) and (0 < int2 <= 2) and (0 < int3 <= 2)
                    all_down = (-2 <= int1 < 0) and (-2 <= int2 < 0) and (-2 <= int3 < 0)
                    
                    if all_up or all_down:
                        df.loc[i, 'in_scale_pattern'] = 1
    
    # Features 18 & 19: Measure boundary flags
    df['is_first_in_measure'] = 0
    df['is_last_in_measure'] = 0
    
    for file in df['file'].unique():
        file_df = df[df['file'] == file]
        for part in file_df['part'].unique():
            part_df = file_df[file_df['part'] == part]
            for measure in part_df['measure'].unique():
                measure_notes = part_df[part_df['measure'] == measure].index.tolist()
                if len(measure_notes) > 0:
                    df.loc[measure_notes[0], 'is_first_in_measure'] = 1
                    df.loc[measure_notes[-1], 'is_last_in_measure'] = 1
    
    print(f"✓ Added 5 new V2 features")
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
    
    # Add V2 features
    df = add_v2_features(df)
    
    features = [
        'midi', 'interval', 'abs_interval', 'measure_position_norm',
        'hand_encoding', 'prev_fingering', 'next_fingering',
        'is_black_key', 'is_step_motion', 'is_large_jump',
        'is_high_register', 'is_low_register', 'is_ascending', 'is_descending',
        # V2 FEATURES:
        'finger_span', 'hand_position', 'in_scale_pattern',
        'is_first_in_measure', 'is_last_in_measure'
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
    print("\nSample to verify features:")
    sample = df_features[['pitch', 'fingering', 'prev_fingering', 'next_fingering', 
                          'in_scale_pattern', 'hand_position']].head(10)
    print(sample)
    
    output_path = Path('data/training_data/features.csv')
    df_features.to_csv(output_path, index=False)
    
    print(f"\n✓ Features saved with {len(feature_names)} total features")
    print(f"\nNext: python src/ml/train_xgboost.py")

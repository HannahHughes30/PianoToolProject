"""
UNIVERSAL MODEL TRAINING
- Trains on annotated data
- Infers missing with pedagogy rules
- Uses common piano pedagogy
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def infer_with_pedagogy(row):
    """Infer fingering using piano pedagogy - returns (finger, confidence)"""
    midi = row['midi']
    interval = row['interval']
    prev_f = row['prev_fingering']
    next_f = row['next_fingering']
    is_black = row['is_black_key']
    
    # Rule 1: Use next finger context (look ahead)
    if next_f > 0 and row['is_step_motion']:
        if row['is_ascending'] and next_f == 1:
            return 3, 0.9  # Before thumb cross
        if row['is_ascending'] and next_f > 1:
            return int(next_f - 1), 0.9
        if row['is_descending'] and next_f == 3:
            return 1, 0.9  # Before cross over
        if row['is_descending'] and next_f < 5:
            return int(next_f + 1), 0.9
    
    # Rule 2: Step motion with previous context
    if prev_f > 0 and row['is_step_motion']:
        if row['is_ascending']:
            if prev_f == 3:
                return 2 if is_black else 1, 0.85  # Thumb cross, avoid black
            return min(int(prev_f + 1), 5), 0.9
        if row['is_descending']:
            if prev_f == 1:
                return 3, 0.85  # Cross over
            return max(int(prev_f - 1), 1), 0.9
    
    # Rule 3: Large jumps
    if row['is_large_jump']:
        if row['is_high_register']:
            return 5, 0.8
        if row['is_low_register']:
            return 1, 0.8
        return 1, 0.7
    
    # Rule 4: Starting note
    if prev_f == 0:
        if is_black:
            return 2, 0.6
        if row['is_high_register']:
            return 3, 0.6
        return 1, 0.7
    
    # Rule 5: Repeated note
    if interval == 0 and prev_f > 0:
        return int(prev_f), 0.6
    
    return 3, 0.4  # Default: middle finger

def train_model():
    features_path = Path('data/training_data/features.csv')
    if not features_path.exists():
        print("❌ Run: python src/ml/feature_engineering.py")
        return
    
    df = pd.read_csv(features_path)
    
    print("="*60)
    print("UNIVERSAL MODEL TRAINING")
    print("="*60)
    print(f"\nLoaded {len(df)} notes")
    
    original = df['has_annotation'].sum()
    missing = (~df['has_annotation']).sum()
    
    print(f"Annotated: {original} ({original/len(df)*100:.1f}%)")
    print(f"Missing: {missing}")
    
    # Infer missing fingerings
    if missing > 0:
        print("\nInferring with pedagogy rules...")
        for idx, row in df[~df['has_annotation']].iterrows():
            finger, conf = infer_with_pedagogy(row)
            df.at[idx, 'fingering'] = finger
            df.at[idx, 'confidence'] = conf
        print(f"✓ Inferred {missing} fingerings")
    
    # Mark confidence
    df.loc[df['has_annotation'], 'confidence'] = 1.0
    
    # Clean
    df = df[df['fingering'].between(1, 5)].copy()
    
    print(f"\nTraining on {len(df)} notes")
    
    # Features
    feature_cols = [
        'midi', 'interval', 'abs_interval', 'measure_position_norm',
        'hand_encoding', 'prev_fingering', 'next_fingering',
        'is_black_key', 'is_step_motion', 'is_large_jump',
        'is_high_register', 'is_low_register', 'is_ascending', 'is_descending'
    ]
    
    X = df[feature_cols].values
    y = df['fingering'].astype(int).values
    
    # Split smartly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.2%}")
    print(f"{'='*60}")
    
    print("\n" + classification_report(y_test, test_pred, 
                                       target_names=[f'Finger {i}' for i in range(1,6)],
                                       zero_division=0))
    
    # Feature importance
    print("Top Features:")
    importances = sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:8]:
        print(f"  {feat:25s} {'█'*int(imp*50)} {imp:.3f}")
    
    # Save
    model_path = Path('models/fingering_model.pkl')
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': feature_cols}, f)
    
    print(f"\n✓ Model saved to {model_path}")

if __name__ == '__main__':
    train_model()

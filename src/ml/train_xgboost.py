"""
XGBOOST MODEL TRAINING V2
- Enhanced inference rules using V2 features
- Better pedagogy for missing annotations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

def infer_with_pedagogy(row):
    """Infer fingering using enhanced piano pedagogy - returns (finger, confidence)"""
    midi = row['midi']
    interval = row['interval']
    prev_f = row['prev_fingering']
    next_f = row['next_fingering']
    is_black = row['is_black_key']
    
    # ============= NEW V2 RULES (HIGHEST PRIORITY) =============
    
    # NEW RULE 1: Detect scale patterns with new feature
    if row.get('in_scale_pattern', 0) == 1 and prev_f > 0:
        if row['is_ascending']:
            if prev_f == 3:
                return 1, 0.95  # Thumb cross after 3 in ascending scale
            elif prev_f < 5:
                return int(prev_f + 1), 0.93  # Sequential in scale
        if row['is_descending']:
            if prev_f == 1:
                return 3, 0.93  # Cross over thumb in descending
            elif prev_f > 1:
                return int(prev_f - 1), 0.93  # Sequential descending
    
    # NEW RULE 2: Use finger 4 more in high register (it's underused)
    if row['is_high_register'] == 1 and row['is_step_motion'] == 1:
        if prev_f == 3:
            return 4, 0.90  # Prefer 3-4 in high register
        if prev_f == 4:
            return 5, 0.90  # Continue 4-5
        if prev_f == 2:
            return 3, 0.88  # 2-3-4 pattern
    
    # NEW RULE 3: First note in measure - strong fingers
    if row.get('is_first_in_measure', 0) == 1:
        if prev_f == 0:  # Very first note
            if row['is_high_register']:
                return 3, 0.80  # Start high passages with 3
            if row['is_low_register']:
                return 1, 0.85  # Start low with thumb
            return 1, 0.75  # Default: thumb
        # If previous measure ended with 5, next should be 1 or 2
        if prev_f == 5:
            return 1, 0.88
        if prev_f == 4:
            return 1, 0.85
    
    # NEW RULE 4: Last note in measure - prepare for next measure
    if row.get('is_last_in_measure', 0) == 1:
        if row['is_high_register']:
            return 5, 0.82  # End high phrases with 5
        if interval == 0:  # Repeated note at phrase end
            return int(prev_f) if prev_f > 0 else 3, 0.75
    
    # NEW RULE 5: Hand position awareness
    hand_pos = row.get('hand_position', midi)
    # If current note is far from average hand position, use edge fingers
    distance_from_center = abs(midi - hand_pos)
    if distance_from_center > 7:  # More than a 5th away
        if midi > hand_pos:
            return 5, 0.85  # Reaching up
        else:
            return 1, 0.85  # Reaching down
    
    # NEW RULE 6: Avoid weak finger 4 on black keys
    if is_black and prev_f == 3:
        if row['is_ascending']:
            return 5, 0.87  # Skip 4, go to 5 on black key
        else:
            return 2, 0.85  # Go back to 2
    
    # NEW RULE 7: Repeated notes in high register
    if interval == 0 and row['is_high_register'] and prev_f > 0:
        # Alternate fingers on repeated high notes
        if prev_f == 5:
            return 4, 0.80
        if prev_f == 4:
            return 5, 0.80
        return int(prev_f), 0.70
    
    # ============= ORIGINAL RULES (FALLBACK) =============
    
    # Rule 1: Use next finger context (look ahead)
    if next_f > 0 and row['is_step_motion']:
        if row['is_ascending'] and next_f == 1:
            return 3, 0.90  # Before thumb cross
        if row['is_ascending'] and next_f > 1:
            return int(next_f - 1), 0.90
        if row['is_descending'] and next_f == 3:
            return 1, 0.90  # Before cross over
        if row['is_descending'] and next_f < 5:
            return int(next_f + 1), 0.90
    
    # Rule 2: Step motion with previous context
    if prev_f > 0 and row['is_step_motion']:
        if row['is_ascending']:
            if prev_f == 3:
                return 2 if is_black else 1, 0.85  # Thumb cross, avoid black
            return min(int(prev_f + 1), 5), 0.88
        if row['is_descending']:
            if prev_f == 1:
                return 3, 0.85  # Cross over
            return max(int(prev_f - 1), 1), 0.88
    
    # Rule 3: Large jumps
    if row['is_large_jump']:
        if row['is_high_register']:
            return 5, 0.82
        if row['is_low_register']:
            return 1, 0.82
        return 1, 0.75
    
    # Rule 4: Starting note (no previous context)
    if prev_f == 0:
        if is_black:
            return 2, 0.65
        if row['is_high_register']:
            return 3, 0.65
        return 1, 0.70
    
    # Rule 5: Repeated note
    if interval == 0 and prev_f > 0:
        return int(prev_f), 0.65
    
    # Default: middle finger (safest choice)
    return 3, 0.45

def train_model():
    features_path = Path('data/training_data/features.csv')
    if not features_path.exists():
        print("❌ Run: python src/ml/feature_engineering_v2.py")
        return
    
    df = pd.read_csv(features_path)
    
    print("="*60)
    print("XGBOOST MODEL TRAINING V2 (Enhanced Rules)")
    print("="*60)
    print(f"\nLoaded {len(df)} notes")
    
    original = df['has_annotation'].sum()
    missing = (~df['has_annotation']).sum()
    
    print(f"Annotated: {original} ({original/len(df)*100:.1f}%)")
    print(f"Missing: {missing}")
    
    # Infer missing fingerings with enhanced rules
    if missing > 0:
        print("\nInferring with enhanced pedagogy rules...")
        inferred_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for idx, row in df[~df['has_annotation']].iterrows():
            finger, conf = infer_with_pedagogy(row)
            df.at[idx, 'fingering'] = finger
            df.at[idx, 'confidence'] = conf
            inferred_counts[finger] += 1
        
        print(f"✓ Inferred {missing} fingerings")
        print(f"  Finger distribution: {inferred_counts}")
    
    # Mark confidence
    df.loc[df['has_annotation'], 'confidence'] = 1.0
    
    # Clean
    df = df[df['fingering'].between(1, 5)].copy()
    
    print(f"\nTraining on {len(df)} notes")
    
    # Features - using V2 feature set
    feature_cols = [
        'midi', 'interval', 'abs_interval', 'measure_position_norm',
        'hand_encoding', 'prev_fingering', 'next_fingering',
        'is_black_key', 'is_step_motion', 'is_large_jump',
        'is_high_register', 'is_low_register', 'is_ascending', 'is_descending',
        # V2 FEATURES:
        'finger_span', 'hand_position', 'in_scale_pattern',
        'is_first_in_measure', 'is_last_in_measure'
    ]
    
    X = df[feature_cols].values
    y = df['fingering'].astype(int).values
    
    # Convert to 0-indexed for XGBoost
    y = y - 1  # 1-5 becomes 0-4
    
    # Split smartly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train XGBoost with slightly tuned parameters
    print("\nTraining XGBoost V2...")
    model = xgb.XGBClassifier(
        n_estimators=250,  # Increased from 200
        max_depth=7,       # Increased from 6
        learning_rate=0.08,  # Slightly lower for better generalization
        objective='multi:softmax',
        num_class=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,  # Prevent overfitting
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate - convert back to 1-5 for display
    test_pred = model.predict(X_test) + 1
    y_test_display = y_test + 1
    
    test_acc = accuracy_score(y_test_display, test_pred)
    
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY: {test_acc:.2%}")
    print(f"{'='*60}")
    
    print("\n" + classification_report(y_test_display, test_pred, 
                                       target_names=[f'Finger {i}' for i in range(1,6)],
                                       zero_division=0))
    
    # Feature importance
    print("Top Features:")
    importances = sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:10]:  # Show top 10
        bar_length = int(imp * 50)
        print(f"  {feat:25s} {'█' * bar_length} {imp:.3f}")
    
    # Analyze per-finger performance
    print("\nPer-Finger Analysis:")
    for finger in range(1, 6):
        mask = (y_test_display == finger)
        if mask.sum() > 0:
            finger_acc = (test_pred[mask] == finger).sum() / mask.sum()
            print(f"  Finger {finger}: {finger_acc:.1%} ({mask.sum()} examples)")
    
    # Save model
    model_path = Path('models/fingering_model_xgb_v2.pkl')
    model_path.parent.mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'features': feature_cols,
            'needs_conversion': True,
            'version': 'v2'
        }, f)
    
    print(f"\n✓ XGBoost V2 model saved to {model_path}")
    
    # Compare to baselines
    print(f"\n{'='*60}")
    print("COMPARISON TO BASELINES:")
    print(f"{'='*60}")
    print(f"  Random Forest:        79.7%")
    print(f"  XGBoost V1:           80.7%")
    print(f"  XGBoost V2 (current): {test_acc:.1%}")
    
    if test_acc > 0.817:
        improvement = (test_acc - 0.807) * 100
        print(f"\n  ✅ V2 is BETTER by {improvement:.1f} percentage points!")
        print(f"  → Enhanced rules + V2 features working!")
    elif test_acc > 0.807:
        improvement = (test_acc - 0.807) * 100
        print(f"\n  ✅ Small improvement: +{improvement:.1f} points")
        print(f"  → Need more diverse training data for bigger gains")
    else:
        print(f"\n  ⚠️  No improvement - V2 rules need more diverse data")
        print(f"  → Proceed to Week 2: data collection")

if __name__ == '__main__':
    train_model()

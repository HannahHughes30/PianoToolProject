"""
ENSEMBLE: V2 + Physics + Pianoplayer Predictions
Maximum accuracy approach
"""
import pandas as pd
import sys
sys.path.append('src/ml')
from feature_engineering_v2 import extract_features as extract_v2_features
from pianoplayer_features import extract_pianoplayer_features
from pianoplayer_predictor import add_pianoplayer_predictions

def extract_ensemble_features(df):
    """
    Full ensemble: All features + pianoplayer predictions
    """
    print("="*60)
    print("ENSEMBLE FEATURE EXTRACTION")
    print("  ML + Physics + Expert Algorithm")
    print("="*60)
    
    # Step 1: V2 pedagogy features
    print("\nStep 1: V2 pedagogy features...")
    df_v2, v2_names = extract_v2_features(df)
    
    # Step 2: Physics features
    print("\nStep 2: Pianoplayer physics features...")
    df_physics = extract_pianoplayer_features(df_v2)
    
    # Step 3: Pianoplayer predictions
    print("\nStep 3: Pianoplayer expert predictions...")
    df_ensemble = add_pianoplayer_predictions(df_physics)
    
    # All features
    ensemble_features = v2_names + [
        'pp_speed_cost',
        'pp_hand_stretch',
        'pp_forbidden',
        'pp_black_penalty',
        'pp_prediction',
        'pp_confidence'
    ]
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE FEATURE SUMMARY:")
    print(f"  V2 pedagogy:        {len(v2_names)}")
    print(f"  Physics features:   4")
    print(f"  Expert predictions: 2")
    print(f"  TOTAL:              {len(ensemble_features)}")
    print(f"{'='*60}")
    
    return df_ensemble, ensemble_features


if __name__ == '__main__':
    df = pd.read_csv('data/training_data/fingering_data.csv')
    print(f"\nLoaded {len(df)} notes\n")
    
    df_ensemble, features = extract_ensemble_features(df)
    
    output_path = 'data/training_data/features_ensemble.csv'
    df_ensemble.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

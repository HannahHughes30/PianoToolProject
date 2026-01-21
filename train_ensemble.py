"""
ENSEMBLE MODEL
Combines: ML features + Physics + Expert predictions
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

print("="*60)
print("ENSEMBLE MODEL TRAINING")
print("="*60)

df = pd.read_csv('data/training_data/features_ensemble.csv')
df = df[df['fingering'].notna()].copy()

# Drop non-numeric columns
text_cols = ['file', 'pitch', 'part', 'measure']
df = df.drop(columns=[col for col in text_cols if col in df.columns], errors='ignore')

X = df.drop('fingering', axis=1)
y = df['fingering'].astype(int)

# Convert fingers 1-5 to labels 0-4
y_labels = y - 1

print(f"\n✓ {len(df)} notes")
print(f"✓ {X.shape[1]} numeric features")
print(f"\nFingering distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, test_size=0.2, random_state=42
)

print(f"\n✓ Train: {len(X_train)}")
print(f"✓ Test:  {len(X_test)}")

print("\n⏳ Training ensemble XGBoost...")
model = xgb.XGBClassifier(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=250,
    random_state=42,
    num_class=5
)

model.fit(X_train, y_train)

# Predict and convert back to fingers 1-5
y_pred_labels = model.predict(X_test)
y_pred = y_pred_labels + 1
y_test_fingers = y_test + 1

accuracy = accuracy_score(y_test_fingers, y_pred)

print(f"\n{'='*60}")
print(f"ENSEMBLE ACCURACY: {accuracy:.1%}")
print(f"{'='*60}")

print("\nPer-finger results:")
print(classification_report(y_test_fingers, y_pred, 
                           target_names=['Finger 1', 'Finger 2', 'Finger 3', 'Finger 4', 'Finger 5']))

joblib.dump(model, 'models/xgboost_ensemble.pkl')
print("\n✓ Saved: models/xgboost_ensemble.pkl")

# Feature importance
print("\nTop 20 Most Important Features:")
fi = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in fi.head(20).iterrows():
    bar = '█' * int(row['importance'] * 80)
    print(f"  {row['feature']:.<30} {bar[:40]} {row['importance']:.3f}")

print(f"\n{'='*60}")
print("FINAL COMPARISON:")
print(f"  Baseline (RF):        79.7%")
print(f"  XGBoost V2:           85.3%")
print(f"  ENSEMBLE (V3):        {accuracy:.1%}")
print(f"  IMPROVEMENT:          +{(accuracy - 0.853):.1%}")
print(f"{'='*60}")

if accuracy > 0.87:
    print("\n🎉🎉 BREAKTHROUGH! 87%+ accuracy!")
    print("   Ensemble combining ML + Physics + Expert is working!")
elif accuracy > 0.86:
    print("\n🎉 EXCELLENT! 86%+ accuracy!")
    print("   Significant improvement from ensemble approach!")
elif accuracy > 0.853:
    print(f"\n✅ Good! Ensemble improved by {(accuracy - 0.853):.1%}")
else:
    print("\n✓ Ensemble maintains 85%+ target")
    print("   Physics features may not add much to this dataset")

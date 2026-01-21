"""
Train XGBoost on combined dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

print("="*60)
print("TRAINING XGBOOST - COMBINED DATA")
print("="*60)

# Load features
df = pd.read_csv('data/training_data/features_combined.csv')
print(f"\n✓ Loaded {len(df)} samples")

# Prepare X and y
X = df.drop('fingering', axis=1)
y = df['fingering']

print(f"  Features: {X.shape[1]}")
print(f"  Fingering distribution:\n{y.value_counts().sort_index()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {len(X_train)} samples")
print(f"✓ Test set:  {len(X_test)} samples")

# Train XGBoost
print("\n⏳ Training XGBoost...")
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"COMBINED MODEL ACCURACY: {accuracy:.1%}")
print(f"{'='*60}")

print("\nDetailed Results:")
print(classification_report(y_test, y_pred))

# Save model
model_path = 'models/xgboost_combined.pkl'
joblib.dump(model, model_path)
print(f"\n✓ Model saved: {model_path}")

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

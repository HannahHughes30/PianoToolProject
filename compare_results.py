"""
Compare old vs new model performance
"""
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

print("="*60)
print("MODEL COMPARISON")
print("="*60)

# Load test data
df = pd.read_csv('data/training_data/features_combined.csv')
X = df.drop('fingering', axis=1)
y = df['fingering']

# Old model (if exists)
try:
    old_model = joblib.load('models/xgboost_model.pkl')
    old_pred = old_model.predict(X)
    old_acc = accuracy_score(y, old_pred)
    print(f"\n📊 OLD MODEL (810 notes):  {old_acc:.1%}")
except:
    print("\n⚠️ Old model not found")
    old_acc = 0.797  # Your reported accuracy

# New model
new_model = joblib.load('models/xgboost_combined.pkl')
new_pred = new_model.predict(X)
new_acc = accuracy_score(y, new_pred)
print(f"📊 NEW MODEL (combined):  {new_acc:.1%}")

# Improvement
improvement = new_acc - old_acc
print(f"\n{'='*60}")
print(f"IMPROVEMENT: +{improvement:.1%}")
print(f"{'='*60}")

if new_acc >= 0.85:
    print("\n🎉 SUCCESS! You hit 85% accuracy!")
elif new_acc >= 0.82:
    print(f"\n✅ Good progress! {0.85 - new_acc:.1%} away from 85%")
    print("💡 Adding physics features should get you there!")
else:
    print(f"\n⏳ {0.85 - new_acc:.1%} away from 85%")
    print("💡 Consider: more data OR physics features")

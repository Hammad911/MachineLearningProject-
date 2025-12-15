"""
Retrain LightGBM with Better Balance
Goal: Catch malware while minimizing false positives on benign apps
Strategy: Use moderate (not aggressive) class weights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
from datetime import datetime
import shutil
import os
import gc

print("="*80)
print("🔄 RETRAINING LIGHTGBM WITH BETTER BALANCE")
print("="*80)

# Load data
print("\n📦 Loading dataset...")
df = pd.read_parquet('DataSet/cicandmal2020-static.parquet')
print(f"   ✓ Loaded {len(df):,} samples")

# Prepare data
X = df.drop(['F0', 'Label'], axis=1)
y = df['Label']
feature_columns = X.columns.tolist()

# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split data
print("\n   Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training: {len(X_train):,}, Test: {len(X_test):,}")

# Calculate class weights
print("\n📊 Calculating MODERATE class weights...")
print("   (Not too aggressive this time)")

classes = np.unique(y_train)
base_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

# Create MODERATE weights (50% of full balanced weights)
# This reduces the aggressiveness
moderate_weights = {}
for label, weight in zip(classes, base_weights):
    if label == 'Benign':
        moderate_weights[label] = 1.0  # Keep Benign at 1.0
    else:
        # For rare classes, use 50% of balanced weight
        moderate_weights[label] = 1.0 + (weight - 1.0) * 0.5

print("\n   Moderate Class Weights:")
for label in sorted(moderate_weights.keys(), key=lambda x: moderate_weights[x], reverse=True)[:10]:
    print(f"   {label:15s}: {moderate_weights[label]:6.2f}x")

# Calculate sample weights
sample_weights = np.array([moderate_weights[label] for label in y_train])

# Clear memory
gc.collect()

# Train with BETTER parameters for precision
print("\n🚀 Training NEW LightGBM with better parameters...")
print("-" * 80)

lgb_model = LGBMClassifier(
    n_estimators=150,           # Fewer trees (faster, less overfitting)
    max_depth=8,                # Shallower trees (better generalization)
    learning_rate=0.1,          # Faster learning
    num_leaves=31,
    min_child_samples=50,       # Higher minimum (prevents overfitting to rare classes)
    subsample=0.7,              # Use 70% of data
    colsample_bytree=0.7,       # Use 70% of features
    reg_alpha=0.5,              # Stronger L1 regularization
    reg_lambda=0.5,             # Stronger L2 regularization
    min_split_gain=0.1,         # Require minimum gain to split
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("   Training...")
lgb_model.fit(
    X_train, 
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    eval_metric='multi_logloss'
)

print("   ✓ Training complete!")

# Evaluate
print("\n📊 Evaluating model...")
y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   🎯 Overall Accuracy: {accuracy*100:.2f}%")

# Test on actual files
print("\n🧪 Testing on Real APK Files...")
print("="*80)

from apk_analyzer import APKAnalyzer
analyzer = APKAnalyzer(feature_columns)

test_files = [
    ('MALWARE_Dendroid.apk', 'Malware', 'Dendroid'),
    ('MALWARE_CandyCorn.apk', 'Malware', 'CandyCorn'),
    ('MALWARE_xHelper.apk', 'Malware', 'xHelper'),
    ('jojoy-app_v3.3.0_release.apk', 'Benign', 'JoJoy'),
    ('Temple Run_1.34.1_APKPure.xapk', 'Benign', 'Temple Run'),
    ('Higgs Domino Global_2.37_APKPure.xapk', 'Benign', 'Higgs Domino'),
]

results = []

for apk_file, true_label, name in test_files:
    if not os.path.exists(apk_file):
        continue
    
    try:
        file_type = 'xapk' if apk_file.endswith('.xapk') else 'apk'
        features_dict = analyzer.extract_from_file(apk_file, file_type)
        features_array = analyzer.features_to_vector(features_dict)
        
        # Predict
        pred = lgb_model.predict(features_array)[0]
        proba = lgb_model.predict_proba(features_array)[0]
        conf = proba.max()
        
        # Get benign probability
        benign_idx = list(classes).index('Benign')
        benign_prob = proba[benign_idx]
        
        # Check if correct
        correct = (pred != 'Benign' and true_label == 'Malware') or \
                 (pred == 'Benign' and true_label == 'Benign')
        
        status = '✅' if correct else '❌'
        
        print(f"{name:15s} | {true_label:7s} → {pred:12s} ({conf*100:5.1f}%) | Benign: {benign_prob*100:5.1f}% | {status}")
        
        results.append({
            'name': name,
            'true': true_label,
            'pred': pred,
            'correct': correct,
            'benign_prob': benign_prob,
            'conf': conf
        })
        
    except Exception as e:
        print(f"{name:15s} | Error: {e}")

# Calculate metrics
malware_results = [r for r in results if r['true'] == 'Malware']
benign_results = [r for r in results if r['true'] == 'Benign']

malware_detected = sum(1 for r in malware_results if r['pred'] != 'Benign')
benign_correct = sum(1 for r in benign_results if r['pred'] == 'Benign')

print("\n" + "="*80)
print("📊 PERFORMANCE SUMMARY")
print("="*80)

print(f"\n🦠 Malware Detection:")
print(f"   Detected: {malware_detected}/{len(malware_results)} ({malware_detected/len(malware_results)*100:.0f}%)")

print(f"\n✅ Benign Recognition:")
print(f"   Correct: {benign_correct}/{len(benign_results)} ({benign_correct/len(benign_results)*100:.0f}%)")
print(f"   False Positives: {len(benign_results)-benign_correct}/{len(benign_results)}")

print(f"\n🎯 Overall on Test APKs: {malware_detected + benign_correct}/{len(results)} ({(malware_detected + benign_correct)/len(results)*100:.0f}%)")

# Compare to old model
print("\n" + "="*80)
print("📊 COMPARISON: Old vs New LightGBM")
print("="*80)

print("\n❌ OLD LightGBM (Aggressive weights):")
print("   • Malware Detection: 3/3 (100%)")
print("   • Benign Recognition: 0/4 (0%)")
print("   • False Positives: 4/4 (100%)")
print("   → Problem: Too many false alarms!")

print(f"\n✅ NEW LightGBM (Moderate weights):")
print(f"   • Malware Detection: {malware_detected}/3 ({malware_detected/3*100:.0f}%)")
print(f"   • Benign Recognition: {benign_correct}/3 ({benign_correct/3*100:.0f}%)")
print(f"   • False Positives: {3-benign_correct}/3 ({(3-benign_correct)/3*100:.0f}%)")

if benign_correct >= 2 and malware_detected >= 2:
    print("   → 🎊 MUCH BETTER BALANCE!")
elif benign_correct > 0:
    print("   → 🟡 Improvement but needs more tuning")
else:
    print("   → ⚠️  Still too aggressive")

# Save model if it's better
if benign_correct >= 2 or (benign_correct >= 1 and malware_detected == 3):
    print("\n💾 Saving improved model...")
    
    # Backup old
    if os.path.exists('models/lightgbm_model.pkl'):
        backup_name = f'models/lightgbm_model_aggressive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        shutil.copy('models/lightgbm_model.pkl', backup_name)
        print(f"   ✓ Backed up old model: {backup_name}")
    
    # Save new
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
    print(f"   ✓ Saved NEW LightGBM model")
    
    # Update metadata
    metadata = {
        'model_type': 'LGBMClassifier',
        'accuracy': accuracy,
        'n_features': len(feature_columns),
        'n_classes': len(classes),
        'classes': classes.tolist(),
        'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'class_weights': 'moderate (50% of balanced)',
        'notes': 'Retrained with moderate class weights for better precision-recall balance',
        'real_apk_performance': {
            'malware_detection': f'{malware_detected}/3',
            'benign_recognition': f'{benign_correct}/3',
            'false_positives': f'{3-benign_correct}/3'
        }
    }
    
    joblib.dump(metadata, 'models/lightgbm_metadata.pkl')
    print(f"   ✓ Updated metadata")
    
    print("\n✅ Model saved successfully!")
    print("\n🚀 Next step: Restart the app and test!")
    
else:
    print("\n⚠️  Model not saved (not better than old one)")
    print("   Consider trying different parameters or approach")

print("\n" + "="*80)
print("✅ RETRAINING COMPLETE!")
print("="*80)




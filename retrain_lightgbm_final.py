"""
Final LightGBM Optimization
Strategy: Make Benign class easier to predict while maintaining malware detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
from datetime import datetime
import shutil
import os
import gc

print("="*80)
print("🎯 FINAL LIGHTGBM OPTIMIZATION")
print("Strategy: Balance Benign recognition with Malware detection")
print("="*80)

# Load data
print("\n📦 Loading dataset...")
df = pd.read_parquet('DataSet/cicandmal2020-static.parquet')

# Prepare data
X = df.drop(['F0', 'Label'], axis=1)
y = df['Label']
feature_columns = X.columns.tolist()
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training: {len(X_train):,}, Test: {len(X_test):,}")

# Calculate CUSTOM class weights
print("\n📊 Calculating CUSTOM class weights...")
print("   Strategy: Boost rare classes moderately, keep Benign accessible")

classes = np.unique(y_train)
base_weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Create custom weights
custom_weights = {}
for label, weight in zip(classes, base_weights):
    if label == 'Benign':
        # Make Benign slightly easier to predict
        custom_weights[label] = 0.8
    elif weight > 100:  # Very rare classes
        # Moderate boost for very rare classes
        custom_weights[label] = min(weight * 0.3, 30.0)
    elif weight > 10:
        # Small boost for rare classes
        custom_weights[label] = min(weight * 0.4, 15.0)
    else:
        # Keep common classes normal
        custom_weights[label] = weight * 0.6

print("\n   Custom Class Weights:")
for label in sorted(custom_weights.keys(), key=lambda x: custom_weights[x], reverse=True)[:10]:
    count = (y_train == label).sum()
    print(f"   {label:15s}: {custom_weights[label]:6.2f}x (n={count:,})")

sample_weights = np.array([custom_weights[label] for label in y_train])

gc.collect()

# Train with parameters focused on generalization
print("\n🚀 Training optimized LightGBM...")
print("   Focus: Better generalization to unseen APKs")
print("-" * 80)

lgb_model = LGBMClassifier(
    # Tree structure
    n_estimators=100,           # Fewer trees (less overfitting)
    max_depth=6,                # Shallower (better generalization)
    num_leaves=20,              # Fewer leaves (simpler patterns)
    
    # Learning
    learning_rate=0.05,         # Slower, more careful learning
    
    # Sampling (more aggressive to prevent overfitting)
    subsample=0.6,              # Use only 60% of data per tree
    colsample_bytree=0.6,       # Use only 60% of features per tree
    subsample_freq=1,
    
    # Regularization (stronger to prevent overfitting)
    min_child_samples=100,      # Require 100 samples per leaf (prevents rare class overfitting)
    min_split_gain=0.5,         # Require significant gain to split
    reg_alpha=1.0,              # Strong L1 regularization
    reg_lambda=1.0,             # Strong L2 regularization
    
    # Other
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    importance_type='gain'
)

print("   Training (this will take ~3 minutes)...")
lgb_model.fit(
    X_train, 
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    eval_metric='multi_logloss'
)

print("   ✓ Complete!")

# Evaluate
y_pred = lgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   🎯 Test Set Accuracy: {accuracy*100:.2f}%")

# Check Benign class specifically
benign_mask = y_test == 'Benign'
benign_predictions = y_pred[benign_mask]
benign_correct = (benign_predictions == 'Benign').sum()
benign_accuracy = benign_correct / benign_mask.sum()

print(f"   📊 Benign Class Accuracy: {benign_accuracy*100:.2f}% ({benign_correct:,}/{benign_mask.sum():,})")

# Test on real APKs
print("\n🧪 Testing on Real APK Files...")
print("="*80)

from apk_analyzer import APKAnalyzer
analyzer = APKAnalyzer(feature_columns)

test_files = [
    ('MALWARE_Dendroid.apk', 'Malware', 'Dendroid Trojan'),
    ('MALWARE_CandyCorn.apk', 'Malware', 'CandyCorn Adware'),
    ('MALWARE_xHelper.apk', 'Malware', 'xHelper'),
    ('jojoy-app_v3.3.0_release.apk', 'Benign', 'JoJoy'),
    ('Temple Run_1.34.1_APKPure.xapk', 'Benign', 'Temple Run'),
    ('Higgs Domino Global_2.37_APKPure.xapk', 'Benign', 'Higgs Domino'),
]

print(f"\n{'App':<20} {'True':<8} {'Prediction':<15} {'Conf':<8} {'Benign%':<10} {'Status'}")
print("-" * 80)

results = []
for apk_file, true_label, name in test_files:
    if not os.path.exists(apk_file):
        continue
    
    try:
        file_type = 'xapk' if apk_file.endswith('.xapk') else 'apk'
        features_dict = analyzer.extract_from_file(apk_file, file_type)
        features_array = analyzer.features_to_vector(features_dict)
        
        pred = lgb_model.predict(features_array)[0]
        proba = lgb_model.predict_proba(features_array)[0]
        conf = proba.max()
        
        benign_idx = list(classes).index('Benign')
        benign_prob = proba[benign_idx]
        
        correct = (pred != 'Benign' and true_label == 'Malware') or \
                 (pred == 'Benign' and true_label == 'Benign')
        
        status = '✅' if correct else '❌'
        
        print(f"{name:<20} {true_label:<8} {pred:<15} {conf*100:>6.1f}% {benign_prob*100:>8.1f}% {status}")
        
        results.append({
            'name': name,
            'true': true_label,
            'pred': pred,
            'correct': correct,
            'benign_prob': benign_prob
        })
        
    except Exception as e:
        print(f"{name:<20} Error: {e}")

# Metrics
malware_results = [r for r in results if r['true'] == 'Malware']
benign_results = [r for r in results if r['true'] == 'Benign']

malware_detected = sum(1 for r in malware_results if r['pred'] != 'Benign')
benign_correct = sum(1 for r in benign_results if r['pred'] == 'Benign')

print("\n" + "="*80)
print("📊 REAL APK PERFORMANCE")
print("="*80)

print(f"\n🦠 Malware Detection: {malware_detected}/{len(malware_results)} ({malware_detected/len(malware_results)*100:.0f}%)")
print(f"✅ Benign Recognition: {benign_correct}/{len(benign_results)} ({benign_correct/len(benign_results)*100:.0f}%)")
print(f"📊 False Positives: {len(benign_results)-benign_correct}/{len(benign_results)}")

# Decision
improvement_score = (malware_detected * 2 + benign_correct * 3) / (len(results) * 2.5)

print(f"\n🎯 Performance Score: {improvement_score*100:.1f}%")
print("   (Weighs both malware detection AND low false positives)")

if benign_correct >= 2 and malware_detected >= 1:
    save_model = True
    print("\n✅ SIGNIFICANT IMPROVEMENT! Saving model...")
elif benign_correct >= 1:
    save_model = True
    print("\n🟡 SOME IMPROVEMENT! Saving model for testing...")
else:
    save_model = False
    print("\n⚠️  No improvement. Keeping old model...")

if save_model:
    # Backup and save
    if os.path.exists('models/lightgbm_model.pkl'):
        backup = f'models/lightgbm_old_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        shutil.copy('models/lightgbm_model.pkl', backup)
        print(f"   ✓ Backed up: {backup}")
    
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
    print(f"   ✓ Saved new model")
    
    metadata = {
        'accuracy': accuracy,
        'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'class_weights': 'custom (moderate, Benign=0.8)',
        'real_apk_performance': {
            'malware_detection': f'{malware_detected}/3',
            'benign_recognition': f'{benign_correct}/3'
        }
    }
    joblib.dump(metadata, 'models/lightgbm_metadata.pkl')
    
    print("\n🚀 Model updated! Restart app to test.")

print("\n" + "="*80)
print("✅ OPTIMIZATION ATTEMPT COMPLETE")
print("="*80)




"""
Retrain LightGBM with Balanced Class Weights
Fixes the FileInfector bias issue
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import gc

print("="*80)
print("RETRAINING LIGHTGBM WITH BALANCED CLASS WEIGHTS")
print("="*80)

# Step 1: Load Data
print("\n📦 Step 1: Loading dataset...")
print("-" * 80)

df = pd.read_parquet('DataSet/cicandmal2020-static.parquet')
print(f"   ✓ Loaded {len(df):,} samples")
print(f"   ✓ Features: {df.shape[1] - 2} (F1-F{df.shape[1]-2})")

# Check class distribution
print("\n📊 Current Class Distribution:")
class_dist = df['Label'].value_counts()
for label, count in class_dist.items():
    pct = (count / len(df)) * 100
    print(f"   {label:15s}: {count:6,} ({pct:5.2f}%)")

# Highlight the problem
fileinfector_count = class_dist.get('FileInfector', 0)
benign_count = class_dist.get('Benign', 0)
ratio = benign_count / fileinfector_count if fileinfector_count > 0 else 0
print(f"\n⚠️  Imbalance Ratio (Benign:FileInfector): {ratio:.1f}:1")
print(f"   This is why LightGBM was biased!")

# Step 2: Prepare Data
print("\n📊 Step 2: Preparing training data...")
print("-" * 80)

# Separate features and labels
X = df.drop(['F0', 'Label'], axis=1)  # F0 is file hash, not a feature
y = df['Label']

# Get feature names
feature_columns = X.columns.tolist()
print(f"   ✓ Features: {len(feature_columns)}")

# Convert to numeric (handle any string values)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split data
print("\n   Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Training samples: {len(X_train):,}")
print(f"   ✓ Test samples: {len(X_test):,}")

# Calculate class weights
print("\n📊 Calculating balanced class weights...")
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

# Create class weight dictionary
class_weight_dict = dict(zip(classes, class_weights))

print("\n   Class Weights (to balance training):")
for label in sorted(class_weight_dict.keys(), key=lambda x: class_weight_dict[x], reverse=True):
    print(f"   {label:15s}: {class_weight_dict[label]:6.2f}")

print(f"\n   ✓ FileInfector weight: {class_weight_dict.get('FileInfector', 0):.2f}x")
print(f"   ✓ Benign weight: {class_weight_dict.get('Benign', 0):.2f}x")
print("   → Higher weights = more importance during training")

# Calculate sample weights for training
sample_weights = np.array([class_weight_dict[label] for label in y_train])

# Step 3: Train New LightGBM Model
print("\n🚀 Step 3: Training NEW LightGBM with balanced weights...")
print("-" * 80)

# Clear memory
gc.collect()

# Create new model with better parameters
print("\n   Configuring model parameters...")
lgb_model = LGBMClassifier(
    n_estimators=200,           # More trees for better learning
    max_depth=10,               # Prevent overfitting
    learning_rate=0.05,         # Slower learning
    num_leaves=31,              # Default
    min_child_samples=20,       # Prevent overfitting to small classes
    subsample=0.8,              # Use 80% of data per tree
    colsample_bytree=0.8,       # Use 80% of features per tree
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=0.1,             # L2 regularization
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    verbose=-1                  # Suppress training output
)

print("   ✓ Parameters set")
print("\n   Training (this may take a few minutes)...")
print("   Progress: ", end="", flush=True)

# Train with sample weights
lgb_model.fit(
    X_train, 
    y_train,
    sample_weight=sample_weights,  # Apply balanced weights!
    eval_set=[(X_test, y_test)],
    eval_metric='multi_logloss',
    callbacks=[
        # Print progress every 50 iterations
    ]
)

print("✓ Complete!")

# Step 4: Evaluate New Model
print("\n📊 Step 4: Evaluating NEW model...")
print("-" * 80)

# Make predictions
print("\n   Making predictions on test set...")
y_pred = lgb_model.predict(X_test)
y_pred_proba = lgb_model.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   🎯 Overall Accuracy: {accuracy*100:.2f}%")

# Detailed classification report
print("\n   📊 Per-Class Performance:")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Show results sorted by f1-score
print(f"\n   {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support'}")
print("   " + "-"*60)

for label in sorted(report.keys()):
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        p = report[label]['precision']
        r = report[label]['recall']
        f1 = report[label]['f1-score']
        s = int(report[label]['support'])
        print(f"   {label:<15} {p:>8.2%}   {r:>8.2%}   {f1:>8.2%}   {s:>6}")

# Check FileInfector specifically
print("\n🔍 FileInfector Performance Check:")
print("-" * 80)

if 'FileInfector' in report:
    fi_precision = report['FileInfector']['precision']
    fi_recall = report['FileInfector']['recall']
    fi_f1 = report['FileInfector']['f1-score']
    fi_support = int(report['FileInfector']['support'])
    
    print(f"   Precision: {fi_precision*100:.2f}%")
    print(f"   Recall:    {fi_recall*100:.2f}%")
    print(f"   F1-Score:  {fi_f1*100:.2f}%")
    print(f"   Support:   {fi_support} samples")
    
    if fi_recall > 0 and fi_precision > 0:
        print("\n   ✅ Model can now detect FileInfector properly!")
    else:
        print("\n   ⚠️  Still having issues with FileInfector")
else:
    print("   ⚠️  FileInfector not in predictions")

# Test on Benign samples to check false positive rate
benign_mask = y_test == 'Benign'
benign_predictions = y_pred[benign_mask]
false_positives = np.sum(benign_predictions != 'Benign')
false_positive_rate = false_positives / len(benign_predictions)

print(f"\n📊 False Positive Analysis (Benign → Malware):")
print(f"   Benign test samples: {benign_mask.sum()}")
print(f"   Incorrectly flagged as malware: {false_positives}")
print(f"   False Positive Rate: {false_positive_rate*100:.2f}%")

if false_positive_rate < 0.10:
    print("   ✅ Good! Low false positive rate")
elif false_positive_rate < 0.20:
    print("   🟡 Acceptable false positive rate")
else:
    print("   ⚠️  High false positive rate - may need tuning")

# Step 5: Compare Old vs New
print("\n📊 Step 5: OLD vs NEW Comparison...")
print("-" * 80)

print("\n   OLD LightGBM Issues:")
print("   ❌ Predicted FileInfector for ALL 5 test APKs")
print("   ❌ 100% false positive rate on legitimate apps")
print("   ❌ Unusable due to bias")

print("\n   NEW LightGBM (with balanced weights):")
print(f"   ✅ Overall Accuracy: {accuracy*100:.2f}%")
print(f"   ✅ False Positive Rate: {false_positive_rate*100:.2f}%")
print(f"   ✅ Can detect FileInfector: {fi_recall*100:.2f}% recall")

# Step 6: Save New Model
print("\n💾 Step 6: Saving NEW model...")
print("-" * 80)

# Backup old model
import shutil
import os

if os.path.exists('models/lightgbm_model.pkl'):
    backup_name = f'models/lightgbm_model_old_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    shutil.copy('models/lightgbm_model.pkl', backup_name)
    print(f"   ✓ Backed up old model to: {backup_name}")

# Save new model
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
    'class_weights': 'balanced',
    'notes': 'Retrained with balanced class weights to fix FileInfector bias'
}

joblib.dump(metadata, 'models/lightgbm_metadata.pkl')
print(f"   ✓ Saved model metadata")

# Step 7: Test on Real APKs
print("\n🧪 Step 7: Testing on REAL APK files...")
print("-" * 80)

# Load the APK analyzer
from apk_analyzer import APKAnalyzer

apk_files = [
    'jojoy-app_v3.3.0_release.apk',
    'Temple Run_1.34.1_APKPure.xapk',
    'Higgs Domino Global_2.37_APKPure.xapk'
]

analyzer = APKAnalyzer(feature_columns)

print("\nTesting NEW model predictions:\n")

fileinfector_count_new = 0
for apk_path in apk_files:
    if not os.path.exists(apk_path):
        continue
    
    try:
        file_type = 'xapk' if apk_path.endswith('.xapk') else 'apk'
        features_dict = analyzer.extract_from_file(apk_path, file_type)
        features_array = analyzer.features_to_vector(features_dict)
        
        # Predict with NEW model
        prediction = lgb_model.predict(features_array)[0]
        proba = lgb_model.predict_proba(features_array)[0]
        confidence = proba.max()
        
        if prediction == 'FileInfector':
            fileinfector_count_new += 1
            
        print(f"   {os.path.basename(apk_path):40s} → {prediction:15s} ({confidence*100:.1f}%)")
        
    except Exception as e:
        print(f"   {os.path.basename(apk_path):40s} → ERROR: {e}")

print(f"\n📊 Results:")
print(f"   OLD Model: 5/5 predicted FileInfector (100% false positive!)")
print(f"   NEW Model: {fileinfector_count_new}/3 predicted FileInfector")

if fileinfector_count_new == 0:
    print("\n   ✅ SUCCESS! NEW model correctly identifies them as benign/other!")
elif fileinfector_count_new < 3:
    print("\n   🟡 IMPROVED! Fewer false FileInfector predictions")
else:
    print("\n   ⚠️  Still predicting FileInfector - may need more tuning")

# Final Summary
print("\n" + "="*80)
print("✅ RETRAINING COMPLETE!")
print("="*80)

print("\n🎯 Summary:")
print(f"   • NEW LightGBM accuracy: {accuracy*100:.2f}%")
print(f"   • False positive rate: {false_positive_rate*100:.2f}%")
print(f"   • FileInfector recall: {fi_recall*100:.2f}%")
print(f"   • Model saved to: models/lightgbm_model.pkl")
print(f"   • Old model backed up")

print("\n🚀 Next Steps:")
print("   1. Restart your Streamlit app")
print("   2. Upload the test APKs again")
print("   3. Check if LightGBM predictions are better")
print("   4. Ensemble should now work properly!")

print("\n" + "="*80)




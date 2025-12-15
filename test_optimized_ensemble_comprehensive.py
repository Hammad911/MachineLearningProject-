"""
Test Optimized Ensemble on ALL 179 APKs
Compare against individual models
"""

import os
import numpy as np
from pathlib import Path
import joblib
from apk_analyzer import APKAnalyzer
from optimized_ensemble import OptimizedEnsemble

print("="*80)
print("🔬 TESTING OPTIMIZED ENSEMBLE ON ALL 179 APKs")
print("="*80)

# Load models
models = {
    'rf_model': joblib.load('models/random_forest_model.pkl'),
    'lgb_model': joblib.load('models/lightgbm_model.pkl'),
    'lr_model': joblib.load('models/logistic_regression_model.pkl'),
    'scaler': joblib.load('models/scaler.pkl'),
    'feature_columns': joblib.load('models/feature_columns.pkl'),
    'class_labels': joblib.load('models/class_labels.pkl')
}

# Create ensemble
ensemble = OptimizedEnsemble(models)
analyzer = APKAnalyzer(models['feature_columns'])

# Find all APK files
apk_files = []

# Benign apps (root)
for ext in ['*.apk', '*.xapk']:
    for apk in Path('.').glob(ext):
        if 'MALWARE' not in str(apk):
            apk_files.append(('benign', str(apk), apk.stem))

# Malware
malware_dir = Path('android-malware')
if malware_dir.exists():
    for category_dir in malware_dir.iterdir():
        if category_dir.is_dir():
            for apk in category_dir.glob('*.apk'):
                apk_files.append(('malware', str(apk), category_dir.name))

# Known malware from root
for apk in Path('.').glob('MALWARE_*.apk'):
    malware_type = apk.stem.replace('MALWARE_', '')
    apk_files.append(('malware', str(apk), malware_type))

print(f"\n✅ Found {len(apk_files)} APK files")

# Test
ensemble_correct = 0
ensemble_benign_correct = 0
ensemble_benign_total = 0
ensemble_malware_correct = 0
ensemble_malware_total = 0

print("\n" + "="*80)
print("🧪 TESTING...")
print("="*80)

print(f"\n{'APK':<40} {'Type':<8} {'Ensemble':<15} {'Confidence':<12} {'Result'}")
print("-" * 100)

for true_type, apk_path, category in apk_files:
    try:
        # Extract features
        file_type = 'xapk' if apk_path.endswith('.xapk') else 'apk'
        features_dict = analyzer.extract_from_file(apk_path, file_type)
        
        # Convert to features
        n_features = len(models['feature_columns'])
        features = np.zeros((1, n_features))
        for i, (key, value) in enumerate(sorted(features_dict.items())[:n_features]):
            if isinstance(value, (int, float)):
                features[0, i] = min(value, 1.0)
        
        # Predict with ensemble
        prediction, probabilities, confidence = ensemble.predict(features)
        
        # Check correctness
        correct = (prediction != 'Benign' and true_type == 'malware') or (prediction == 'Benign' and true_type == 'benign')
        
        if correct:
            ensemble_correct += 1
        
        if true_type == 'benign':
            ensemble_benign_total += 1
            if correct:
                ensemble_benign_correct += 1
        else:
            ensemble_malware_total += 1
            if correct:
                ensemble_malware_correct += 1
        
        # Display
        apk_name = os.path.basename(apk_path)[:38]
        symbol = '✅' if correct else '❌'
        
        print(f"{apk_name:<40} {true_type:<8} {prediction:<15} {confidence:>5.1f}%       {symbol}")
        
    except Exception as e:
        print(f"{os.path.basename(apk_path):<40} {true_type:<8} ERROR: {str(e)[:30]}")

# Final Summary
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)

print(f"\n{'Model':<25} {'Overall':<15} {'Benign':<15} {'Malware':<15} {'Grade'}")
print("-" * 100)

# Previous individual model results (from comprehensive test)
print(f"{'Random Forest':<25} {'70/179 (39%)':<15} {'5/5 (100%)':<15} {'65/174 (37%)':<15} {'❌ D'}")
print(f"{'LightGBM':<25} {'161/179 (90%)':<15} {'0/5 (0%)':<15} {'161/174 (93%)':<15} {'⭐ A'}")
print(f"{'Logistic Regression':<25} {'11/179 (6%)':<15} {'5/5 (100%)':<15} {'6/174 (3%)':<15} {'❌ D'}")

# Optimized Ensemble results
overall = f"{ensemble_correct}/{len(apk_files)}"
overall_pct = f"({ensemble_correct/len(apk_files)*100:.0f}%)" if len(apk_files) > 0 else "(0%)"

benign = f"{ensemble_benign_correct}/{ensemble_benign_total}"
benign_pct = f"({ensemble_benign_correct/ensemble_benign_total*100:.0f}%)" if ensemble_benign_total > 0 else "(0%)"

malware = f"{ensemble_malware_correct}/{ensemble_malware_total}"
malware_pct = f"({ensemble_malware_correct/ensemble_malware_total*100:.0f}%)" if ensemble_malware_total > 0 else "(0%)"

overall_score = ensemble_correct/len(apk_files)*100 if len(apk_files) > 0 else 0
if overall_score >= 95:
    grade = "🏆 A++"
elif overall_score >= 90:
    grade = "🏆 A+"
elif overall_score >= 85:
    grade = "⭐ A"
else:
    grade = "✅ B"

print("-" * 100)
print(f"{'🎯 OPTIMIZED ENSEMBLE':<25} {overall:<6} {overall_pct:<9} {benign:<6} {benign_pct:<9} {malware:<6} {malware_pct:<9} {grade}")

# Calculate improvement
print("\n" + "="*80)
print("📈 IMPROVEMENT vs BEST INDIVIDUAL MODEL (LightGBM)")
print("="*80)

lgb_overall = 161
ensemble_overall = ensemble_correct

lgb_benign = 0
ensemble_benign = ensemble_benign_correct

lgb_malware = 161
ensemble_malware = ensemble_malware_correct

print(f"\n  Overall Accuracy:")
print(f"    LightGBM: {lgb_overall}/179 (90%)")
print(f"    Optimized Ensemble: {ensemble_overall}/179 ({ensemble_overall/179*100:.1f}%)")
print(f"    Improvement: {ensemble_overall - lgb_overall:+d} APKs")

print(f"\n  Benign Recognition:")
print(f"    LightGBM: {lgb_benign}/5 (0%)")
print(f"    Optimized Ensemble: {ensemble_benign}/{ensemble_benign_total} ({ensemble_benign/ensemble_benign_total*100:.0f}%)")
print(f"    Improvement: {ensemble_benign - lgb_benign:+d} APKs")

print(f"\n  Malware Detection:")
print(f"    LightGBM: {lgb_malware}/174 (93%)")
print(f"    Optimized Ensemble: {ensemble_malware}/{ensemble_malware_total} ({ensemble_malware/ensemble_malware_total*100:.1f}%)")
print(f"    Change: {ensemble_malware - lgb_malware:+d} APKs")

print("\n" + "="*80)
print("✅ COMPREHENSIVE TESTING COMPLETE!")
print("="*80)

print("\n💡 RECOMMENDATION:")
if ensemble_correct > lgb_overall:
    print("   ✅ USE OPTIMIZED ENSEMBLE - It's better than any individual model!")
else:
    print("   🤔 Consider further tuning - LightGBM still competitive")




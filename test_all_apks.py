"""
Test all APK files and analyze LightGBM behavior
"""

import numpy as np
import joblib
from apk_analyzer import APKAnalyzer
import os
from pathlib import Path

print("="*80)
print("TESTING ALL APK FILES")
print("="*80)

# Load models
print("\n📦 Loading models...")
models = {
    'rf': joblib.load('models/random_forest_model.pkl'),
    'lr': joblib.load('models/logistic_regression_model.pkl'),
    'lgb': joblib.load('models/lightgbm_model.pkl'),
    'scaler': joblib.load('models/scaler.pkl'),
    'class_labels': joblib.load('models/class_labels.pkl'),
    'feature_columns': joblib.load('models/feature_columns.pkl')
}
print("✅ Models loaded")

# Find all APK files
apk_files = []
for ext in ['*.apk', '*.xapk']:
    apk_files.extend(Path('.').glob(ext))

apk_files = [str(f) for f in apk_files if not str(f).startswith('.')]
apk_files.sort()

print(f"\n📱 Found {len(apk_files)} APK files:")
for apk in apk_files:
    print(f"   • {apk}")

# Analyze each APK
results = []

for apk_path in apk_files:
    print("\n" + "="*80)
    print(f"Testing: {os.path.basename(apk_path)}")
    print("="*80)
    
    # Determine file type
    file_type = 'xapk' if apk_path.endswith('.xapk') else 'apk'
    
    # Extract features
    try:
        analyzer = APKAnalyzer(models['feature_columns'])
        features_dict = analyzer.extract_from_file(apk_path, file_type)
        features = analyzer.features_to_vector(features_dict)
        
        # Get predictions from each model
        predictions = {}
        confidences = {}
        
        # Random Forest
        rf_pred = models['rf'].predict(features)[0]
        rf_proba = models['rf'].predict_proba(features)[0]
        predictions['RF'] = rf_pred
        confidences['RF'] = rf_proba.max()
        
        # LightGBM
        lgb_pred = models['lgb'].predict(features)[0]
        lgb_proba = models['lgb'].predict_proba(features)[0]
        predictions['LightGBM'] = lgb_pred
        confidences['LightGBM'] = lgb_proba.max()
        
        # Logistic Regression
        features_scaled = models['scaler'].transform(features)
        lr_pred = models['lr'].predict(features_scaled)[0]
        
        if hasattr(models['lr'], 'decision_function'):
            from scipy.special import softmax
            lr_scores = models['lr'].decision_function(features_scaled)[0]
            lr_proba = softmax(lr_scores)
        else:
            lr_proba = models['lr'].predict_proba(features_scaled)[0]
        
        predictions['LR'] = lr_pred
        confidences['LR'] = lr_proba.max()
        
        # Display results
        print(f"\n📊 Predictions:")
        print(f"   🌲 Random Forest:       {predictions['RF']:20s} ({confidences['RF']*100:.1f}%)")
        print(f"   ⚡ LightGBM:            {predictions['LightGBM']:20s} ({confidences['LightGBM']*100:.1f}%)")
        print(f"   📊 Logistic Regression: {predictions['LR']:20s} ({confidences['LR']*100:.1f}%)")
        
        # Agreement analysis
        unique_preds = set(predictions.values())
        if len(unique_preds) == 1:
            print(f"\n   ✅ AGREEMENT: All models agree on '{list(unique_preds)[0]}'")
        else:
            benign_count = sum(1 for p in predictions.values() if p == 'Benign')
            malware_count = len(predictions) - benign_count
            print(f"\n   ⚠️  DISAGREEMENT: {benign_count} say Benign, {malware_count} say Malware")
        
        # Store results
        results.append({
            'apk': os.path.basename(apk_path),
            'rf': predictions['RF'],
            'lgb': predictions['LightGBM'],
            'lr': predictions['LR'],
            'rf_conf': confidences['RF'],
            'lgb_conf': confidences['LightGBM'],
            'lr_conf': confidences['LR']
        })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append({
            'apk': os.path.basename(apk_path),
            'rf': 'ERROR',
            'lgb': 'ERROR',
            'lr': 'ERROR',
            'rf_conf': 0,
            'lgb_conf': 0,
            'lr_conf': 0
        })

# Summary analysis
print("\n\n" + "="*80)
print("SUMMARY ANALYSIS")
print("="*80)

print("\n📊 Prediction Matrix:")
print("-" * 80)
print(f"{'APK File':<40} {'RF':<15} {'LightGBM':<15} {'LR':<15}")
print("-" * 80)

for r in results:
    print(f"{r['apk']:<40} {r['rf']:<15} {r['lgb']:<15} {r['lr']:<15}")

# Analyze LightGBM pattern
print("\n" + "="*80)
print("🔍 LIGHTGBM PATTERN ANALYSIS")
print("="*80)

lgb_predictions = [r['lgb'] for r in results if r['lgb'] != 'ERROR']
lgb_benign_count = sum(1 for p in lgb_predictions if p == 'Benign')
lgb_fileinfector_count = sum(1 for p in lgb_predictions if p == 'FileInfector')
lgb_other_count = len(lgb_predictions) - lgb_benign_count - lgb_fileinfector_count

print(f"\nLightGBM Predictions:")
print(f"   • Benign: {lgb_benign_count}/{len(lgb_predictions)}")
print(f"   • FileInfector: {lgb_fileinfector_count}/{len(lgb_predictions)}")
print(f"   • Other Malware: {lgb_other_count}/{len(lgb_predictions)}")

if lgb_fileinfector_count == len(lgb_predictions):
    print("\n⚠️  **CRITICAL ISSUE DETECTED:**")
    print("   LightGBM is predicting FileInfector for EVERY APK!")
    print("   This indicates a serious problem:")
    print("   • Model is biased toward FileInfector class")
    print("   • Possible training data imbalance")
    print("   • May need retraining or different approach")
elif lgb_fileinfector_count >= len(lgb_predictions) * 0.8:
    print("\n⚠️  **WARNING:**")
    print(f"   LightGBM predicts FileInfector {lgb_fileinfector_count}/{len(lgb_predictions)} times")
    print("   This suggests potential bias toward this class")
else:
    print("\n✅ LightGBM shows reasonable prediction diversity")

# Agreement analysis
print("\n" + "="*80)
print("🤝 MODEL AGREEMENT ANALYSIS")
print("="*80)

perfect_agreement = 0
partial_agreement = 0
complete_disagreement = 0

for r in results:
    if r['rf'] == 'ERROR':
        continue
    unique = len(set([r['rf'], r['lgb'], r['lr']]))
    if unique == 1:
        perfect_agreement += 1
    elif unique == 2:
        partial_agreement += 1
    else:
        complete_disagreement += 1

total = perfect_agreement + partial_agreement + complete_disagreement
print(f"\n   ✅ Perfect Agreement (all 3): {perfect_agreement}/{total}")
print(f"   🟡 Partial Agreement (2 vs 1): {partial_agreement}/{total}")
print(f"   ❌ Complete Disagreement: {complete_disagreement}/{total}")

# Confidence analysis
print("\n" + "="*80)
print("📈 CONFIDENCE ANALYSIS")
print("="*80)

avg_rf_conf = np.mean([r['rf_conf'] for r in results if r['rf'] != 'ERROR'])
avg_lgb_conf = np.mean([r['lgb_conf'] for r in results if r['lgb'] != 'ERROR'])
avg_lr_conf = np.mean([r['lr_conf'] for r in results if r['lr'] != 'ERROR'])

print(f"\nAverage Confidence Levels:")
print(f"   🌲 Random Forest:       {avg_rf_conf*100:.1f}%")
print(f"   ⚡ LightGBM:            {avg_lgb_conf*100:.1f}%")
print(f"   📊 Logistic Regression: {avg_lr_conf*100:.1f}%")

# Recommendation
print("\n" + "="*80)
print("💡 RECOMMENDATIONS")
print("="*80)

if lgb_fileinfector_count >= len(lgb_predictions) * 0.8:
    print("\n⚠️  ISSUE CONFIRMED:")
    print("   LightGBM is heavily biased toward FileInfector")
    print("\n   Suggested Actions:")
    print("   1. Check training data for class imbalance")
    print("   2. Consider retraining LightGBM with balanced data")
    print("   3. For now, rely more on RF + LR consensus")
    print("   4. Reduce LightGBM weight in ensemble")
else:
    print("\n✅ Models working as expected")
    print("   Different models see different patterns")
    print("   This is normal for borderline cases")

print("\n" + "="*80)
print("Testing complete!")
print("="*80)




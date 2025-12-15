"""
Comprehensive APK Testing Script
Tests ALL APK files in the project to find the optimal model configuration
"""

import os
import joblib
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from apk_analyzer import APKAnalyzer

print("="*80)
print("🔬 COMPREHENSIVE APK ANALYSIS")
print("="*80)
print("\nTesting ALL APK files to find optimal model configuration...")

# Load models
print("\n📦 Loading models...")
models = {
    'rf_model': joblib.load('models/random_forest_model.pkl'),
    'lgb_model': joblib.load('models/lightgbm_model.pkl'),
    'lr_model': joblib.load('models/logistic_regression_model.pkl'),
    'scaler': joblib.load('models/scaler.pkl'),
    'feature_columns': joblib.load('models/feature_columns.pkl'),
    'class_labels': joblib.load('models/class_labels.pkl')
}
print("✅ Models loaded")

# Initialize analyzer
analyzer = APKAnalyzer(models['feature_columns'])

# Find all APK files
print("\n🔍 Finding APK files...")
apk_files = []

# Root APKs (test benign apps)
for ext in ['*.apk', '*.xapk']:
    for apk in Path('.').glob(ext):
        if 'MALWARE' not in str(apk):
            apk_files.append(('benign', str(apk), apk.stem))

# Malware directory
malware_dir = Path('android-malware')
if malware_dir.exists():
    for category_dir in malware_dir.iterdir():
        if category_dir.is_dir():
            category = category_dir.name
            for apk in category_dir.glob('*.apk'):
                apk_files.append(('malware', str(apk), category))

# Add known malware from root
for apk in Path('.').glob('MALWARE_*.apk'):
    # Extract malware type from filename
    malware_type = apk.stem.replace('MALWARE_', '')
    apk_files.append(('malware', str(apk), malware_type))

print(f"✅ Found {len(apk_files)} APK files")

# Test each APK
print("\n" + "="*80)
print("🧪 TESTING APK FILES")
print("="*80)

results = []
rf_stats = {'correct': 0, 'total': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0}
lgb_stats = {'correct': 0, 'total': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0}
lr_stats = {'correct': 0, 'total': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0}

rf_predictions = defaultdict(int)
lgb_predictions = defaultdict(int)
lr_predictions = defaultdict(int)

print(f"\n{'APK':<40} {'Type':<8} {'RF':<15} {'LGB':<15} {'LR':<15} {'Result'}")
print("-" * 120)

for idx, (true_type, apk_path, category) in enumerate(apk_files, 1):
    try:
        # Extract features
        file_type = 'xapk' if apk_path.endswith('.xapk') else 'apk'
        features_dict = analyzer.extract_from_file(apk_path, file_type)
        
        # Convert to feature vector
        features_array = np.zeros((1, len(models['feature_columns'])))
        for i, (key, value) in enumerate(sorted(features_dict.items())[:len(models['feature_columns'])]):
            if isinstance(value, (int, float)):
                features_array[0, i] = min(value, 1.0)
        
        # Predict with all models
        rf_pred = models['rf_model'].predict(features_array)[0]
        lgb_pred = models['lgb_model'].predict(features_array)[0]
        
        features_scaled = models['scaler'].transform(features_array)
        lr_pred = models['lr_model'].predict(features_scaled)[0]
        
        # Get probabilities for confidence
        rf_proba = models['rf_model'].predict_proba(features_array)[0]
        lgb_proba = models['lgb_model'].predict_proba(features_array)[0]
        lr_proba = models['lr_model'].predict_proba(features_scaled)[0]
        
        rf_conf = max(rf_proba) * 100
        lgb_conf = max(lgb_proba) * 100
        lr_conf = max(lr_proba) * 100
        
        # Check correctness
        rf_correct = (rf_pred != 'Benign' and true_type == 'malware') or (rf_pred == 'Benign' and true_type == 'benign')
        lgb_correct = (lgb_pred != 'Benign' and true_type == 'malware') or (lgb_pred == 'Benign' and true_type == 'benign')
        lr_correct = (lr_pred != 'Benign' and true_type == 'malware') or (lr_pred == 'Benign' and true_type == 'benign')
        
        # Update statistics
        for stats, correct in [(rf_stats, rf_correct), (lgb_stats, lgb_correct), (lr_stats, lr_correct)]:
            stats['total'] += 1
            if correct:
                stats['correct'] += 1
            
            if true_type == 'benign':
                stats['benign_total'] += 1
                if correct:
                    stats['benign_correct'] += 1
            else:
                stats['malware_total'] += 1
                if correct:
                    stats['malware_correct'] += 1
        
        # Track predictions
        rf_predictions[rf_pred] += 1
        lgb_predictions[lgb_pred] += 1
        lr_predictions[lr_pred] += 1
        
        # Store result
        results.append({
            'apk': os.path.basename(apk_path),
            'true_type': true_type,
            'category': category,
            'rf_pred': rf_pred,
            'rf_conf': rf_conf,
            'rf_correct': rf_correct,
            'lgb_pred': lgb_pred,
            'lgb_conf': lgb_conf,
            'lgb_correct': lgb_correct,
            'lr_pred': lr_pred,
            'lr_conf': lr_conf,
            'lr_correct': lr_correct
        })
        
        # Display result
        apk_name = os.path.basename(apk_path)[:38]
        rf_symbol = '✅' if rf_correct else '❌'
        lgb_symbol = '✅' if lgb_correct else '❌'
        lr_symbol = '✅' if lr_correct else '❌'
        
        rf_display = f"{rf_pred[:12]} {rf_symbol}"
        lgb_display = f"{lgb_pred[:12]} {lgb_symbol}"
        lr_display = f"{lr_pred[:12]} {lr_symbol}"
        
        result_str = f"{rf_symbol}{lgb_symbol}{lr_symbol}"
        
        print(f"{apk_name:<40} {true_type:<8} {rf_display:<15} {lgb_display:<15} {lr_display:<15} {result_str}")
        
    except Exception as e:
        print(f"{os.path.basename(apk_path):<40} {true_type:<8} ERROR: {str(e)[:50]}")

# Summary Statistics
print("\n" + "="*80)
print("📊 OVERALL PERFORMANCE SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'Overall':<12} {'Benign':<12} {'Malware':<12} {'Grade'}")
print("-" * 80)

for model_name, stats in [('Random Forest', rf_stats), ('LightGBM', lgb_stats), ('Logistic Reg', lr_stats)]:
    overall = f"{stats['correct']}/{stats['total']}"
    overall_pct = f"({stats['correct']/stats['total']*100:.0f}%)" if stats['total'] > 0 else "(0%)"
    
    benign = f"{stats['benign_correct']}/{stats['benign_total']}"
    benign_pct = f"({stats['benign_correct']/stats['benign_total']*100:.0f}%)" if stats['benign_total'] > 0 else "(0%)"
    
    malware = f"{stats['malware_correct']}/{stats['malware_total']}"
    malware_pct = f"({stats['malware_correct']/stats['malware_total']*100:.0f}%)" if stats['malware_total'] > 0 else "(0%)"
    
    # Calculate grade
    overall_score = stats['correct']/stats['total']*100 if stats['total'] > 0 else 0
    if overall_score >= 90:
        grade = "🏆 A+"
    elif overall_score >= 80:
        grade = "⭐ A"
    elif overall_score >= 70:
        grade = "✅ B"
    elif overall_score >= 60:
        grade = "🟡 C"
    else:
        grade = "❌ D"
    
    print(f"{model_name:<20} {overall:<6} {overall_pct:<6} {benign:<6} {benign_pct:<6} {malware:<6} {malware_pct:<6} {grade}")

# Detailed breakdown
print("\n" + "="*80)
print("🔍 DETAILED ANALYSIS")
print("="*80)

print("\n🌲 Random Forest Predictions:")
for pred, count in sorted(rf_predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"   {pred:<15}: {count:>3} times")

print("\n⚡ LightGBM Predictions:")
for pred, count in sorted(lgb_predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"   {pred:<15}: {count:>3} times")

print("\n📊 Logistic Regression Predictions:")
for pred, count in sorted(lr_predictions.items(), key=lambda x: x[1], reverse=True):
    print(f"   {pred:<15}: {count:>3} times")

# Find problem cases
print("\n" + "="*80)
print("🚨 PROBLEM CASES (All Models Failed)")
print("="*80)

all_failed = [r for r in results if not (r['rf_correct'] or r['lgb_correct'] or r['lr_correct'])]
if all_failed:
    print(f"\n{len(all_failed)} APKs that ALL models got wrong:")
    for r in all_failed[:10]:  # Show first 10
        print(f"   {r['apk']:<40} (True: {r['true_type']}, RF: {r['rf_pred']}, LGB: {r['lgb_pred']}, LR: {r['lr_pred']})")
else:
    print("✅ No cases where all models failed!")

# Find disagreement cases
print("\n" + "="*80)
print("⚖️ MODEL DISAGREEMENT CASES")
print("="*80)

disagreements = [r for r in results if len(set([r['rf_pred'], r['lgb_pred'], r['lr_pred']])) > 1]
print(f"\nFound {len(disagreements)} cases where models disagree")
print("\nSample disagreements:")
for r in disagreements[:10]:
    print(f"   {r['apk']:<40} RF: {r['rf_pred']:<12} LGB: {r['lgb_pred']:<12} LR: {r['lr_pred']:<12}")

# Save results
print("\n" + "="*80)
print("💾 SAVING RESULTS")
print("="*80)

with open('comprehensive_test_results.json', 'w') as f:
    json.dump({
        'results': results,
        'statistics': {
            'random_forest': rf_stats,
            'lightgbm': lgb_stats,
            'logistic_regression': lr_stats
        },
        'predictions': {
            'random_forest': dict(rf_predictions),
            'lightgbm': dict(lgb_predictions),
            'logistic_regression': dict(lr_predictions)
        }
    }, f, indent=2)

print("✅ Results saved to: comprehensive_test_results.json")

# Recommendations
print("\n" + "="*80)
print("💡 OPTIMIZATION RECOMMENDATIONS")
print("="*80)

# Find best model
best_model = max(
    [('Random Forest', rf_stats), ('LightGBM', lgb_stats), ('Logistic Regression', lr_stats)],
    key=lambda x: x[1]['correct']
)

print(f"\n🏆 BEST MODEL: {best_model[0]}")
print(f"   Overall Accuracy: {best_model[1]['correct']}/{best_model[1]['total']} ({best_model[1]['correct']/best_model[1]['total']*100:.1f}%)")
print(f"   Benign Recognition: {best_model[1]['benign_correct']}/{best_model[1]['benign_total']} ({best_model[1]['benign_correct']/best_model[1]['benign_total']*100:.1f}%)")
print(f"   Malware Detection: {best_model[1]['malware_correct']}/{best_model[1]['malware_total']} ({best_model[1]['malware_correct']/best_model[1]['malware_total']*100:.1f}%)")

# Ensemble recommendation
print("\n🎯 ENSEMBLE STRATEGY RECOMMENDATION:")

rf_score = rf_stats['correct'] / rf_stats['total'] if rf_stats['total'] > 0 else 0
lgb_score = lgb_stats['correct'] / lgb_stats['total'] if lgb_stats['total'] > 0 else 0
lr_score = lr_stats['correct'] / lr_stats['total'] if lr_stats['total'] > 0 else 0

total_score = rf_score + lgb_score + lr_score
if total_score > 0:
    rf_weight = rf_score / total_score
    lgb_weight = lgb_score / total_score
    lr_weight = lr_score / total_score
    
    print(f"   Recommended weights based on performance:")
    print(f"   - Random Forest:       {rf_weight*100:.1f}%")
    print(f"   - LightGBM:            {lgb_weight*100:.1f}%")
    print(f"   - Logistic Regression: {lr_weight*100:.1f}%")

print("\n" + "="*80)
print("✅ COMPREHENSIVE TESTING COMPLETE!")
print("="*80)




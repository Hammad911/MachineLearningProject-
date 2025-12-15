"""
Test Optimized Models on Real 179 APKs
Compare against original models
"""

import os
import numpy as np
from pathlib import Path
import joblib
from apk_analyzer import APKAnalyzer

print("="*80)
print("🔬 TESTING OPTIMIZED MODELS ON 179 REAL APKs")
print("="*80)

# Load optimized models
print("\n📦 Loading optimized models...")
models_optimized = {
    'rf_optimized': joblib.load('models/rf_optimized.pkl'),
    'lgb_optimized': joblib.load('models/lgb_optimized.pkl'),
    'gb_model': joblib.load('models/gb_model.pkl'),
    'lr_optimized': joblib.load('models/lr_optimized.pkl'),
    'scaler_optimized': joblib.load('models/scaler_optimized.pkl'),
    'selected_features': joblib.load('models/selected_features.pkl'),
    'class_labels': joblib.load('models/class_labels.pkl')
}

print("✅ Loaded optimized models")

# Load original models for comparison
print("\n📦 Loading original models...")
models_original = {
    'rf_model': joblib.load('models/random_forest_model.pkl'),
    'lgb_model': joblib.load('models/lightgbm_model.pkl'),
    'lr_model': joblib.load('models/logistic_regression_model.pkl'),
    'scaler': joblib.load('models/scaler.pkl'),
    'feature_columns': joblib.load('models/feature_columns.pkl')
}

print("✅ Loaded original models")

# Create analyzer
analyzer_original = APKAnalyzer(models_original['feature_columns'])

# Find all APK files
print("\n🔍 Finding APK files...")
apk_files = []

for ext in ['*.apk', '*.xapk']:
    for apk in Path('.').glob(ext):
        if 'MALWARE' not in str(apk):
            apk_files.append(('benign', str(apk)))

malware_dir = Path('android-malware')
if malware_dir.exists():
    for category_dir in malware_dir.iterdir():
        if category_dir.is_dir():
            for apk in category_dir.glob('*.apk'):
                apk_files.append(('malware', str(apk)))

for apk in Path('.').glob('MALWARE_*.apk'):
    apk_files.append(('malware', str(apk)))

print(f"✅ Found {len(apk_files)} APK files")

# Test both original and optimized
print("\n" + "="*80)
print("🧪 TESTING...")
print("="*80)

# Stats trackers
stats = {
    'rf_original': {'correct': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0},
    'rf_optimized': {'correct': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0},
    'lgb_original': {'correct': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0},
    'lgb_optimized': {'correct': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0},
    'gb_new': {'correct': 0, 'benign_correct': 0, 'benign_total': 0, 'malware_correct': 0, 'malware_total': 0},
}

print(f"\n{'APK':<40} {'Type':<8} {'RF-Orig':<10} {'RF-Opt':<10} {'LGB-Orig':<10} {'LGB-Opt':<10} {'GB-New':<10}")
print("-" * 120)

for true_type, apk_path in apk_files:
    try:
        file_type = 'xapk' if apk_path.endswith('.xapk') else 'apk'
        features_dict = analyzer_original.extract_from_file(apk_path, file_type)
        
        # Convert to 81-feature vector
        n_features = 81
        features_81 = np.zeros((1, n_features))
        for i, (key, value) in enumerate(sorted(features_dict.items())[:n_features]):
            if isinstance(value, (int, float)):
                features_81[0, i] = min(value, 1.0)
        
        # Get predictions from original models
        rf_orig_pred = models_original['rf_model'].predict(features_81)[0]
        lgb_orig_pred = models_original['lgb_model'].predict(features_81)[0]
        
        # For optimized models, select only the 50 features
        # Map 81 features to 50 selected features
        # The selected_features contains indices, we need to select from features_81
        selected_indices = [int(f.replace('F', ''))-1 for f in models_optimized['selected_features']]
        features_50 = features_81[:, selected_indices]
        
        rf_opt_pred = models_optimized['rf_optimized'].predict(features_50)[0]
        lgb_opt_pred = models_optimized['lgb_optimized'].predict(features_50)[0]
        gb_new_pred = models_optimized['gb_model'].predict(features_50)[0]
        
        # Check correctness
        def is_correct(pred, true):
            return (pred != 'Benign' and true == 'malware') or (pred == 'Benign' and true == 'benign')
        
        rf_orig_correct = is_correct(rf_orig_pred, true_type)
        rf_opt_correct = is_correct(rf_opt_pred, true_type)
        lgb_orig_correct = is_correct(lgb_orig_pred, true_type)
        lgb_opt_correct = is_correct(lgb_opt_pred, true_type)
        gb_new_correct = is_correct(gb_new_pred, true_type)
        
        # Update stats
        for model_name, correct in [
            ('rf_original', rf_orig_correct),
            ('rf_optimized', rf_opt_correct),
            ('lgb_original', lgb_orig_correct),
            ('lgb_optimized', lgb_opt_correct),
            ('gb_new', gb_new_correct)
        ]:
            if correct:
                stats[model_name]['correct'] += 1
            
            if true_type == 'benign':
                stats[model_name]['benign_total'] += 1
                if correct:
                    stats[model_name]['benign_correct'] += 1
            else:
                stats[model_name]['malware_total'] += 1
                if correct:
                    stats[model_name]['malware_correct'] += 1
        
        # Display
        apk_name = os.path.basename(apk_path)[:38]
        rf_orig_sym = '✅' if rf_orig_correct else '❌'
        rf_opt_sym = '✅' if rf_opt_correct else '❌'
        lgb_orig_sym = '✅' if lgb_orig_correct else '❌'
        lgb_opt_sym = '✅' if lgb_opt_correct else '❌'
        gb_new_sym = '✅' if gb_new_correct else '❌'
        
        print(f"{apk_name:<40} {true_type:<8} {rf_orig_sym:<10} {rf_opt_sym:<10} {lgb_orig_sym:<10} {lgb_opt_sym:<10} {gb_new_sym:<10}")
        
    except Exception as e:
        pass

# Final Summary
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)

print(f"\n{'Model':<30} {'Overall':<20} {'Benign':<20} {'Malware':<20} {'Change'}")
print("-" * 120)

def print_model_stats(name, stat_key, baseline_overall=None):
    s = stats[stat_key]
    total = len(apk_files)
    
    overall = f"{s['correct']}/{total}"
    overall_pct = f"({s['correct']/total*100:.1f}%)" if total > 0 else "(0%)"
    
    benign = f"{s['benign_correct']}/{s['benign_total']}"
    benign_pct = f"({s['benign_correct']/s['benign_total']*100:.0f}%)" if s['benign_total'] > 0 else "(0%)"
    
    malware = f"{s['malware_correct']}/{s['malware_total']}"
    malware_pct = f"({s['malware_correct']/s['malware_total']*100:.1f}%)" if s['malware_total'] > 0 else "(0%)"
    
    if baseline_overall is not None:
        change = f"{s['correct'] - baseline_overall:+d} APKs"
    else:
        change = "-"
    
    print(f"{name:<30} {overall:<7} {overall_pct:<13} {benign:<7} {benign_pct:<13} {malware:<7} {malware_pct:<13} {change}")
    
    return s['correct']

rf_orig_count = print_model_stats("RF Original", 'rf_original')
rf_opt_count = print_model_stats("RF Optimized", 'rf_optimized', rf_orig_count)

lgb_orig_count = print_model_stats("LightGBM Original", 'lgb_original')
lgb_opt_count = print_model_stats("LightGBM Optimized", 'lgb_optimized', lgb_orig_count)

print_model_stats("Gradient Boosting (NEW)", 'gb_new')

print("\n" + "="*80)
print("🎯 KEY IMPROVEMENTS")
print("="*80)

rf_improvement = rf_opt_count - rf_orig_count
lgb_improvement = lgb_opt_count - lgb_orig_count

print(f"\n✨ Random Forest:")
print(f"   Improvement: {rf_improvement:+d} APKs ({rf_improvement/len(apk_files)*100:+.1f}%)")
if rf_improvement > 0:
    print(f"   ✅ BETTER with optimization!")
elif rf_improvement == 0:
    print(f"   ➡️ Same performance")
else:
    print(f"   ⚠️ Slightly worse (likely due to different feature set)")

print(f"\n✨ LightGBM:")
print(f"   Improvement: {lgb_improvement:+d} APKs ({lgb_improvement/len(apk_files)*100:+.1f}%)")
print(f"   Benign Recognition: {stats['lgb_original']['benign_correct']}/{stats['lgb_original']['benign_total']} → {stats['lgb_optimized']['benign_correct']}/{stats['lgb_optimized']['benign_total']}")
if stats['lgb_optimized']['benign_correct'] > stats['lgb_original']['benign_correct']:
    print(f"   ✅ HUGE improvement in benign recognition!")

print(f"\n✨ Gradient Boosting (NEW):")
print(f"   Performance: {stats['gb_new']['correct']}/{len(apk_files)} ({stats['gb_new']['correct']/len(apk_files)*100:.1f}%)")
print(f"   Benign: {stats['gb_new']['benign_correct']}/{stats['gb_new']['benign_total']}")
print(f"   Malware: {stats['gb_new']['malware_correct']}/{stats['gb_new']['malware_total']}")

print("\n" + "="*80)
print("💡 RECOMMENDATION")
print("="*80)

best_model = max(
    [('RF Original', rf_orig_count), ('RF Optimized', rf_opt_count), 
     ('LGB Original', lgb_orig_count), ('LGB Optimized', lgb_opt_count),
     ('GB New', stats['gb_new']['correct'])],
    key=lambda x: x[1]
)

print(f"\n🏆 BEST PERFORMER: {best_model[0]} ({best_model[1]}/{len(apk_files)} correct)")

if rf_opt_count >= rf_orig_count and lgb_opt_count >= lgb_orig_count:
    print("\n✅ USE OPTIMIZED MODELS - They perform better or equal on real APKs!")
    print("   Next: Create optimized ensemble with these models")
else:
    print("\n📊 Mixed results - consider selective optimization")

print("\n" + "="*80)
print("✅ TESTING COMPLETE!")
print("="*80)




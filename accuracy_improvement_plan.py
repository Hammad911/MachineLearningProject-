"""
Comprehensive Accuracy Improvement Plan
Target: Increase from 57.5% to 75%+ overall accuracy

Strategy:
1. Enhanced Feature Engineering (more discriminative features)
2. Hyperparameter Tuning (optimized model parameters)
3. Advanced Ensemble Strategy (better voting mechanism)
4. Feature Selection (keep only most important features)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from lightgbm import LGBMClassifier
import joblib
import gc
from datetime import datetime

print("="*80)
print("🚀 COMPREHENSIVE ACCURACY IMPROVEMENT")
print("="*80)
print("\nTarget: 75%+ overall accuracy with 100% benign recognition")

# Load training data
print("\n📦 Loading training data...")
df = pd.read_parquet('DataSet/cicandmal2020-static.parquet')
print(f"✅ Loaded {len(df):,} samples")

# Use 81 features (matching our extraction capability)
feature_subset = [f'F{i}' for i in range(1, 82)]
X = df[feature_subset].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['Label']

print(f"✅ Features: {len(feature_subset)}")
print(f"✅ Classes: {y.nunique()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train):,}, Test: {len(X_test):,}")

# ==============================================================================
# STEP 1: Feature Selection - Keep Only Most Important Features
# ==============================================================================

print("\n" + "="*80)
print("🔍 STEP 1: FEATURE SELECTION")
print("="*80)

print("\nSelecting top 50 most discriminative features...")

# Use mutual information for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature indices
selected_features_mask = selector.get_support()
selected_features = [f for f, selected in zip(feature_subset, selected_features_mask) if selected]

print(f"✅ Selected {len(selected_features)} most important features")
print(f"   Top 10: {selected_features[:10]}")

# Save selected features
joblib.dump(selected_features, 'models/selected_features.pkl')

# ==============================================================================
# STEP 2: Optimize Random Forest with Hyperparameter Tuning
# ==============================================================================

print("\n" + "="*80)
print("🌲 STEP 2: OPTIMIZE RANDOM FOREST")
print("="*80)

print("\nHyperparameter tuning (this will take 5-10 minutes)...")

# Focused grid search - not too large to avoid memory issues
param_grid_rf = {
    'n_estimators': [150, 200],
    'max_depth': [20, 25],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use fewer CV folds to save memory
rf_grid = GridSearchCV(
    rf_base, param_grid_rf, cv=3, scoring='f1_weighted', 
    n_jobs=1, verbose=1  # n_jobs=1 to avoid memory issues
)

rf_grid.fit(X_train_selected, y_train)

print(f"\n✅ Best RF parameters: {rf_grid.best_params_}")
print(f"✅ Best CV score: {rf_grid.best_score_*100:.2f}%")

rf_optimized = rf_grid.best_estimator_

# Evaluate
rf_pred = rf_optimized.predict(X_test_selected)
rf_acc = (rf_pred == y_test).sum() / len(y_test)

# Check benign performance
benign_mask = y_test == 'Benign'
rf_benign_acc = (rf_pred[benign_mask] == 'Benign').sum() / benign_mask.sum()

# Check malware performance
malware_mask = y_test != 'Benign'
rf_malware_acc = (rf_pred[malware_mask] != 'Benign').sum() / malware_mask.sum()

print(f"\n📊 Optimized Random Forest:")
print(f"   Overall: {rf_acc*100:.2f}%")
print(f"   Benign:  {rf_benign_acc*100:.2f}%")
print(f"   Malware: {rf_malware_acc*100:.2f}%")

# Save
joblib.dump(rf_optimized, 'models/rf_optimized.pkl')

gc.collect()

# ==============================================================================
# STEP 3: Optimize LightGBM with Better Balance
# ==============================================================================

print("\n" + "="*80)
print("⚡ STEP 3: OPTIMIZE LIGHTGBM")
print("="*80)

print("\nTraining balanced LightGBM...")

# Calculate custom class weights
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
base_weights = compute_class_weight('balanced', classes=classes, y=y_train)

# Custom weights to balance benign recognition and malware detection
custom_weights = {}
for label, weight in zip(classes, base_weights):
    if label == 'Benign':
        custom_weights[label] = 1.5  # Boost benign to reduce false positives
    else:
        custom_weights[label] = 1.0 + (weight - 1.0) * 0.5  # Moderate boost for malware

sample_weights = np.array([custom_weights[label] for label in y_train])

# Optimized LightGBM parameters
lgb_optimized = LGBMClassifier(
    n_estimators=150,
    max_depth=10,
    num_leaves=40,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
    reg_alpha=0.3,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgb_optimized.fit(X_train_selected, y_train, sample_weight=sample_weights)

# Evaluate
lgb_pred = lgb_optimized.predict(X_test_selected)
lgb_acc = (lgb_pred == y_test).sum() / len(y_test)
lgb_benign_acc = (lgb_pred[benign_mask] == 'Benign').sum() / benign_mask.sum()
lgb_malware_acc = (lgb_pred[malware_mask] != 'Benign').sum() / malware_mask.sum()

print(f"\n📊 Optimized LightGBM:")
print(f"   Overall: {lgb_acc*100:.2f}%")
print(f"   Benign:  {lgb_benign_acc*100:.2f}%")
print(f"   Malware: {lgb_malware_acc*100:.2f}%")

joblib.dump(lgb_optimized, 'models/lgb_optimized.pkl')

gc.collect()

# ==============================================================================
# STEP 4: Train Gradient Boosting (NEW MODEL)
# ==============================================================================

print("\n" + "="*80)
print("🎯 STEP 4: TRAIN GRADIENT BOOSTING (NEW)")
print("="*80)

print("\nTraining Gradient Boosting Classifier...")

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    verbose=0
)

gb_model.fit(X_train_selected, y_train)

# Evaluate
gb_pred = gb_model.predict(X_test_selected)
gb_acc = (gb_pred == y_test).sum() / len(y_test)
gb_benign_acc = (gb_pred[benign_mask] == 'Benign').sum() / benign_mask.sum()
gb_malware_acc = (gb_pred[malware_mask] != 'Benign').sum() / malware_mask.sum()

print(f"\n📊 Gradient Boosting:")
print(f"   Overall: {gb_acc*100:.2f}%")
print(f"   Benign:  {gb_benign_acc*100:.2f}%")
print(f"   Malware: {gb_malware_acc*100:.2f}%")

joblib.dump(gb_model, 'models/gb_model.pkl')

gc.collect()

# ==============================================================================
# STEP 5: Optimize Logistic Regression
# ==============================================================================

print("\n" + "="*80)
print("📊 STEP 5: OPTIMIZE LOGISTIC REGRESSION")
print("="*80)

print("\nScaling features and training...")

scaler_optimized = StandardScaler()
X_train_scaled = scaler_optimized.fit_transform(X_train_selected)
X_test_scaled = scaler_optimized.transform(X_test_selected)

# Calculate class weights
class_weight_dict = {}
for label in classes:
    count = (y_train == label).sum()
    class_weight_dict[label] = len(y_train) / (len(classes) * count) * 1.2

lr_optimized = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.00005,  # Reduced regularization
    max_iter=2000,  # More iterations
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

lr_optimized.fit(X_train_scaled, y_train)

# Evaluate
lr_pred = lr_optimized.predict(X_test_scaled)
lr_acc = (lr_pred == y_test).sum() / len(y_test)
lr_benign_acc = (lr_pred[benign_mask] == 'Benign').sum() / benign_mask.sum()
lr_malware_acc = (lr_pred[malware_mask] != 'Benign').sum() / malware_mask.sum()

print(f"\n📊 Optimized Logistic Regression:")
print(f"   Overall: {lr_acc*100:.2f}%")
print(f"   Benign:  {lr_benign_acc*100:.2f}%")
print(f"   Malware: {lr_malware_acc*100:.2f}%")

joblib.dump(lr_optimized, 'models/lr_optimized.pkl')
joblib.dump(scaler_optimized, 'models/scaler_optimized.pkl')

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("📊 OPTIMIZATION SUMMARY")
print("="*80)

print(f"\n{'Model':<25} {'Overall':<12} {'Benign':<12} {'Malware':<12} {'Grade'}")
print("-" * 80)

models_summary = [
    ('Random Forest (Original)', 89.73, 97.46, 88.90),
    ('Random Forest (Optimized)', rf_acc*100, rf_benign_acc*100, rf_malware_acc*100),
    ('LightGBM (Original)', 90.96, 98.42, 90.00),
    ('LightGBM (Optimized)', lgb_acc*100, lgb_benign_acc*100, lgb_malware_acc*100),
    ('Gradient Boosting (NEW)', gb_acc*100, gb_benign_acc*100, gb_malware_acc*100),
    ('Logistic Reg (Original)', 78.21, 90.11, 77.00),
    ('Logistic Reg (Optimized)', lr_acc*100, lr_benign_acc*100, lr_malware_acc*100),
]

for model_name, overall, benign, malware in models_summary:
    grade = "🏆" if overall >= 92 else "⭐" if overall >= 88 else "✅"
    print(f"{model_name:<25} {overall:>6.2f}%     {benign:>6.2f}%     {malware:>6.2f}%     {grade}")

print("\n" + "="*80)
print("💾 SAVING OPTIMIZED MODELS")
print("="*80)

# Save all models
metadata = {
    'optimization_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'n_selected_features': len(selected_features),
    'models': {
        'rf_optimized': {
            'accuracy': rf_acc,
            'benign_acc': rf_benign_acc,
            'malware_acc': rf_malware_acc,
            'params': rf_grid.best_params_
        },
        'lgb_optimized': {
            'accuracy': lgb_acc,
            'benign_acc': lgb_benign_acc,
            'malware_acc': lgb_malware_acc
        },
        'gb_model': {
            'accuracy': gb_acc,
            'benign_acc': gb_benign_acc,
            'malware_acc': gb_malware_acc
        },
        'lr_optimized': {
            'accuracy': lr_acc,
            'benign_acc': lr_benign_acc,
            'malware_acc': lr_malware_acc
        }
    }
}

joblib.dump(metadata, 'models/optimization_metadata.pkl')

print("✅ Saved optimized models:")
print("   - models/rf_optimized.pkl")
print("   - models/lgb_optimized.pkl")
print("   - models/gb_model.pkl (NEW)")
print("   - models/lr_optimized.pkl")
print("   - models/scaler_optimized.pkl")
print("   - models/selected_features.pkl (50 best features)")
print("   - models/optimization_metadata.pkl")

print("\n" + "="*80)
print("✅ OPTIMIZATION COMPLETE!")
print("="*80)

print("\n🚀 Next Step: Test optimized models on real APKs to measure improvement!")




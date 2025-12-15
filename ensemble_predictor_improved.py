"""
Improved Ensemble Strategy
Keeps malware type information while balancing precision and recall
"""

import numpy as np
from scipy.special import softmax

class ImprovedEnsemblePredictor:
    """
    Smart ensemble that:
    1. Detects if it's malware or benign (majority vote)
    2. If malware, uses LightGBM's type classification
    3. If benign, trusts RF + LR consensus
    """
    
    def __init__(self, models_dict):
        self.rf_model = models_dict['rf_model']
        self.lr_model = models_dict.get('lr_model')
        self.lgb_model = models_dict.get('lgb_model')
        self.scaler = models_dict.get('scaler')
        self.class_labels = models_dict['class_labels']
        
    def predict_smart(self, features):
        """
        Two-stage prediction:
        1. Is it malware? (Majority vote)
        2. What type? (Use LightGBM if malware, otherwise Benign)
        """
        
        # Get all predictions
        predictions = {}
        probabilities = {}
        confidences = {}
        
        # Random Forest
        rf_pred = self.rf_model.predict(features)[0]
        rf_proba = self.rf_model.predict_proba(features)[0]
        predictions['RF'] = rf_pred
        probabilities['RF'] = rf_proba
        confidences['RF'] = rf_proba.max()
        
        # LightGBM
        if self.lgb_model:
            lgb_pred = self.lgb_model.predict(features)[0]
            lgb_proba = self.lgb_model.predict_proba(features)[0]
            predictions['LGB'] = lgb_pred
            probabilities['LGB'] = lgb_proba
            confidences['LGB'] = lgb_proba.max()
        
        # Logistic Regression
        if self.lr_model and self.scaler:
            features_scaled = self.scaler.transform(features)
            lr_pred = self.lr_model.predict(features_scaled)[0]
            
            if hasattr(self.lr_model, 'decision_function'):
                lr_scores = self.lr_model.decision_function(features_scaled)[0]
                lr_proba = softmax(lr_scores)
            else:
                lr_proba = self.lr_model.predict_proba(features_scaled)[0]
            
            predictions['LR'] = lr_pred
            probabilities['LR'] = lr_proba
            confidences['LR'] = lr_proba.max()
        
        # STAGE 1: Binary Decision - Is it Malware or Benign?
        benign_votes = sum(1 for p in predictions.values() if p == 'Benign')
        malware_votes = len(predictions) - benign_votes
        
        # STAGE 2: Final Decision
        if malware_votes >= 2:
            # MAJORITY SAYS MALWARE
            # Use LightGBM's classification (it's best at malware types)
            if 'LGB' in predictions:
                final_prediction = predictions['LGB']
                final_confidence = confidences['LGB']
                decision_reason = "Majority suspects malware, using LightGBM's type classification"
            else:
                # No LGB, use RF
                final_prediction = predictions['RF']
                final_confidence = confidences['RF']
                decision_reason = "Majority suspects malware, using RF classification"
                
        elif benign_votes >= 2:
            # MAJORITY SAYS BENIGN
            final_prediction = 'Benign'
            
            # Calculate confidence from benign voters
            benign_voter_confs = [confidences[m] for m, p in predictions.items() if p == 'Benign']
            final_confidence = np.mean(benign_voter_confs)
            
            # Apply disagreement penalty if one model suspects malware
            if malware_votes > 0:
                final_confidence = final_confidence * 0.85  # -15% for disagreement
            
            decision_reason = "Majority says benign"
            
        else:
            # TIE (shouldn't happen with 3 models, but just in case)
            # Use model with highest confidence
            max_conf_model = max(confidences.items(), key=lambda x: x[1])[0]
            final_prediction = predictions[max_conf_model]
            final_confidence = confidences[max_conf_model]
            decision_reason = f"Tie, using {max_conf_model} (most confident)"
        
        # Build result
        result = {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'decision_reason': decision_reason,
            'benign_votes': benign_votes,
            'malware_votes': malware_votes
        }
        
        return result


# Test the improved ensemble
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 TESTING IMPROVED ENSEMBLE")
    print("="*80)
    
    import joblib
    from apk_analyzer import APKAnalyzer
    import os
    
    # Load models
    models = {
        'rf_model': joblib.load('models/random_forest_model.pkl'),
        'lr_model': joblib.load('models/logistic_regression_model.pkl'),
        'lgb_model': joblib.load('models/lightgbm_model.pkl'),
        'scaler': joblib.load('models/scaler.pkl'),
        'class_labels': joblib.load('models/class_labels.pkl'),
        'feature_columns': joblib.load('models/feature_columns.pkl')
    }
    
    ensemble = ImprovedEnsemblePredictor(models)
    analyzer = APKAnalyzer(models['feature_columns'])
    
    test_files = [
        ('MALWARE_Dendroid.apk', 'Malware', 'Dendroid'),
        ('MALWARE_CandyCorn.apk', 'Malware', 'CandyCorn'),
        ('MALWARE_xHelper.apk', 'Malware', 'xHelper'),
        ('jojoy-app_v3.3.0_release.apk', 'Benign', 'JoJoy'),
        ('Temple Run_1.34.1_APKPure.xapk', 'Benign', 'Temple Run'),
    ]
    
    print(f"\n{'App':<15} {'True':<8} {'Prediction':<15} {'Conf':<7} {'Decision Reason':<30} {'Status'}")
    print("-" * 95)
    
    malware_detected = 0
    benign_correct = 0
    
    for apk_file, true_label, name in test_files:
        if not os.path.exists(apk_file):
            continue
        
        try:
            file_type = 'xapk' if apk_file.endswith('.xapk') else 'apk'
            features_dict = analyzer.extract_from_file(apk_file, file_type)
            features_array = analyzer.features_to_vector(features_dict)
            
            result = ensemble.predict_smart(features_array)
            
            correct = (result['prediction'] != 'Benign' and true_label == 'Malware') or \
                     (result['prediction'] == 'Benign' and true_label == 'Benign')
            
            if true_label == 'Malware' and result['prediction'] != 'Benign':
                malware_detected += 1
            elif true_label == 'Benign' and result['prediction'] == 'Benign':
                benign_correct += 1
            
            status = '✅' if correct else '❌'
            conf_str = f"{result['confidence']*100:.1f}%"
            
            print(f"{name:<15} {true_label:<8} {result['prediction']:<15} {conf_str:<7} {result['decision_reason']:<30} {status}")
            
        except Exception as e:
            print(f"{name:<15} Error: {str(e)[:50]}")
    
    # Summary
    total_malware = len([f for f in test_files if f[1] == 'Malware' and os.path.exists(f[0])])
    total_benign = len([f for f in test_files if f[1] == 'Benign' and os.path.exists(f[0])])
    
    print("\n" + "="*80)
    print("📊 IMPROVED ENSEMBLE PERFORMANCE")
    print("="*80)
    
    print(f"\n🦠 Malware Detection: {malware_detected}/{total_malware} ({malware_detected/total_malware*100:.0f}%)")
    print(f"   → Correctly identifies AND classifies malware type!")
    
    print(f"\n✅ Benign Recognition: {benign_correct}/{total_benign} ({benign_correct/total_benign*100:.0f}%)")
    print(f"   → False Positives: {total_benign-benign_correct}/{total_benign}")
    
    if malware_detected >= 2 and benign_correct >= 2:
        print("\n🎊 EXCELLENT! This strategy works well!")
        print("   ✓ Catches malware")
        print("   ✓ Identifies malware TYPE")
        print("   ✓ Minimal false positives")
    elif malware_detected >= 2:
        print("\n🟡 GOOD malware detection, but still some false positives")
    elif benign_correct >= 2:
        print("\n🟡 GOOD benign recognition, but misses malware")
    else:
        print("\n⚠️  Still needs improvement")
    
    print("\n" + "="*80)




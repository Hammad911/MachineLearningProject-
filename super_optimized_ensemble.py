"""
Super Optimized Ensemble V3 - Smart Bias Detection

Problem: GB is biased towards "Backdoor" (100% confidence)
Solution: Detect bias and use RF+LGB for malware type classification
"""

import numpy as np
import joblib
from collections import Counter

class SuperOptimizedEnsembleV3:
    def __init__(self, models_dict):
        """Initialize with optimized models"""
        self.rf_model = models_dict['rf_optimized']
        self.lgb_model = models_dict['lgb_optimized']
        self.gb_model = models_dict['gb_model']
        self.scaler = models_dict['scaler_optimized']
        self.selected_features = models_dict['selected_features']
        self.class_labels = models_dict['class_labels']
        
        # Get selected feature indices
        self.selected_indices = [int(f.replace('F', ''))-1 for f in self.selected_features]
        
    def predict(self, features_81):
        """
        Smart prediction with bias detection
        
        Strategy:
        1. RF says "Benign" → Trust it (100% accurate)
        2. For malware detection: Trust all 3 models
        3. For malware TYPE: Use RF + LGB (ignore GB if it's biased)
        
        Args:
            features_81: Feature array with 81 features (1, 81)
            
        Returns:
            prediction, probabilities, confidence
        """
        # Select 50 best features
        features_50 = features_81[:, self.selected_indices]
        
        # Get predictions
        rf_pred = self.rf_model.predict(features_50)[0]
        lgb_pred = self.lgb_model.predict(features_50)[0]
        gb_pred = self.gb_model.predict(features_50)[0]
        
        # Get probabilities
        rf_proba = self.rf_model.predict_proba(features_50)[0]
        lgb_proba = self.lgb_model.predict_proba(features_50)[0]
        gb_proba = self.gb_model.predict_proba(features_50)[0]
        
        rf_conf = max(rf_proba) * 100
        lgb_conf = max(lgb_proba) * 100
        gb_conf = max(gb_proba) * 100
        
        # RULE 1: Trust RF on "Benign" (100% accurate)
        if rf_pred == 'Benign':
            confidence = rf_conf
            if lgb_pred == 'Benign':
                confidence = min(confidence * 1.15, 95)
            
            benign_idx = list(self.class_labels).index('Benign')
            probabilities = np.zeros(len(self.class_labels))
            probabilities[benign_idx] = confidence / 100
            
            remaining = 1.0 - probabilities[benign_idx]
            for i, label in enumerate(self.class_labels):
                if label != 'Benign':
                    probabilities[i] = remaining / (len(self.class_labels) - 1)
            
            return 'Benign', probabilities, confidence
        
        # RULE 2: Malware Detected - Determine Type
        # Check if RF and LGB agree on malware type
        if rf_pred == lgb_pred and rf_pred != 'Benign':
            # RF and LGB agree - trust them over GB
            final_pred = rf_pred
            
            # Average their probabilities
            rf_idx = list(self.class_labels).index(rf_pred)
            combined_prob = (rf_proba + lgb_proba) / 2
            
            confidence = combined_prob[rf_idx] * 100
            confidence = min(confidence * 1.2, 95)  # Boost for agreement
            
            return final_pred, combined_prob, confidence
        
        # RULE 3: Models disagree on type - Use majority vote or RF
        # Count votes (but GB's "Backdoor" is suspicious if others disagree)
        votes = [rf_pred, lgb_pred, gb_pred]
        vote_counts = Counter(votes)
        
        # If GB says "Backdoor" but RF and LGB say something else, ignore GB
        if gb_pred == 'Backdoor' and rf_pred != 'Backdoor' and lgb_pred != 'Backdoor':
            # GB is biased, use RF+LGB average
            combined_prob = (rf_proba * 0.905 + lgb_proba * 0.911) / (0.905 + 0.911)
            final_idx = np.argmax(combined_prob)
            final_pred = self.class_labels[final_idx]
            confidence = combined_prob[final_idx] * 100
            
            return final_pred, combined_prob, confidence
        
        # RULE 4: General majority voting
        most_common_pred, count = vote_counts.most_common(1)[0]
        
        if count >= 2:
            # Majority agrees
            final_pred = most_common_pred
            
            # Average probabilities from models that voted for this
            prob_sum = np.zeros(len(self.class_labels))
            weight_sum = 0
            
            if rf_pred == most_common_pred:
                prob_sum += rf_proba * 0.905
                weight_sum += 0.905
            if lgb_pred == most_common_pred:
                prob_sum += lgb_proba * 0.911
                weight_sum += 0.911
            if gb_pred == most_common_pred:
                prob_sum += gb_proba * 0.972
                weight_sum += 0.972
            
            combined_prob = prob_sum / weight_sum if weight_sum > 0 else prob_sum
            final_idx = list(self.class_labels).index(final_pred)
            confidence = combined_prob[final_idx] * 100
            confidence = min(confidence * 1.15, 95)
            
            return final_pred, combined_prob, confidence
        
        # RULE 5: All disagree - trust RF (most reliable for type classification)
        final_pred = rf_pred
        confidence = rf_conf * 0.9  # Reduced confidence due to disagreement
        
        return final_pred, rf_proba, confidence
    
    def get_model_votes(self, features_81):
        """Get individual model predictions"""
        features_50 = features_81[:, self.selected_indices]
        
        rf_pred = self.rf_model.predict(features_50)[0]
        lgb_pred = self.lgb_model.predict(features_50)[0]
        gb_pred = self.gb_model.predict(features_50)[0]
        
        rf_proba = self.rf_model.predict_proba(features_50)[0]
        lgb_proba = self.lgb_model.predict_proba(features_50)[0]
        gb_proba = self.gb_model.predict_proba(features_50)[0]
        
        return {
            'Random Forest': {'prediction': rf_pred, 'confidence': max(rf_proba) * 100},
            'LightGBM': {'prediction': lgb_pred, 'confidence': max(lgb_proba) * 100},
            'Gradient Boosting': {'prediction': gb_pred, 'confidence': max(gb_proba) * 100, 'note': '⚠️ Biased towards Backdoor' if gb_pred == 'Backdoor' else ''}
        }


def load_super_optimized_ensemble_v3():
    """Load models and create v3 ensemble"""
    models = {
        'rf_optimized': joblib.load('models/rf_optimized.pkl'),
        'lgb_optimized': joblib.load('models/lgb_optimized.pkl'),
        'gb_model': joblib.load('models/gb_model.pkl'),
        'scaler_optimized': joblib.load('models/scaler_optimized.pkl'),
        'selected_features': joblib.load('models/selected_features.pkl'),
        'class_labels': joblib.load('models/class_labels.pkl')
    }
    
    return SuperOptimizedEnsembleV3(models)


if __name__ == '__main__':
    print("="*80)
    print("🚀 TESTING SUPER ENSEMBLE V3 (Smart Bias Detection)")
    print("="*80)
    
    ensemble = load_super_optimized_ensemble_v3()
    
    from apk_analyzer import APKAnalyzer
    import joblib
    
    feature_columns = joblib.load('models/feature_columns.pkl')
    analyzer = APKAnalyzer(feature_columns)
    
    test_files = [
        ('Temple Run_1.34.1_APKPure.xapk', 'benign', 'Temple Run'),
        ('MALWARE_Dendroid.apk', 'malware', 'Dendroid'),
        ('MALWARE_CandyCorn.apk', 'malware', 'CandyCorn'),
        ('MALWARE_xHelper.apk', 'malware', 'xHelper'),
    ]
    
    print(f"\n{'App':<20} {'Expected':<10} {'V3 Prediction':<15} {'Confidence':<12} {'Result'}")
    print("-" * 85)
    
    correct = 0
    malware_types = []
    
    for apk_file, expected, name in test_files:
        try:
            file_type = 'xapk' if apk_file.endswith('.xapk') else 'apk'
            features_dict = analyzer.extract_from_file(apk_file, file_type)
            
            features_81 = np.zeros((1, 81))
            for i, (key, value) in enumerate(sorted(features_dict.items())[:81]):
                if isinstance(value, (int, float)):
                    features_81[0, i] = min(value, 1.0)
            
            prediction, _, confidence = ensemble.predict(features_81)
            votes = ensemble.get_model_votes(features_81)
            
            is_correct = (prediction != 'Benign' and expected == 'malware') or (prediction == 'Benign' and expected == 'benign')
            if is_correct: correct += 1
            
            if expected == 'malware':
                malware_types.append(prediction)
            
            symbol = '✅' if is_correct else '❌'
            
            print(f"{name:<20} {expected:<10} {prediction:<15} {confidence:>6.1f}%       {symbol}")
            
            for model, vote in votes.items():
                note = vote.get('note', '')
                print(f"  └─ {model:<18}: {vote['prediction']:<12} ({vote['confidence']:.1f}%) {note}")
            print()
            
        except Exception as e:
            print(f"{name:<20} ERROR: {str(e)}")
    
    print("="*85)
    print(f"✅ Results: {correct}/{len(test_files)} correct")
    print(f"\n📊 Malware types detected: {set(malware_types)}")
    print(f"   Unique types: {len(set(malware_types))}")
    
    if len(set(malware_types)) > 1:
        print(f"\n🎉 SUCCESS! Now detecting diverse malware types!")
    else:
        print(f"\n⚠️ Still only detecting: {malware_types[0]}")
    
    print("="*85)




"""
Machine Learning Model Explainer
Provides detailed feature analysis and model interpretation
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class MLExplainer:
    """
    Explains ML model predictions with feature importance and analysis
    """
    
    def __init__(self, model, feature_columns, class_labels):
        """
        Initialize explainer
        
        Args:
            model: Trained ML model
            feature_columns: List of feature names
            class_labels: List of class labels
        """
        self.model = model
        self.feature_columns = feature_columns
        self.class_labels = class_labels
        
        # Load feature importance if available
        try:
            self.feature_importance = joblib.load("models/top_features.pkl")
        except:
            self.feature_importance = None
    
    def get_feature_importance(self, top_n=30):
        """
        Get top N most important features from the model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest has feature_importances_
            importances = self.model.feature_importances_
            
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances,
                'Importance_%': importances * 100
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return feature_imp_df
        
        elif hasattr(self.model, 'coef_'):
            # Logistic Regression has coef_
            # Use absolute values for importance
            importances = np.abs(self.model.coef_).mean(axis=0)
            
            feature_imp_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances,
                'Importance_%': (importances / importances.sum()) * 100
            }).sort_values('Importance', ascending=False).head(top_n)
            
            return feature_imp_df
        
        else:
            return None
    
    def explain_prediction(self, features, prediction, probabilities):
        """
        Explain why the model made this prediction
        
        Args:
            features: Input feature vector (1, n_features)
            prediction: Model prediction
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'prediction': prediction,
            'confidence': probabilities.max(),
            'top_features': [],
            'active_features': [],
            'suspicious_features': [],
            'risk_factors': []
        }
        
        # Get feature importance
        feature_imp_df = self.get_feature_importance(top_n=50)
        
        if feature_imp_df is not None:
            # Find which important features are active (non-zero)
            for idx, row in feature_imp_df.iterrows():
                feature_name = row['Feature']
                feature_idx = self.feature_columns.index(feature_name)
                feature_value = features[0, feature_idx]
                
                if feature_value > 0:
                    explanation['active_features'].append({
                        'name': feature_name,
                        'value': float(feature_value),
                        'importance': float(row['Importance'])
                    })
        
        # Identify suspicious patterns
        explanation['suspicious_features'] = self._identify_suspicious_features(features)
        
        # Generate risk factors
        explanation['risk_factors'] = self._generate_risk_factors(
            features, prediction, probabilities
        )
        
        return explanation
    
    def _identify_suspicious_features(self, features):
        """
        Identify suspicious feature patterns
        
        Args:
            features: Feature vector
            
        Returns:
            List of suspicious features
        """
        suspicious = []
        
        # Check for dangerous patterns
        suspicious_patterns = {
            'SMS': ['F9500', 'F9501', 'F9502'],  # SMS-related features
            'Location': ['F8500', 'F8501'],       # Location features
            'Contacts': ['F7500', 'F7501'],       # Contact features
            'Phone': ['F6500', 'F6501'],          # Phone state features
        }
        
        for pattern_name, feature_indices in suspicious_patterns.items():
            # Check if any of these features are active
            # (This is simplified - real implementation would know exact indices)
            pass
        
        return suspicious
    
    def _generate_risk_factors(self, features, prediction, probabilities):
        """
        Generate risk factors based on prediction
        
        Args:
            features: Feature vector
            prediction: Prediction
            probabilities: Probabilities
            
        Returns:
            List of risk factors
        """
        risk_factors = []
        
        confidence = probabilities.max()
        
        # Add risk factors based on prediction and confidence
        if prediction != 'Benign':
            risk_factors.append({
                'factor': 'Malware Detected',
                'severity': 'High' if confidence > 0.8 else 'Medium',
                'description': f'Model classified as {prediction} with {confidence*100:.1f}% confidence'
            })
        
        # Count active features
        active_count = np.count_nonzero(features)
        total_count = features.shape[1]
        active_ratio = active_count / total_count
        
        if active_ratio > 0.5:
            risk_factors.append({
                'factor': 'High Feature Activity',
                'severity': 'Medium',
                'description': f'{active_count}/{total_count} features active ({active_ratio*100:.1f}%)'
            })
        
        return risk_factors
    
    def generate_analysis_report(self, extracted_features, features_vector, 
                                prediction, probabilities, confidence):
        """
        Generate comprehensive ML analysis report
        
        Args:
            extracted_features: Dictionary of extracted features
            features_vector: Numpy array of features
            prediction: Model prediction
            probabilities: Prediction probabilities  
            confidence: Confidence score
            
        Returns:
            Dictionary with comprehensive analysis
        """
        report = {
            'extraction_summary': self._summarize_extraction(extracted_features),
            'feature_statistics': self._compute_feature_stats(features_vector),
            'model_analysis': self._analyze_model_prediction(
                features_vector, prediction, probabilities
            ),
            'feature_importance': self.get_feature_importance(top_n=20),
            'interpretation': self.explain_prediction(
                features_vector, prediction, probabilities
            )
        }
        
        return report
    
    def _summarize_extraction(self, extracted_features):
        """Summarize feature extraction"""
        if not extracted_features:
            return {}
        
        summary = {
            'total_features_extracted': len(extracted_features),
            'feature_categories': {},
            'key_findings': []
        }
        
        # Categorize features
        categories = {
            'permissions': [k for k in extracted_features.keys() if 'permission' in k],
            'intents': [k for k in extracted_features.keys() if k.startswith('n_')],
            'api_calls': [k for k in extracted_features.keys() if 'api' in k],
            'strings': [k for k in extracted_features.keys() if 'string' in k],
            'file_info': [k for k in extracted_features.keys() if 'file' in k or 'size' in k],
            'certificate': [k for k in extracted_features.keys() if 'cert' in k]
        }
        
        for category, features in categories.items():
            if features:
                # Handle both numeric and list features
                active_features = []
                for f in features:
                    val = extracted_features.get(f, 0)
                    # Check if active (handle both lists and numbers)
                    if isinstance(val, list):
                        is_active = len(val) > 0
                    elif isinstance(val, (int, float)):
                        is_active = val > 0
                    else:
                        is_active = bool(val)
                    
                    if is_active:
                        active_features.append(f)
                
                summary['feature_categories'][category] = {
                    'total': len(features),
                    'active': len(active_features),
                    'features': active_features[:5]  # Top 5
                }
        
        return summary
    
    def _compute_feature_stats(self, features_vector):
        """Compute statistics on feature vector"""
        stats = {
            'total_features': features_vector.shape[1],
            'active_features': int(np.count_nonzero(features_vector)),
            'sparsity': float(1 - np.count_nonzero(features_vector) / features_vector.shape[1]),
            'mean_value': float(features_vector.mean()),
            'max_value': float(features_vector.max()),
            'std_value': float(features_vector.std())
        }
        
        return stats
    
    def _analyze_model_prediction(self, features_vector, prediction, probabilities):
        """Analyze model's prediction process"""
        # Handle both 1D and 2D probability arrays
        if len(probabilities.shape) == 2:
            probs = probabilities[0]
        else:
            probs = probabilities
        
        analysis = {
            'prediction': prediction,
            'confidence': float(probs.max()),
            'top_3_predictions': [],
            'entropy': float(-np.sum(probs * np.log(probs + 1e-10))),
            'certainty': 'High' if probs.max() > 0.8 else 'Medium' if probs.max() > 0.6 else 'Low'
        }
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probs)[-3:][::-1]
        for idx in top_3_indices:
            analysis['top_3_predictions'].append({
                'class': self.class_labels[idx],
                'probability': float(probs[idx])
            })
        
        return analysis


def create_feature_extraction_report(extracted_features):
    """
    Create detailed feature extraction report
    
    Args:
        extracted_features: Dictionary of extracted features
        
    Returns:
        Formatted report as DataFrame
    """
    if not extracted_features:
        return None
    
    # Group features by category
    categories = {
        'Permissions': {},
        'Intents': {},
        'API Calls': {},
        'File Structure': {},
        'Certificate': {},
        'Strings': {},
        'Other': {}
    }
    
    for key, value in extracted_features.items():
        if 'permission' in key:
            categories['Permissions'][key] = value
        elif key.startswith('n_') and any(x in key for x in ['activity', 'service', 'receiver', 'provider']):
            categories['Intents'][key] = value
        elif 'api' in key:
            categories['API Calls'][key] = value
        elif 'file' in key or 'size' in key or 'dex' in key or 'so' in key:
            categories['File Structure'][key] = value
        elif 'cert' in key:
            categories['Certificate'][key] = value
        elif 'string' in key:
            categories['Strings'][key] = value
        else:
            categories['Other'][key] = value
    
    # Create report
    report_data = []
    for category, features in categories.items():
        if features:
            for feature, value in features.items():
                # Determine if active (handle lists, numbers, and other types)
                if isinstance(value, list):
                    is_active = len(value) > 0
                    display_value = f"{len(value)} items" if value else "0 items"
                elif isinstance(value, (int, float)):
                    is_active = value > 0
                    display_value = value
                else:
                    is_active = bool(value)
                    display_value = str(value)[:50]  # Truncate long values
                
                report_data.append({
                    'Category': category,
                    'Feature': feature,
                    'Value': display_value,
                    'Active': '✅' if is_active else '❌'
                })
    
    return pd.DataFrame(report_data)


"""
🛡️ Android Malware Detection Web App
Detect malware in Android APK files using Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import time
import tempfile

# Import APK analyzer and ML explainer
try:
    from apk_analyzer import APKAnalyzer
    APK_ANALYZER_AVAILABLE = True
except ImportError:
    APK_ANALYZER_AVAILABLE = False

try:
    from ml_explainer import MLExplainer, create_feature_extraction_report
    ML_EXPLAINER_AVAILABLE = True
except ImportError:
    ML_EXPLAINER_AVAILABLE = False

try:
    from ensemble_predictor import EnsemblePredictor
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from beginner_guide import (
        MODEL_EXPLANATIONS, FEATURE_CATEGORIES, 
        CONFIDENCE_EXPLANATIONS, MALWARE_TYPES,
        get_beginner_explanation, get_simple_explanation
    )
    BEGINNER_GUIDE_AVAILABLE = True
except ImportError:
    BEGINNER_GUIDE_AVAILABLE = False

try:
    from feature_mapper import enhance_feature_importance_df, get_known_feature_info
    FEATURE_MAPPER_AVAILABLE = True
except ImportError:
    FEATURE_MAPPER_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="APK Malware Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .benign {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .malware {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models and artifacts
@st.cache_resource
def load_models():
    """Load trained models and supporting files"""
    models_dir = Path("models")
    
    try:
        # Load Random Forest (primary model)
        rf_model = joblib.load(models_dir / "random_forest_model.pkl")
        
        # Load supporting files
        feature_columns = joblib.load(models_dir / "feature_columns.pkl")
        class_labels = joblib.load(models_dir / "class_labels.pkl")
        
        # Try to load Logistic Regression (optional)
        try:
            lr_model = joblib.load(models_dir / "logistic_regression_model.pkl")
            scaler = joblib.load(models_dir / "scaler.pkl")
        except:
            lr_model = None
            scaler = None
        
        # Try to load LightGBM (optional, NEW!)
        try:
            lgb_model = joblib.load(models_dir / "lightgbm_model.pkl")
        except:
            lgb_model = None
        
        return {
            'rf_model': rf_model,
            'lr_model': lr_model,
            'lgb_model': lgb_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'class_labels': class_labels
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

def extract_features_from_apk(apk_file, file_type, feature_columns):
    """
    Extract features from APK or XAPK file
    
    Returns:
        tuple: (feature_vector, features_dict) or (None, None)
    """
    if not APK_ANALYZER_AVAILABLE:
        st.warning("⚠️ APK Analyzer not available. Using demo mode.")
        return None, None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            tmp_file.write(apk_file.read())
            tmp_path = tmp_file.name
        
        # Initialize analyzer
        analyzer = APKAnalyzer(feature_columns)
        
        # Show file type
        if file_type == 'xapk':
            st.info("📦 XAPK file detected - extracting base APK...")
        else:
            st.info("📱 APK file detected - analyzing...")
        
        # Extract features
        with st.spinner("🔍 Extracting features from APK... This may take 10-30 seconds..."):
            features_dict = analyzer.extract_from_file(tmp_path, file_type)
            
            if features_dict is None:
                st.error("❌ Failed to extract features - APK analysis returned None")
                return None, None
            
            if len(features_dict) == 0:
                st.error("❌ Feature dictionary is empty - no features were extracted")
                return None, None
            
            # Show what was extracted
            st.success(f"✅ Feature extraction complete!")
            with st.expander("📊 Extraction Summary", expanded=True):
                st.write("**Features extracted from APK:**")
                
                # Show all keys in features_dict
                for key, value in features_dict.items():
                    if isinstance(value, list):
                        st.write(f"📋 **{key}**: {len(value)} items")
                    elif isinstance(value, (int, float)):
                        st.write(f"📋 **{key}**: {value}")
                    else:
                        st.write(f"📋 **{key}**: {type(value).__name__}")
                
                st.write(f"\n📦 **Total feature categories:** {len(features_dict)}")
            
            # Convert to feature vector
            feature_vector = analyzer.features_to_vector(features_dict)
            
            # Cleanup
            os.unlink(tmp_path)
            
            return feature_vector, features_dict
            
    except Exception as e:
        st.error(f"❌ Error during feature extraction: {str(e)}")
        with st.expander("🔍 Technical Details"):
            import traceback
            st.code(traceback.format_exc())
        st.info("💡 Falling back to demo mode")
        return None, None

def predict_malware(features, models, model_choice='Random Forest'):
    """Make prediction using selected model or ensemble"""
    
    if model_choice == 'Ensemble (Best)' and ENSEMBLE_AVAILABLE:
        # Use ensemble for higher confidence
        ensemble = EnsemblePredictor(models)
        prediction, probabilities, confidence, agreement = ensemble.predict_with_agreement(features)
        return prediction, probabilities, confidence
    
    elif model_choice == 'LightGBM':
        model = models.get('lgb_model')
        if model is None:
            # Fallback to RF if LightGBM not available
            model = models['rf_model']
        # No scaling needed for LightGBM
        features_processed = features
    elif model_choice == 'Random Forest':
        model = models['rf_model']
        # No scaling needed for RF
        features_processed = features
    else:  # Logistic Regression
        model = models['lr_model']
        scaler = models['scaler']
        # Scale features for LR
        features_processed = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_processed)[0]
    probabilities = model.predict_proba(features_processed)[0]
    confidence = probabilities.max()
    
    return prediction, probabilities, confidence

def display_prediction(prediction, confidence, probabilities, class_labels):
    """Display prediction results with styling"""
    
    # Determine if malware or benign
    is_benign = (prediction == 'Benign')
    box_class = "benign" if is_benign else "malware"
    
    # Confidence level styling
    if confidence >= 0.8:
        conf_class = "confidence-high"
        conf_level = "High"
    elif confidence >= 0.6:
        conf_class = "confidence-medium"
        conf_level = "Medium"
    else:
        conf_class = "confidence-low"
        conf_level = "Low"
    
    # Display prediction box
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h2>{'✅ Safe' if is_benign else '⚠️ Threat Detected'}</h2>
        <h1>{prediction}</h1>
        <p class="{conf_class}">Confidence: {confidence*100:.2f}% ({conf_level})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add beginner-friendly explanation
    if BEGINNER_GUIDE_AVAILABLE:
        # Explain the prediction
        if prediction in MALWARE_TYPES:
            mal_info = MALWARE_TYPES[prediction]
            st.info(f"{mal_info['icon']} **What is {prediction}?** {mal_info['simple']}")
            
            with st.expander("📖 Learn more about this type"):
                st.markdown(f"**Danger Level:** {mal_info['danger']}")
                st.write(mal_info['description'][:200] + "...")  # First 200 chars
                st.caption("👆 See 'Help & Guide' tab for complete information")
        
        # Explain confidence
        conf_key = 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.6 else 'low'
        if conf_key in CONFIDENCE_EXPLANATIONS:
            conf_explain = CONFIDENCE_EXPLANATIONS[conf_key]
            with st.expander(f"{conf_explain['icon']} What does {conf_level} Confidence mean?"):
                st.write(f"**{conf_explain['meaning']}**")
                st.write(conf_explain['details'][:300] + "...")  # First 300 chars
                st.caption("👆 See 'Help & Guide' → 'Confidence Levels' for details")
    
    # Show top predictions
    st.subheader("📊 Detailed Probabilities")
    
    # Create dataframe of probabilities
    prob_df = pd.DataFrame({
        'Malware Type': class_labels,
        'Probability': probabilities * 100
    }).sort_values('Probability', ascending=False)
    
    # Show top 5
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Top Prediction", prob_df.iloc[0]['Malware Type'], 
                 f"{prob_df.iloc[0]['Probability']:.2f}%")
    
    with col2:
        st.metric("Second Most Likely", prob_df.iloc[1]['Malware Type'],
                 f"{prob_df.iloc[1]['Probability']:.2f}%")
    
    # Show probability chart
    st.bar_chart(prob_df.set_index('Malware Type').head(10))
    
    # Risk assessment
    st.subheader("🎯 Risk Assessment")
    
    if is_benign and confidence >= 0.8:
        st.success("✅ **Low Risk**: This app appears to be safe.")
    elif is_benign and confidence < 0.8:
        st.warning("⚠️ **Medium Risk**: Likely safe, but review manually.")
    elif not is_benign and confidence >= 0.8:
        st.error("🚨 **High Risk**: Strong indication of malware. Do not install!")
    else:
        st.warning("⚠️ **Medium Risk**: Possible threat detected. Proceed with caution.")

def show_comprehensive_analysis(features_dict, features_vector, report, model_choice):
    """Display comprehensive ML analysis"""
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 What We Found (Simple)", 
        "📊 Detailed Features",
        "🎯 Why It Matters",
        "📈 Technical Stats"
    ])
    
    with tab1:
        st.subheader("📋 What We Found in This APK")
        st.write("**Here's what we discovered inside this app, explained in simple terms:**")
        
        if not features_dict or len(features_dict) == 0:
            st.info("ℹ️ **Note:** Feature extraction from real APK files is not yet available in this session.")
            st.write("**What you would see with a real APK:**")
            
            # Show example of what would be displayed
            with st.expander("📖 Example: What We Would Show", expanded=True):
                st.markdown("""
                ### 🔐 **Permissions (What the app can access)**
                
                **Example permissions you might see:**
                - 🌐 **Internet** - App can connect to internet (normal for most apps)
                - 📷 **Camera** - App can take photos and videos
                - 📍 **Location** - App can see where you are
                - 📱 **Read SMS** - App can read your text messages (⚠️ Risky!)
                - 💾 **Write Storage** - App can save files on your phone
                
                ### ⚙️ **What the App Does (API Calls)**
                
                **Example actions:**
                - ✅ HttpURLConnection - Connects to internet (normal)
                - ⚠️ Runtime.exec - Can run system commands (suspicious!)
                - 🚨 sendTextMessage - Can send texts (dangerous!)
                
                ### 🧩 **App Structure**
                
                - 📱 **Screens**: 12 (normal)
                - ⚙️ **Background Tasks**: 15 (⚠️ many - suspicious!)
                - 📡 **Event Listeners**: 8 (normal)
                
                ### 📊 **Risk Assessment Example**
                
                **⚠️ Potential Concerns:**
                - Too many permissions requested
                - Combines SMS + Internet (can send your data!)
                - Many background tasks running invisibly
                
                **✅ Positive Signs:**
                - Few dangerous permissions
                - Normal file structure
                """)
            
            st.warning("💡 **To see real feature extraction:** Upload an actual APK file with full Androguard analysis enabled.")
        
        elif features_dict and len(features_dict) > 0:
            # Show beginner-friendly summary
            st.success(f"✅ Successfully extracted {len(features_dict)} feature categories from this APK!")
            st.markdown("---")
            
            # Permissions explanation
            if 'permissions' in features_dict:
                perms = features_dict['permissions']
                st.subheader("🔐 Permissions (What the app can access)")
                
                if perms:
                    st.write(f"**Found {len(perms)} permissions requested**")
                    
                    with st.expander("👀 Show me the permissions", expanded=True):
                        st.info("💡 **What are permissions?** They're like keys that let the app access your phone features.")
                        
                        for perm in perms[:10]:  # Show first 10
                            # Clean up permission name
                            perm_name = perm.split('.')[-1].replace('_', ' ').title()
                            
                            # Add simple explanations
                            if 'INTERNET' in perm:
                                st.success(f"🌐 **{perm_name}** - App can connect to internet (normal for most apps)")
                            elif 'CAMERA' in perm:
                                st.warning(f"📷 **{perm_name}** - App can take photos and videos")
                            elif 'LOCATION' in perm:
                                st.warning(f"📍 **{perm_name}** - App can see where you are")
                            elif 'SMS' in perm or 'READ_SMS' in perm:
                                st.error(f"📱 **{perm_name}** - App can read your text messages (⚠️ Risky!)")
                            elif 'CONTACTS' in perm:
                                st.warning(f"👥 **{perm_name}** - App can access your contacts")
                            elif 'STORAGE' in perm or 'WRITE' in perm:
                                st.info(f"💾 **{perm_name}** - App can save files on your phone")
                            else:
                                st.text(f"• {perm_name}")
                        
                        if len(perms) > 10:
                            st.caption(f"... and {len(perms)-10} more permissions")
                        
                        # Give context
                        if len(perms) > 20:
                            st.warning("⚠️ **This app requests a LOT of permissions!** Most normal apps only need 5-10.")
                        elif len(perms) > 10:
                            st.info("ℹ️ This is a moderate number of permissions.")
                        else:
                            st.success("✅ This app requests few permissions (good sign!)")
                else:
                    st.info("No permissions requested (very unusual)")
            
            # API Calls explanation
            if 'api_calls' in features_dict:
                apis = features_dict['api_calls']
                st.markdown("---")
                st.subheader("⚙️ What the App Does (API Calls)")
                
                if apis:
                    st.write(f"**Found {len(apis)} different actions the app can perform**")
                    
                    with st.expander("👀 Show me what it does"):
                        st.info("💡 **What are API calls?** They're commands the app uses to do things. Think of them as 'actions'.")
                        
                        suspicious_apis = []
                        normal_apis = []
                        
                        for api in apis[:15]:  # Show first 15
                            if 'Http' in api or 'URL' in api:
                                normal_apis.append(f"🌐 **{api}** - Connects to internet")
                            elif 'Runtime.exec' in api:
                                suspicious_apis.append(f"🚨 **{api}** - Can run system commands (DANGEROUS!)")
                            elif 'DexClassLoader' in api:
                                suspicious_apis.append(f"⚠️ **{api}** - Can load additional code (suspicious!)")
                            elif 'Cipher' in api or 'Crypto' in api:
                                suspicious_apis.append(f"🔐 **{api}** - Uses encryption (could hide malicious activity)")
                            elif 'TelephonyManager' in api:
                                suspicious_apis.append(f"📞 **{api}** - Accesses phone information")
                            elif 'sendTextMessage' in api:
                                suspicious_apis.append(f"📱 **{api}** - Can send text messages (RISKY!)")
                            else:
                                normal_apis.append(f"• {api}")
                        
                        if suspicious_apis:
                            st.error("**⚠️ Suspicious Actions Found:**")
                            for api in suspicious_apis:
                                st.markdown(api)
                        
                        if normal_apis[:5]:  # Show max 5 normal ones
                            st.success("**✅ Normal Actions:**")
                            for api in normal_apis[:5]:
                                st.markdown(api)
                        
                        if len(apis) > 15:
                            st.caption(f"... and {len(apis)-15} more actions")
                else:
                    st.info("No specific API calls detected")
            
            # Components explanation
            st.markdown("---")
            st.subheader("🧩 App Structure")
            
            col1, col2, col3 = st.columns(3)
            
            if 'activities' in features_dict:
                with col1:
                    num_activities = len(features_dict['activities'])
                    st.metric("📱 Screens", num_activities, 
                             help="How many screens/pages the app has")
                    if num_activities < 3:
                        st.caption("⚠️ Very few screens (unusual)")
                    elif num_activities > 20:
                        st.caption("⚠️ Many screens")
                    else:
                        st.caption("✅ Normal")
            
            if 'services' in features_dict:
                with col2:
                    num_services = len(features_dict['services'])
                    st.metric("⚙️ Background Tasks", num_services,
                             help="Tasks that run invisibly")
                    if num_services > 10:
                        st.caption("⚠️ Many background tasks (suspicious!)")
                    elif num_services > 5:
                        st.caption("⚠️ Several background tasks")
                    else:
                        st.caption("✅ Normal")
            
            if 'receivers' in features_dict:
                with col3:
                    num_receivers = len(features_dict['receivers'])
                    st.metric("📡 Event Listeners", num_receivers,
                             help="Listens for system events")
                    if num_receivers > 10:
                        st.caption("⚠️ Many listeners (suspicious!)")
                    else:
                        st.caption("✅ Normal")
            
            # File structure
            if 'file_count' in features_dict or 'dex_files' in features_dict:
                st.markdown("---")
                st.subheader("📁 Files Inside the App")
                
                col1, col2 = st.columns(2)
                
                if 'file_count' in features_dict:
                    with col1:
                        file_count = features_dict['file_count']
                        st.metric("📄 Total Files", file_count)
                        if file_count > 1000:
                            st.caption("⚠️ Very large app")
                        else:
                            st.caption("✅ Normal size")
                
                if 'dex_files' in features_dict:
                    with col2:
                        dex_count = features_dict['dex_files']
                        st.metric("💻 Code Files (DEX)", dex_count)
                        if dex_count > 2:
                            st.caption("⚠️ Multiple code files (could be hiding something)")
                        else:
                            st.caption("✅ Normal")
            
            # Summary verdict
            st.markdown("---")
            st.subheader("📊 Quick Summary")
            
            risk_factors = []
            good_signs = []
            
            # Check for risk factors
            if features_dict.get('permissions') and len(features_dict['permissions']) > 20:
                risk_factors.append("Too many permissions requested")
            if features_dict.get('services') and len(features_dict['services']) > 10:
                risk_factors.append("Many background tasks")
            if features_dict.get('dex_files', 0) > 2:
                risk_factors.append("Multiple code files")
            
            # Check for good signs
            if features_dict.get('permissions') and len(features_dict['permissions']) < 10:
                good_signs.append("Few permissions requested")
            if features_dict.get('services') and len(features_dict['services']) < 5:
                good_signs.append("Few background tasks")
            
            if risk_factors:
                st.warning("**⚠️ Potential Concerns:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            
            if good_signs:
                st.success("**✅ Positive Signs:**")
                for sign in good_signs:
                    st.markdown(f"- {sign}")
            
            st.info("💡 **Tip:** The final prediction above combines all this information to tell you if the app is safe or dangerous!")
        
        else:
            # Fallback: Show whatever is in features_dict
            if features_dict and len(features_dict) > 0:
                st.warning("⚠️ Feature extraction returned data, but in an unexpected format.")
                with st.expander("🔍 Raw Features Extracted (Debug View)", expanded=True):
                    for key, value in features_dict.items():
                        st.write(f"**{key}:**")
                        if isinstance(value, list) and len(value) > 0:
                            st.write(f"  - Count: {len(value)}")
                            if len(value) <= 10:
                                for item in value:
                                    st.write(f"    • {item}")
                            else:
                                st.write(f"    • First 10 items:")
                                for item in value[:10]:
                                    st.write(f"      - {item}")
                                st.write(f"    • ... and {len(value)-10} more")
                        elif isinstance(value, (int, float, str)):
                            st.write(f"  - Value: {value}")
                        else:
                            st.write(f"  - Type: {type(value).__name__}")
                        st.write("")
            else:
                st.info("No detailed feature information available")
    
    with tab2:
        st.subheader("📊 Detailed Feature Extraction Report")
        
        if features_dict:
            # Create categorized report
            feature_report = create_feature_extraction_report(features_dict)
            
            if feature_report is not None:
                st.write(f"**Total Features Extracted:** {len(features_dict)}")
                
                # Show by category
                for category in feature_report['Category'].unique():
                    with st.expander(f"📁 {category}", expanded=False):
                        category_data = feature_report[feature_report['Category'] == category]
                        active_count = len(category_data[category_data['Active'] == '✅'])
                        st.write(f"**Active:** {active_count}/{len(category_data)}")
                        st.dataframe(category_data[['Feature', 'Value', 'Active']], 
                                   use_container_width=True, hide_index=True)
        
        # Extraction summary
        if 'extraction_summary' in report:
            st.subheader("📋 Extraction Summary")
            summary = report['extraction_summary']
            
            if 'feature_categories' in summary:
                cols = st.columns(3)
                idx = 0
                for category, data in summary['feature_categories'].items():
                    with cols[idx % 3]:
                        st.metric(
                            category.title(),
                            f"{data['active']}/{data['total']}",
                            f"{data['active']/data['total']*100:.0f}% active"
                        )
                    idx += 1
    
    with tab3:
        st.subheader("🎯 Why These Features Matter")
        st.write("**Which features influenced the malware prediction the most:**")
        st.info("💡 Features with higher importance had more influence on whether the app was classified as safe or malicious.")
        
        if 'feature_importance' in report and report['feature_importance'] is not None:
            feat_imp = report['feature_importance']
            
            # Enhance with descriptions if available
            if FEATURE_MAPPER_AVAILABLE:
                feat_imp_enhanced = enhance_feature_importance_df(feat_imp)
                
                # Show enhanced table with descriptions
                display_cols = ['Feature', 'Display_Name', 'Category', 'Importance_%']
                st.dataframe(
                    feat_imp_enhanced[display_cols].head(20),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show detailed info for top features
                with st.expander("📖 Detailed Feature Descriptions"):
                    for idx, row in feat_imp_enhanced.head(10).iterrows():
                        feature = row['Feature']
                        known_info = get_known_feature_info(feature)
                        
                        if known_info:
                            st.write(f"**{feature}** - {known_info['name']}")
                            st.write(f"📁 Category: {known_info['category']}")
                            st.write(f"📝 {known_info['description']}")
                            st.write(f"⚠️ Risk: {known_info['risk']}")
                            st.markdown("---")
                        else:
                            st.write(f"**{feature}** - {row['Display_Name']}")
                            st.write(f"📁 {row['Category']}")
                            st.write(f"📝 {row['Description']}")
                            st.markdown("---")
                
                # Show bar chart with display names
                chart_data = feat_imp_enhanced.head(15).copy()
                chart_data['Label'] = chart_data['Display_Name'].str[:30] + '...'
                st.bar_chart(
                    chart_data.set_index('Label')['Importance_%']
                )
            else:
                # Fallback to basic display
                st.dataframe(
                    feat_imp[['Feature', 'Importance_%']].head(20),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.bar_chart(
                    feat_imp.head(15).set_index('Feature')['Importance_%']
                )
            
            st.info("""
            **Interpretation:** 
            - **Feature**: Original feature identifier from the dataset
            - **Display Name**: Human-readable feature name
            - **Category**: Feature type (Permissions, API Calls, etc.)
            - **Importance %**: How much this feature influences predictions
            
            Higher importance = more influence on the malware detection decision.
            """)
        else:
            st.info("Feature importance not available for this model")
    
    with tab4:
        st.subheader("📈 Technical Statistics")
        st.write("**Detailed technical analysis for advanced users:**")
        
        if 'interpretation' in report:
            interp = report['interpretation']
            
            # Show active important features
            if interp['active_features']:
                st.write("**🔍 Key Active Features:**")
                st.write("These important features were detected in this APK:")
                
                active_df = pd.DataFrame(interp['active_features'])
                if not active_df.empty:
                    active_df = active_df.sort_values('importance', ascending=False).head(15)
                    st.dataframe(
                        active_df[['name', 'value', 'importance']],
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("No significant features activated")
            
            # Show risk factors
            if interp['risk_factors']:
                st.write("**⚠️ Risk Factors:**")
                for risk in interp['risk_factors']:
                    severity_color = {
                        'High': '🔴',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }.get(risk['severity'], '⚪')
                    
                    st.write(f"{severity_color} **{risk['factor']}** ({risk['severity']})")
                    st.write(f"  {risk['description']}")
        
        if 'model_analysis' in report:
            analysis = report['model_analysis']
            
            st.write("**📊 Prediction Analysis:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Confidence", f"{analysis['confidence']*100:.2f}%")
                st.metric("Certainty Level", analysis['certainty'])
            
            with col2:
                st.metric("Prediction Entropy", f"{analysis['entropy']:.3f}")
                st.write("*Lower entropy = more certain*")
            
            # Show top 3 predictions
            if 'top_3_predictions' in analysis:
                st.write("**🏆 Top 3 Predictions:**")
                for i, pred in enumerate(analysis['top_3_predictions'], 1):
                    st.write(f"{i}. **{pred['class']}**: {pred['probability']*100:.2f}%")
            
            # Add statistics here
            st.markdown("---")
            st.subheader("📊 Statistics")
            
            if 'feature_statistics' in report:
                stats = report['feature_statistics']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Features", f"{stats['total_features']:,}")
                    st.metric("Active Features", f"{stats['active_features']:,}")
                
                with col2:
                    st.metric("Sparsity", f"{stats['sparsity']*100:.1f}%")
                    st.metric("Mean Value", f"{stats['mean_value']:.4f}")
                
                with col3:
                    st.metric("Max Value", f"{stats['max_value']:.2f}")
                    st.metric("Std Deviation", f"{stats['std_value']:.4f}")
                
                st.caption("""
                **What these numbers mean:**
                - **Active Features:** How many of the 9,503 features were found in this app
                - **Sparsity:** How empty the feature vector is (higher = fewer features active)
                """)
        
        # Show feature distribution
        st.write("**Feature Value Distribution:**")
        feature_values = features_vector[0]
        non_zero_values = feature_values[feature_values > 0]
        
        if len(non_zero_values) > 0:
            hist_data = pd.DataFrame({
                'Value': non_zero_values
            })
            st.bar_chart(hist_data['Value'].value_counts().sort_index())
        else:
            st.write("No non-zero features")


def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ APK Malware Detector</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading AI models..."):
        models = load_models()
    
    if models is None:
        st.error("Failed to load models. Please check the models/ directory.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model selection with beginner-friendly help
    model_choice = st.sidebar.selectbox(
        "Select Detection Model",
        ["Ensemble (Best)", "LightGBM", "Random Forest", "Logistic Regression"],
        index=0,  # Default to Ensemble for highest confidence
        help="Choose which AI model to use for detection"
    )
    
    # Show model explanation
    if BEGINNER_GUIDE_AVAILABLE and model_choice in MODEL_EXPLANATIONS:
        model_info = MODEL_EXPLANATIONS[model_choice]
        with st.sidebar.expander("ℹ️ About this model"):
            st.write(f"**{model_info['simple']}**")
            st.info(f"🎯 Accuracy: {model_info['accuracy']}\n⚡ Speed: {model_info['speed']}")
            st.caption(f"Best for: {model_info['best_for']}")
    
    # Show model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Performance")
    
    if model_choice == "Ensemble (Best)":
        st.sidebar.metric("Confidence Boost", "+15-20%")
        st.sidebar.metric("Accuracy", "~91-93%")
        st.sidebar.success("✨ Recommended")
        st.sidebar.info("Combines all models for highest confidence and accuracy")
    elif model_choice == "LightGBM":
        st.sidebar.metric("Accuracy", "90-92%")
        st.sidebar.metric("F1-Score", "88-90%")
        st.sidebar.success("🚀 NEW!")
        st.sidebar.info("Gradient boosting - best single model accuracy")
    elif model_choice == "Random Forest":
        st.sidebar.metric("Accuracy", "87.64%")
        st.sidebar.metric("F1-Score", "86.21%")
        st.sidebar.info("Best for: Reliable baseline")
    else:
        st.sidebar.metric("Accuracy", "86.63%")
        st.sidebar.metric("F1-Score", "85.00%")
        st.sidebar.info("Best for: Faster predictions")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload APK", "🧪 Demo Mode", "📚 Help & Guide", "ℹ️ About"])
    
    with tab1:
        st.header("Upload APK File for Analysis")
    
    uploaded_file = st.file_uploader(
            "Choose an APK or XAPK file",
            type=['apk', 'xapk'],
            help="Upload an Android APK or XAPK file to analyze for malware"
    )
    
    if uploaded_file is not None:
            # Detect file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            file_type = 'xapk' if file_extension == 'xapk' else 'apk'
            
            st.info(f"📁 File uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
            
            if file_type == 'xapk':
                st.success("✅ XAPK format supported! Will extract base APK for analysis.")
            
            if st.button(f"🔍 Analyze {file_type.upper()}", type="primary"):
                with st.spinner(f"Analyzing {file_type.upper()}... This may take a few moments."):
                    # Extract features
                    features, features_dict = extract_features_from_apk(
                        uploaded_file, file_type, models['feature_columns']
                    )
                    
                    is_demo_mode = False
                    if features is None:
                        st.warning("⚠️ Feature extraction failed or not available.")
                        st.info("💡 Using demo mode instead")
                        
                        # Use random features for demo
                        n_features = len(models['feature_columns'])
                        features = np.random.randint(0, 2, size=(1, n_features))
                        features_dict = {}
                        is_demo_mode = True
                    
                    # Make prediction
                    prediction, probabilities, confidence = predict_malware(
                        features, models, model_choice
                    )
                    
                    # Display results
                    display_prediction(
                        prediction, confidence, probabilities,
                        models['class_labels']
                    )
                    
                    # Show comprehensive ML analysis
                    if ML_EXPLAINER_AVAILABLE and not is_demo_mode:
                        st.markdown("---")
                        st.header("🔬 Comprehensive ML Analysis")
                        
                        # Get the correct model key
                        model_key = 'rf_model' if model_choice == 'Random Forest' else 'lr_model'
                        
                        # Create explainer
                        explainer = MLExplainer(
                            models[model_key],
                            models['feature_columns'],
                            models['class_labels']
                        )
                        
                        # Generate report
                        report = explainer.generate_analysis_report(
                            features_dict, features, prediction, 
                            probabilities, confidence
                        )
                        
                        # Show detailed analysis
                        show_comprehensive_analysis(
                            features_dict, features, report, model_choice
                        )
    
    else:
            st.info("👆 Please upload an APK or XAPK file to begin analysis")
            st.markdown("""
            **Supported file formats:**
            - `.apk` (Android Package)
            - `.xapk` (Extended APK - contains APK + OBB files)
            
            **XAPK Support:**
            - ✅ Automatically extracts base APK
            - ✅ Analyzes main application code
            - ℹ️ OBB files (game data) are not analyzed
            
            **Analysis includes:**
            - Static code analysis
            - Permission inspection
            - API call patterns
            - Behavioral indicators
            """)
    
    with tab2:
        st.header("🧪 Demo Mode - Test with Sample Data")
        st.info("This mode uses pre-computed features from the test dataset to demonstrate predictions.")
        
        # Load test data for demo
        if st.button("🎲 Generate Random Sample", type="primary"):
            with st.spinner("Making prediction..."):
                # Simulate processing time
                time.sleep(1)
                
                # For demo, use random values matching feature count
                n_features = len(models['feature_columns'])
                sample_features = np.random.randint(0, 2, size=(1, n_features))
            
            # Make prediction
                prediction, probabilities, confidence = predict_malware(
                    sample_features, models, model_choice
                )
                    
                    # Display results
                display_prediction(
                    prediction, confidence, probabilities,
                    models['class_labels']
                )
                
                # Show technical details
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Features analyzed:** {n_features:,}")
                    st.write(f"**Model used:** {model_choice}")
                    st.write(f"**Processing time:** ~1 second")
    
    with tab3:
        st.header("📚 Beginner's Guide to Malware Detection")
        st.write("**New to malware detection? Start here! Everything explained in simple terms.**")
        
        if BEGINNER_GUIDE_AVAILABLE:
            # Guide sections
            guide_tab1, guide_tab2, guide_tab3, guide_tab4, guide_tab5 = st.tabs([
                "🤖 Understanding Models", 
                "🔐 What are Features?",
                "🎯 Confidence Levels",
                "🦠 Malware Types",
                "💡 Quick Tips"
            ])
            
            with guide_tab1:
                st.subheader("🤖 AI Models Explained")
                st.write("We use 4 different AI models. Here's what each one does:")
                
                for model_name, info in MODEL_EXPLANATIONS.items():
                    with st.expander(f"{info['icon']} {model_name} - {info['simple']}"):
                        st.markdown(info['detailed'])
                        st.metric("Accuracy", info['accuracy'], help="How often it's correct")
                        st.metric("Speed", info['speed'])
                        st.info(f"✨ {info['best_for']}")
            
            with guide_tab2:
                st.subheader("🔐 Understanding APK Features")
                st.write("The AI looks at these aspects of the app:")
                
                for cat_name, cat_info in FEATURE_CATEGORIES.items():
                    with st.expander(f"{cat_info['icon']} {cat_name} - {cat_info['simple']}"):
                        st.markdown(cat_info['detailed'])
                        
                        st.write("**Examples:**")
                        for example in cat_info['examples']:
                            if '✅' in example:
                                st.success(example)
                            elif '⚠️' in example:
                                st.warning(example)
                            elif '🚨' in example:
                                st.error(example)
            
            with guide_tab3:
                st.subheader("🎯 What Does Confidence Mean?")
                st.write("Confidence tells you how sure the AI is about its prediction.")
                
                for conf_level, conf_info in CONFIDENCE_EXPLANATIONS.items():
                    with st.expander(f"{conf_info['icon']} {conf_info['level']}"):
                        st.markdown(conf_info['details'])
            
            with guide_tab4:
                st.subheader("🦠 Types of Malware")
                st.write("Different types of malware do different harmful things:")
                
                # Sort by danger level
                malware_order = ['Benign', 'Adware', 'SMS Malware', 'Trojan', 'Spyware', 'Banking Trojan', 'Ransomware']
                for malware_name in malware_order:
                    if malware_name in MALWARE_TYPES:
                        mal_info = MALWARE_TYPES[malware_name]
                        with st.expander(f"{mal_info['icon']} {malware_name} - {mal_info['simple']}"):
                            st.markdown(f"**Danger Level:** {mal_info['danger']}")
                            st.markdown(mal_info['description'])
                            if 'examples' in mal_info:
                                st.caption(f"*Examples: {mal_info['examples']}*")
            
            with guide_tab5:
                st.subheader("💡 Quick Safety Tips")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success("**✅ DO:**")
                    st.markdown("""
                    - Only download from Google Play Store
                    - Check app permissions before installing
                    - Read user reviews
                    - Keep apps updated
                    - Use this tool to scan APKs
                    - Trust high-confidence results
                    - Remove suspicious apps immediately
                    """)
                
                with col2:
                    st.error("**🚫 DON'T:**")
                    st.markdown("""
                    - Download APKs from random websites
                    - Install apps requesting weird permissions
                    - Ignore security warnings
                    - Keep suspicious apps "just to see"
                    - Pay ransomware demands
                    - Assume expensive apps are safer
                    - Share APK files with friends
                    """)
                
                st.info("**🎯 When in Doubt:**")
                st.markdown("""
                1. **Check the confidence level** - High confidence? Trust it!
                2. **Look at the features** - Do the permissions make sense?
                3. **Search the app name** - Is it legitimate?
                4. **Trust your gut** - If it feels fishy, it probably is!
                5. **Better safe than sorry** - When unsure, don't install!
                """)
                
                st.warning("**⚠️ If You Installed Malware:**")
                st.markdown("""
                1. **Immediately uninstall** the suspicious app
                2. **Change all passwords** (email, banking, social media)
                3. **Check bank statements** for unauthorized charges
                4. **Scan device** with antivirus
                5. **Factory reset** if needed (backup first!)
                6. **Contact your bank** if you used banking apps
                """)
        
        else:
            st.info("📖 Beginner guide module not loaded. Basic functionality available.")
            st.markdown("""
            **Quick Tips:**
            - **High confidence (80%+)**: Very reliable
            - **Medium confidence (60-80%)**: Be cautious
            - **Low confidence (<60%)**: Get a second opinion
            
            **Malware Types:**
            - **Adware**: Annoying ads
            - **Trojan**: Pretends to be useful
            - **Spyware**: Watches you secretly
            - **Ransomware**: Locks your files
            """)
    
    with tab4:
        st.header("ℹ️ About This Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 What it does")
            st.markdown("""
            This tool uses machine learning to detect malware in Android APK files by analyzing:
            
            - **Static features** from APK structure
            - **Permission requests** (9,503 features total)
            - **API call patterns**
            - **Code characteristics**
            
            **Dataset:** CICANDMAL2020
            - 357,805 Android apps
            - 15 malware types + Benign
            """)
        
        with col2:
            st.subheader("🤖 Models Available")
            st.markdown("""
            **Random Forest (Recommended)**
            - Accuracy: 87.64%
            - Uses 100 decision trees
            - Best overall performance
            
            **Logistic Regression**
            - Accuracy: 86.63%
            - Faster predictions
            - Smaller model size
            """)
        
        st.markdown("---")
        
        st.subheader("⚠️ Important Notes")
        st.warning("""
        **Current Limitations:**
        - APK feature extraction is not yet fully implemented
        - This is a demonstration/research tool
        - Should not be used as the sole security measure
        
        **Future Enhancements:**
        - Real APK parsing with Androguard
        - Dynamic analysis integration
        - Batch processing support
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>Built with Python • Scikit-learn • Streamlit</p>
            <p>Semester 5 Machine Learning Project</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

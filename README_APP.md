# 🛡️ APK Malware Detection Web Application

A machine learning-powered web application for detecting malware in Android APK files.

## 📋 Features

- 🤖 **Two ML Models**: Random Forest (87.64% accuracy) & Logistic Regression (86.63%)
- 📤 **APK Upload**: Upload APK files for analysis
- 📊 **Detailed Results**: Shows confidence scores and probability distributions
- 🎨 **Modern UI**: Clean, intuitive interface built with Streamlit
- 🧪 **Demo Mode**: Test the models without real APK files

## 🚀 Quick Start

### 1. Install Dependencies (if needed)

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 3. Use the Application

**Option A: Demo Mode (Recommended for testing)**
1. Click on the "🧪 Demo Mode" tab
2. Click "Generate Random Sample"
3. See instant predictions!

**Option B: Upload APK**
1. Click on the "📤 Upload APK" tab
2. Upload an APK file
3. Click "Analyze APK"
4. View results and risk assessment

## 📁 Project Structure

```
MyProject/
├── app.py                          # Main web application
├── feature_extractor.py            # APK feature extraction module
├── malware_detection.ipynb         # Training notebook
├── models/                         # Trained models directory
│   ├── random_forest_model.pkl     # RF model (87.64% accuracy)
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── class_labels.pkl
└── DataSet/
    └── cicandmal2020-static.parquet
```

## 🎯 How It Works

### 1. Feature Extraction
- APK files are analyzed for static features
- 9,503 features extracted including:
  - Permissions requested
  - API calls used
  - Code patterns
  - Structural characteristics

### 2. Prediction
- Features are fed to trained ML model
- Model outputs prediction + confidence score
- 15 malware types + Benign classification

### 3. Results Display
- **Prediction**: Main classification (e.g., "Trojan", "Benign")
- **Confidence**: How sure the model is (0-100%)
- **Risk Level**: High/Medium/Low based on confidence
- **Probability Chart**: Shows likelihood of each malware type

## 🔧 Advanced Usage

### Using with Real APK Files

**Current Status**: Feature extraction is a placeholder. 

**To enable real APK analysis:**

1. **Install Androguard** (comprehensive APK analysis):
```bash
pip install androguard
```

2. **Update feature_extractor.py**:
   - Implement proper feature extraction matching CICANDMAL2020 dataset
   - Map extracted features to the 9,503 feature format
   - See comments in `feature_extractor.py` for guidance

3. **Integrate with app.py**:
```python
from feature_extractor import APKFeatureExtractor

# In extract_features_from_apk function:
extractor = APKFeatureExtractor(feature_columns)
features = extractor.extract_with_androguard(apk_path)
```

### Model Selection

**Random Forest** (Default):
- ✅ Highest accuracy: 87.64%
- ✅ Best F1-Score: 86.21%
- ⚠️ Larger file size: 77 MB
- ⚠️ Slower predictions: ~0.3s

**Logistic Regression**:
- ✅ Fast predictions: ~0.1s
- ✅ Smaller size: 559 KB
- ⚠️ Slightly lower accuracy: 86.63%

Change in sidebar dropdown: Settings → Select Model

## 📊 Model Performance

### Random Forest
```
Accuracy:  87.64%
Precision: 87.86%
Recall:    87.64%
F1-Score:  86.21%
```

### Logistic Regression
```
Accuracy:  86.63%
Precision: 85.26%
Recall:    86.63%
F1-Score:  85.00%
```

### Malware Types Detected
1. Benign (Safe apps)
2. Adware
3. Backdoor
4. Banker
5. Dropper
6. FileInfector
7. NoCategory
8. PUA (Potentially Unwanted Application)
9. Ransomware
10. Riskware
11. SMS Malware
12. Scareware
13. Spyware
14. Trojan
15. Zero-day

## 🎨 Screenshots

### Main Interface
```
🛡️ APK Malware Detector
─────────────────────────
[ Upload APK | Demo Mode | About ]

📤 Choose an APK file...
[Browse...] [Analyze APK]
```

### Prediction Result - Benign
```
┌──────────────────────────┐
│     ✅ Safe              │
│       Benign             │
│  Confidence: 92.5% (High)│
└──────────────────────────┘
```

### Prediction Result - Malware
```
┌──────────────────────────┐
│  ⚠️ Threat Detected      │
│       Trojan             │
│  Confidence: 85.3% (High)│
└──────────────────────────┘
```

## 🛠️ Troubleshooting

### Models not found
```bash
# Check if models directory exists
ls -la models/

# Should contain:
# - random_forest_model.pkl
# - logistic_regression_model.pkl
# - scaler.pkl
# - feature_columns.pkl
# - class_labels.pkl
```

### Streamlit won't start
```bash
# Reinstall streamlit
pip install --upgrade streamlit

# Check Python version (needs 3.8+)
python --version
```

### Import errors
```bash
# Install all dependencies
pip install -r requirements.txt

# Or manually:
pip install streamlit pandas numpy scikit-learn joblib
```

## 📝 Future Enhancements

- [ ] Real APK feature extraction with Androguard
- [ ] Dynamic analysis integration
- [ ] Batch processing (analyze multiple APKs)
- [ ] API endpoint for programmatic access
- [ ] Model retraining interface
- [ ] Export analysis reports (PDF/JSON)
- [ ] VirusTotal integration
- [ ] Historical analysis tracking

## 🔒 Security Notes

⚠️ **Important Disclaimers:**

1. **Research Tool**: This is an academic/research project
2. **Not Production-Ready**: Feature extraction is incomplete
3. **False Positives**: ML models aren't perfect (87% accuracy)
4. **Complement Other Tools**: Use alongside antivirus software
5. **Handle APKs Safely**: Analyze in isolated environment

## 📚 Technical Details

### Dataset
- **Name**: CICANDMAL2020 Static Dataset
- **Samples**: 357,805 Android applications
- **Features**: 9,505 static analysis features
- **Classes**: 15 malware types + Benign

### Machine Learning
- **Algorithms**: Random Forest, Logistic Regression
- **Training Set**: 286,244 samples (80%)
- **Test Set**: 71,561 samples (20%)
- **Cross-Validation**: Skipped (large test set sufficient)

### Technologies
- **Python**: 3.8+
- **Streamlit**: Web framework
- **Scikit-learn**: ML models
- **Pandas/NumPy**: Data processing
- **Joblib**: Model serialization

## 👥 Credits

**Project**: Semester 5 Machine Learning Project
**Dataset**: CICANDMAL2020
**Models**: Random Forest, Logistic Regression
**Framework**: Streamlit

## 📧 Support

For issues or questions:
1. Check Troubleshooting section above
2. Review code comments in `app.py`
3. Consult `malware_detection.ipynb` for model details

---

**Built with ❤️ using Python, Scikit-learn, and Streamlit**








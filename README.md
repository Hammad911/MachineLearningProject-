# 🛡️ Android Malware Detection System

A machine learning-based web application for detecting malware in Android APK files using static analysis.

## 📋 Features

- **APK File Upload**: Easy drag-and-drop interface
- **15-Class Malware Detection**: Identifies various malware types including:
  - Benign, Riskware, Adware, Trojan, Zeroday, Ransomware, Spy, SMS, Dropper, and more
- **Real-time Analysis**: Fast feature extraction and prediction
- **Detailed Results**: Shows prediction, confidence score, risk level, and recommendations
- **User-Friendly Interface**: Clean Streamlit web app

## 🚀 Quick Start

### 1. Set Up Python Environment

**Option A: Automated Setup (Recommended)**

**macOS/Linux:**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
source venv/bin/activate
```

**Windows:**
```bash
setup_environment.bat
venv\Scripts\activate
```

**Option B: Conda Environment**
```bash
conda env create -f environment.yml
conda activate malware_detection
```

**Option C: Manual Setup**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate      # Windows

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

📖 **See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed instructions**

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Load the static dataset
- Train a Random Forest classifier
- Save the model as `malware_model.pkl`
- Save feature columns and class labels

**Note:** 
- Training uses the **complete dataset** (357,805 samples) for maximum accuracy
- Training may take **20-60 minutes** depending on your system
- Requires **8GB+ RAM** (16GB recommended)
- You can also use the Jupyter notebook: `jupyter notebook malware_detection.ipynb`

### 3. Run the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 4. Upload an APK

- Click "Choose an APK file"
- Select an Android APK file
- Wait for analysis (usually 10-30 seconds)
- View results!

## 📁 Project Structure

```
MyProject/
├── DataSet/
│   ├── cicandmal2020-static.parquet    # Training dataset (357K samples)
│   └── cicandmal2020-dynamic.parquet   # (Not used in this project)
├── malware_detection.ipynb             # Industry-level ML pipeline notebook
├── train_model.py                      # Model training script
├── apk_extractor.py                    # APK feature extraction
├── app.py                              # Streamlit web application
├── setup_environment.sh                # Environment setup (macOS/Linux)
├── setup_environment.bat               # Environment setup (Windows)
├── environment.yml                      # Conda environment file
├── requirements.txt                     # Python dependencies
├── ENVIRONMENT_SETUP.md                 # Detailed environment guide
└── README.md                           # This file
```

## 🔧 How It Works

1. **Feature Extraction**: When you upload an APK, the system extracts static features:
   - Permissions requested
   - API calls in code
   - Code structure metrics
   - Intent filters
   - And 9,500+ other features

2. **Model Prediction**: The extracted features are fed into a trained Random Forest model

3. **Results Display**: The model predicts:
   - Malware type (15 classes)
   - Confidence score
   - Risk level (LOW/MEDIUM/HIGH/CRITICAL)
   - Recommendations

## 📊 Model Performance

The model is trained on 357,805 samples with:
- **15 malware classes**
- **9,505 features**
- **Random Forest algorithm**
- **Stratified train/test split (80/20)**

## ⚠️ Important Notes

- **Educational Purpose**: This tool is for learning and demonstration
- **False Positives/Negatives**: ML models are not 100% accurate
- **Use Official Stores**: Always prefer official app stores when possible
- **Exercise Caution**: Be careful when installing apps from unknown sources

## 🛠️ Troubleshooting

### Model files not found
If you see "Model files not found", run:
```bash
python train_model.py
```

### APK extraction errors
Make sure `androguard` is installed:
```bash
pip install androguard
```

### Memory issues
If training fails due to memory:
- Reduce `n_estimators` in `train_model.py`
- Use a smaller sample of the dataset

## 📝 Future Improvements

- [ ] Improve feature extraction to match exact dataset format
- [ ] Add batch processing for multiple APKs
- [ ] Add model explanation (SHAP values)
- [ ] Support for APK file download/export
- [ ] Database integration for storing results

## 📚 References

- Dataset: CICANDMAL2020 from Canadian Institute for Cybersecurity
- Framework: Streamlit, Scikit-learn, Androguard

## 👤 Author

Machine Learning Project - Semester 5

## 📄 License

Educational use only

# MachineLearningProject-

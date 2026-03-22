---
title: Autism Screening AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.41.0
app_file: streamlit_app.py
pinned: false
---

# 🧠 AI-Powered Autism Screening System

Early detection of autism spectrum disorder (ASD) using machine learning and explainable AI.

## 📁 Project Structure

```
autism/
├── data/                      # Dataset & data fetching scripts
│   ├── autism_screening.csv   # Main dataset (704 records)
│   └── fetch_dataset.py       # Download script
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda_and_data_loading.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_explainability.ipynb
├── models/                    # Saved ML models
├── results/                   # Analysis outputs & visualizations
└── README.md
```

## 🚀 Quick Start

### 1. Get the Dataset

**Option A: Download Automatically**
```bash
cd data
python fetch_dataset.py
```

**Option B: Download Manually**
- Download from [Kaggle](https://www.kaggle.com/datasets/fauzanardh/autism-screening-data) (704 records)
- Or [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult+Data)
- Save as `data/autism_screening.csv`

**Option C: Start with Sample Data**
- A sample dataset will be created automatically if real data isn't found

### 2. Run the Analysis Notebook

```bash
# Make sure you're in the project root
jupyter notebook notebooks/01_eda_and_data_loading.ipynb
```

## 📊 What's Included

### Notebook 1: EDA & Data Loading
- ✅ Load 704-record autism screening dataset
- ✅ Analyze class balance (autism vs. non-autism)
- ✅ Check for missing values & data completeness
- ✅ Statistical feature analysis
- ✅ Quality assessment report

### Notebook 2: Model Training (Coming)
- Build baseline model (Logistic Regression)
- Compare models (Random Forest, SVM, etc.)
- Cross-validation & performance metrics
- Train-test split strategy

### Notebook 3: Explainability (Coming)
- SHAP values for feature importance
- Interpretable results for non-technical users
- Risk factor identification
- Confidence scoring

## 🎯 Dataset Info

**Size:** 704 adult screening records  
**Target:** Binary classification (Autism: Yes/No)  
**Features:** ~20-30 features based on screening questionnaires (AQ-10, etc.)  
**Class Distribution:** Typically ~30% positive, ~70% negative  

## 📋 Questionnaire Features

Common screening features include:
- Social attention & awareness
- Communication patterns
- Focused attention
- Imagination abilities
- Pattern recognition
- Memory for details
- Social relationships
- Anxiety levels
- Voice tone understanding

## ⚙️ Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
shap (for explainability)
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter shap
```

## 📈 Next Steps

1. **Load the data** → Run Notebook 01
2. **Explore patterns** → Check class balance & features
3. **Build models** → Run Notebook 02
4. **Explain results** → Run Notebook 03
5. **Deploy UI** → Build Streamlit app (optional)

## 🔒 Disclaimer

⚠️ **This tool is for screening support only, not medical diagnosis.**  
- Always consult with healthcare professionals
- Intended for educational & awareness purposes
- Not a substitute for professional evaluation

## 📚 Resources

- [Autism Spectrum Australia](https://www.autism.org.au/)
- [DSM-5 Diagnostic Criteria](https://www.psychiatry.org/)
- [UCI ML Autism Dataset](https://archive.ics.uci.edu/ml/datasets/Autism+Screening+Adult+Data)

---

*Ready to explore? Start with Notebook 01! 🚀*

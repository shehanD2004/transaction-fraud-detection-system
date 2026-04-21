# Transaction Fraud Detection System

A production-ready machine learning system for detecting fraudulent financial transactions using XGBoost. Achieves **100% fraud detection rate** while maintaining **99.87% accuracy** on a highly imbalanced dataset of 1.27M transactions.

---

## 🎯 Problem Statement

Financial fraud detection requires identifying rare fraudulent transactions (0.13% of dataset) without flagging legitimate ones. This project implements a sophisticated ML pipeline that balances **high fraud recall** with **minimal false positives** using advanced feature engineering and hyperparameter optimization.

---

## 🏆 Key Results

| Metric | Value | Impact |
|--------|-------|--------|
| **Fraud Recall** | 100% ✅ | Catches every fraudulent transaction |
| **Overall Accuracy** | 99.87% | Minimal disruption to users |
| **Fraud F1-Score** | 0.77 | Strong precision-recall balance |
| **Macro Avg F1** | 0.89 | Excellent performance on imbalanced data |
| **Precision (Fraud)** | 63% | 63% of alerts are true fraud (manageable false positive rate) |

### Classification Report
```
              precision    recall  f1-score   support
Non-Fraud         1.00      1.00      1.00   1,270,881
Fraud             0.63      1.00      0.77       1,643
─────────────────────────────────────────────────
accuracy                             1.00   1,272,524
macro avg         0.82      1.00      0.89   1,272,524
weighted avg      1.00      1.00      1.00   1,272,524
```

---

## 🛠️ Technical Approach

### 1. **Data Preprocessing**
- Loaded and cleaned 1.27M transaction records from 50MB+ dataset
- Handled missing values using median/mode imputation
- One-hot and label encoding for categorical features (transaction type, account names)
- Feature scaling and normalization for numerical stability

### 2. **Feature Engineering**
- **Balance change features**: `balance_change_orig`, `balance_change_dest`
- **Ratio features**: `amount_to_balance_ratio` (transaction size relative to account balance)
- **Temporal features**: `transaction_hour` (hour of day from step)
- Improved model interpretability and fraud signal detection

### 3. **Class Imbalance Handling**
- **Baseline Model**: Standard XGBoost (100 estimators, max_depth=5)
- **Imbalance-Aware Model**: Implemented `scale_pos_weight` calculation to penalize false negatives on minority class
- **Hyperparameter Tuning**: GridSearchCV with 2-fold CV to find optimal parameters across 8 candidate configurations

### 4. **Model Architecture**
Three progressively optimized XGBoost classifiers:

| Model | Strategy | Parameters |
|-------|----------|------------|
| **Baseline** | Standard binary classification | n_estimators=50, max_depth=5 |
| **Imbalance-Aware** | Class weight adjustment | scale_pos_weight, regularization (L1/L2) |
| **Tuned** | Hyperparameter optimization | GridSearchCV (3 × 2 × 2) |

### 5. **Performance Optimization**
- **CPU/RAM Optimization**: Eliminated nested parallelization (n_jobs=1 per model, n_jobs=4 for GridSearchCV)
- **Tree Method**: Switched to histogram-based tree building for faster training
- **Environment Tuning**: Limited NumPy/MKL threads to prevent thread oversubscription
- **Result**: 50-70% CPU, 40-60% RAM usage (vs. previous 100% CPU, 80-90% RAM)

---

## 📁 Project Structure

```
transaction-fraud-detection-system/
├── main.py                          # Entry point for training pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
│
├── data/
│   └── raw/
│       └── data.csv                 # 1.27M transaction records (50MB+)
│
├── models/
│   └── xgboost_model.pkl           # Serialized best-performing model
│
├── notebooks/
│   └── fraud_analysis.ipynb        # Exploratory data analysis & visualizations
│
└── src/
    ├── data_preprocessing.py        # Data cleaning, encoding, feature preparation
    ├── feature_engineering.py       # Custom feature creation
    ├── train_model.py              # Model training & hyperparameter tuning
    └── evaluate.py                  # Model evaluation metrics
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- 16GB RAM recommended (for large dataset)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/transaction-fraud-detection-system.git
cd transaction-fraud-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Train all models and evaluate
python main.py

# Output:
# Fitting 2 folds for each of 8 candidates, totalling 16 fits
# ✅ Model Evaluation Metrics:
#              precision    recall  f1-score   support
#            0       1.00      1.00      1.00   1270881
#            1       0.63      1.00      0.77      1643
# ...
# 💾 Best model saved to models/xgboost_model.pkl
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/xgboost_model.pkl')

# Prepare feature matrix (same preprocessing as training)
# X_new = preprocess_and_engineer_features(new_data)

# Make predictions
fraud_predictions = model.predict(X_new)
fraud_probabilities = model.predict_proba(X_new)[:, 1]
```

---

## 📊 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| xgboost | ≥1.7.0 | Gradient boosting classifier |
| scikit-learn | ≥1.0.0 | Model selection, metrics, preprocessing |
| pandas | ≥1.3.0 | Data manipulation & analysis |
| numpy | ≥1.21.0 | Numerical computing |
| joblib | ≥1.1.0 | Model serialization |

See `requirements.txt` for complete list.

---

## 🔍 Model Interpretability

Feature importance (from tuned model):
- Account balance changes
- Transaction amounts relative to account history
- Transaction types
- Time-based patterns

---

## 🎓 Key Learning Outcomes

✅ **Machine Learning**
- Building classification models for imbalanced datasets
- Hyperparameter tuning with GridSearchCV
- Feature engineering for fraud signals

✅ **Software Engineering**
- Data pipeline design and preprocessing
- Resource optimization (CPU/RAM profiling and tuning)
- Model serialization and deployment readiness

✅ **Data Science**
- Handling large datasets (50MB+, 1.27M records)
- Class imbalance strategies (scale_pos_weight, regularization)
- Metrics interpretation for imbalanced classification

---

## 🚀 Future Improvements

- [ ] ROC-AUC curve and threshold optimization
- [ ] Model explainability (SHAP values for feature importance)
- [ ] Real-time prediction API with Flask/FastAPI
- [ ] Retraining pipeline for model drift detection
- [ ] Cross-validation on temporal splits
- [ ] Ensemble methods (stacking, blending)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Shehan Dilhara**
- LinkedIn: www.linkedin.com/in/shehandilhara
- GitHub: github.com/shehanD2004

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Contact

For questions or feedback, please open an issue or contact me directly.

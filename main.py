import os

# Optimize CPU/RAM usage by limiting thread oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from src.data_preprocessing import preprocess_data
from src.feature_engineering import apply_feature_engineering
from src.train_model import train_all_models
from src.evaluate import evaluate_model
import joblib
import pandas as pd

def main():
    # 1️⃣ Load raw data and apply feature engineering, then preprocess
    from src.data_preprocessing import load_data, clean_data, encode_categorical, prepare_features
    
    df = load_data("data/raw/data.csv")
    df = clean_data(df)
    df = apply_feature_engineering(df)
    df, encoders = encode_categorical(df)
    X, y, feature_columns = prepare_features(df)
    
    # 2️⃣ Train all models
    results = train_all_models(X, y)
    
    # 4️⃣ Evaluate tuned model
    metrics = evaluate_model(results["tuned_model"], results["X_test"], results["y_test"])
    
    print("✅ Model Evaluation Metrics:")
    print(metrics["classification report"])
    
    # 5️⃣ Save best model
    joblib.dump(results["tuned_model"], "models/xgboost_model.pkl")
    print("💾 Best model saved to models/xgboost_model.pkl")

if __name__ == "__main__":
    main()
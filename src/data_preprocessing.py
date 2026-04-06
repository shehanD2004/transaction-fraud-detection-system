import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and clean dataset
    """
    # Identify target column
    if 'isFraud' in df.columns:
        target_column = 'isFraud'
    elif 'isFlaggedFraud' in df.columns:
        target_column = 'isFlaggedFraud'
    else:
        raise ValueError("Target column not found")

    # Remove rows where target is NaN
    df = df.dropna(subset=[target_column])

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill numerical missing values with median
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        #df[col].fillna('unknown', inplace=True)
        df[col] = df[col].fillna('unknown')

    return df


def encode_categorical(df: pd.DataFrame):
    """
    Encode categorical variables using Label Encoding
    """
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = pd.Series(le.fit_transform(df[col].astype(str)), index=df.index)
        label_encoders[col] = le

    return df, label_encoders


def prepare_features(df: pd.DataFrame):
    """
    Prepare feature matrix X and target vector y
    """
    # Identify target
    target_column = 'isFraud' if 'isFraud' in df.columns else 'isFlaggedFraud'

    # Exclude original categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    exclude_columns = [target_column] + categorical_columns

    feature_columns = [col for col in df.columns if col not in exclude_columns]

    X = df[feature_columns]
    y = df[target_column]

    # Ensure y is integer
    if y.dtype != 'int':
        y = y.astype(int)

    return X, y, feature_columns


def preprocess_data(file_path: str):
    """
    Full preprocessing pipeline
    """
    df = load_data(file_path)
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    X, y, feature_columns = prepare_features(df)

    return X, y, feature_columns, encoders
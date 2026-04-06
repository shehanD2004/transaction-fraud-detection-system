import pandas as pd


def create_balance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create balance-related features for fraud detection
    """
    # Balance change for origin account
    if all(col in df.columns for col in ['oldbalanceOrg', 'newbalanceOrig']):
        df['balance_change_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']

    # Amount to balance ratio
    if all(col in df.columns for col in ['amount', 'oldbalanceOrg']):
        df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)

    # Balance change for destination account
    if all(col in df.columns for col in ['oldbalanceDest', 'newbalanceDest']):
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features
    """
    if 'step' in df.columns:
        df['transaction_hour'] = df['step'] % 24

    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps
    """
    df = create_balance_features(df)
    df = create_time_features(df)

    return df
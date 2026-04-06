import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
    except Exception:
        # fallback if stratification fails
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    return X_train, X_test, y_train, y_test


# -------------------------------
# MODEL 1: BASELINE XGBOOST
# -------------------------------
def train_baseline_model(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=1
    )

    model.fit(X_train, y_train)
    return model


# -------------------------------
# MODEL 2: IMBALANCE HANDLING
# -------------------------------
def train_imbalanced_model(X_train, y_train):
    # Handle class imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=1
    )

    model.fit(X_train, y_train)
    return model, scale_pos_weight


# -------------------------------
# MODEL 3: HYPERPARAMETER TUNING
# -------------------------------
def train_tuned_model(X_train, y_train, scale_pos_weight):
    param_grid = {
        'max_depth': [3, 4],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            tree_method='hist',
            n_jobs=1
        ),
        param_grid=param_grid,
        cv=2,
        scoring=scorer,
        verbose=1,
        refit=True,
        n_jobs=4
    )

    grid_search.fit(X_train, y_train)

    return grid_search, grid_search.best_estimator_, grid_search.best_params_


# -------------------------------
# FULL TRAINING PIPELINE
# -------------------------------
def train_all_models(X, y):
    """
    Train all three model variations
    """
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    baseline_model = train_baseline_model(X_train, y_train)

    imbalanced_model, scale_pos_weight = train_imbalanced_model(X_train, y_train)

    grid_search, tuned_model, best_params = train_tuned_model(
        X_train, y_train, scale_pos_weight
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "baseline_model": baseline_model,
        "imbalanced_model": imbalanced_model,
        "tuned_model": tuned_model,
        "best_params": best_params,
        "grid_search": grid_search
    }
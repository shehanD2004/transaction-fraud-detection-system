
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    Returns a dictionary of metrics.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion matrix": confusion_matrix(y_test, y_pred),
        "classification report": classification_report(y_test, y_pred)
    }

    return metrics
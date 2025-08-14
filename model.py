from xgboost import XGBClassifier
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_xgboost(X, y, n_splits, num_rounds):
    """
    Train an XGBoost model with K-Fold cross-validation.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        n_splits (int): Number of folds for cross-validation.
        num_rounds (int): Number of boosting rounds.
        
    Returns:
        list: List of trained models for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    
    params = {
        'objective': 'multi:softmax',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_estimators': num_rounds,
    }
    fold_index = 0
    for train_index, val_index in kf.split(X):
        fold_index += 1
        print(f"Training fold {fold_index}...")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        models.append(model)
        print(f"Trained model for fold with {len(train_index)} training samples and {len(val_index)} validation samples.")
    print("Training complete.")
    return models

def predict_xgboost(models, X):
    """
    Make predictions using the trained XGBoost models.
    
    Args:
        models (list): List of trained XGBoost models.
        X (np.ndarray): Feature matrix for prediction.
        
    Returns:
        np.ndarray: Predicted labels.
    """
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0).astype(int)

def evaluate_xgboost(models, X, y):
    """
    Evaluate the XGBoost models on the provided dataset.
    
    Args:
        models (list): List of trained XGBoost models.
        X (np.ndarray): Feature matrix for evaluation.
        y (np.ndarray): True labels for evaluation.
        
    Returns:
        tuple: Accuracy and F1 score of the predictions.
    """
    y_pred = predict_xgboost(models, X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    return accuracy, f1

def train_and_evaluate_xgboost(X, y, n_splits, num_rounds):
    """
    Train and evaluate XGBoost model with K-Fold cross-validation.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        n_splits (int): Number of folds for cross-validation.
        num_rounds (int): Number of boosting rounds.
        num_gpus (int): Number of GPUs to use.
        
    Returns:
        tuple: List of trained models, accuracy, and F1 score.
    """
    models = train_xgboost(X, y, n_splits=n_splits, num_rounds=num_rounds)
    if not models:
        raise ValueError("No models were trained. Check your data and parameters.")
    accuracy, f1 = evaluate_xgboost(models, X, y)
    return models, accuracy, f1



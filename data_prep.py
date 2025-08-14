import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_train_data(file_path):
    """
    Load data from a CSV file and return features and labels.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        tuple: Features (X) and labels (y).
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=['label']).values
    y = data['label'].values
    return X, y

def load_test_data(file_path):
    """
    Load test data from a CSV file and return features.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        ndarray: Features (X).
    """
    data = pd.read_csv(file_path)
    X = data.values
    return X

def normalize(X):
    """
    Normalize the feature set.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

if __name__ == "__main__":
    file_path = 'data/mnist.csv'  # Update with your actual file path
    X, y = load_data(file_path)
    # X = normalize(X)
    print(f"Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
    # Example usage
    # X, y = load_data('path/to/your/data.csv')
    # print(X[:5], y[:5])
    # This will print the first 5 rows of features and labels
    # Ensure to handle the file path correctly based on your environment
    # You can also add error handling for file not found or empty data scenarios


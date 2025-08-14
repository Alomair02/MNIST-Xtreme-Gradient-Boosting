from data_prep import load_train_data, load_test_data
import pandas as pd
import numpy as np
from submission import create_submission
from model import train_and_evaluate_xgboost
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost model on MNIST dataset.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.")
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of boosting rounds.")
    parser.add_argument("--data_directory", type=str, default='data', help="Path to the training data.")
    args = parser.parse_args()

    X, y = load_train_data(args.data_directory + '/train.csv')
    X_test = load_test_data(args.data_directory + '/test.csv')
    
    models, accuracy, f1 = train_and_evaluate_xgboost(X, y, n_splits=args.n_splits, num_rounds=args.num_rounds)
    submission_file = 'submission.csv'
    create_submission(models, X_test, submission_file)
    print(f"Submission file created: {submission_file}")
    print(f"Training complete. Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    print(f"Number of models trained: {len(models)}")

if __name__ == "__main__":
    main()

    # Example usage
    # python main.py --n_splits 5 --num_rounds 1000 --data_directory data/
    # This will train the model with 5 folds and 1000 boosting rounds.

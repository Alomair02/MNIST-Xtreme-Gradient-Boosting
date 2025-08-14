# creating the submission file for the MNIST digit recognizer competition
import numpy as np
import pandas as pd
from model import evaluate_xgboost
import argparse

def create_submission(models, X_test, output_file):
    preds = [model.predict(X_test) for model in models]
    preds = np.mean(preds, axis=0)
    submission = pd.DataFrame({"ImageId": np.arange(1, len(preds) + 1), "Label": preds})
    submission.to_csv(output_file, index=False)
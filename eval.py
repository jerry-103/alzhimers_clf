###Functions that evaluate the performance of ML classifiers

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd

def evaluate(fit_model, x_test, y_test):
    """
    :param fit_model: fitted model to be evaluated
    :param x_test: X test set
    :param y_test: y test set
    :return: dict of evaluations metrics
    """
    #Getting y_predictions for test set
    y_preds = fit_model.predict(x_test)
    #Initializing empty dictionary for evaul metrics
    eval_metrics = dict()
    #Adding metrics to Dictionary
    eval_metrics['precision'] = precision_score(y_test, y_preds)
    eval_metrics['recall'] = recall_score(y_test, y_preds)
    eval_metrics['f1_score'] = f1_score(y_test, y_preds)

    return eval_metrics
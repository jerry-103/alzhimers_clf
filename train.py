### Functions that will train various classifiers

import pandas as pd
from pandas.core.interchange.dataframe_protocol import Column
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_pipe(x_train, y_train, num_feats, cat_feats, random_seed):
    """
    :param x_train: X_train dataframe
    :param y_train: y_train dataframe
    :param num_feats: list of numerical features
    :param cat_feats: list of categorical features
    :param random_seed: random seed
    :return: best performing NaiveBayes, LogisticRegression classifiers
    """
    #creating numeric transformer
    num_trans = Pipeline(steps= [("imputer", SimpleImputer(strategy= 'mean')),
                                 ("scalar", StandardScaler())])

    #creating categorical transformer
    cat_trans = Pipeline(steps= [("imputer", SimpleImputer(strategy= "constant", fill_value= "missing")),
                                 ("onehot", OneHotEncoder(handle_unknown= "ignore"))])

    #Applying both transformers to data w/ columns_transformer
    preprocessor = ColumnTransformer(
        transformers= [
            ("num", num_trans, num_feats),
            ("cat", cat_trans, cat_feats)
        ]
    )

    #Creating Naive Bayes Pipeline
    nb_pipe = Pipeline(steps = [
        ("preprocessor", preprocessor),
        ("clf", BernoulliNB())
    ])
    #Parameters for NB grid search
    nb_params = {
        "clf__alpha": [1, 1e-1, 1e-3]
    }

    #Creating Logistic Regression Pipeline & GS params
    lr_pipe = Pipeline(steps= [
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression())
    ])
    lr_params = {
        "clf__C" : [0.01, 0.1, 1],
        "clf__class_weight": ['balanced', None]
    }

    #Using Grid search to find best params for NB clf
    nb_grid_search = GridSearchCV(
        nb_pipe, nb_params, n_jobs= -1
    )
    nb_grid_search.fit(x_train, y_train)
    #getting best performing estimator
    nb_best = nb_grid_search.best_estimator_
    print("Best Naive Bayes parameters:")
    print(nb_grid_search.best_params_)

    #Gridsearch to find best params for LogReg clf
    lr_grid_search = GridSearchCV(
        lr_pipe, lr_params, n_jobs= -1
    )
    lr_grid_search.fit(x_train, y_train)
    #getting best performing LR estimator
    lr_best = lr_grid_search.best_estimator_
    print("Best Logistic Regression parameters")
    print(lr_grid_search.best_params_)

    return nb_best, lr_best


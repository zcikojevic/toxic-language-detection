from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


def _load_comments(comments_file):
    data = pd.read_csv(comments_file, sep=',')
    X, y = np.array(data['comment_text']), np.array(data[data.columns[2:]])

    return X[:100], y[:100, :]


def _kfold_cv(clf, param_grid, X, y, k, verbose=False):
    inner = KFold(n_splits=k)

    gs = GridSearchCV(clf, param_grid, cv=inner)
    gs.fit(X, y)

    if verbose:
        pprint(gs.cv_results_)

    return gs.best_estimator_, gs.best_params_


def run(param_grid, classifier, multilabel=False, k_folds=5, comments_file='../../data/train.csv'):
    comments_X, comments_y = _load_comments(comments_file)

    clf = OneVsRestClassifier(
            Pipeline([
                ('bag_of_words', TfidfVectorizer()),
                ('dim_reduct', TruncatedSVD()),
                ('normalizer', Normalizer()),
                ('classifier', classifier)
    ]))

    comments_X_train, comments_X_test, comments_y_train, comments_y_test = train_test_split(comments_X, comments_y, train_size=0.7, random_state=1)

    best_estimator, best_params = _kfold_cv(clf, param_grid, comments_X_train, comments_y_train, k_folds)

    print('=================  Classification report  =================')
    print(classification_report(comments_y_test, best_estimator.predict(comments_X_test)))

    print('=================     Best parameters     =================')
    print(best_params)

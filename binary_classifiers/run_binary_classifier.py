from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

def _load_comments(comments_file):
    data = pd.read_csv(comments_file, sep=',')
    X, y = np.array(data['comment_text']), np.array(data['is_toxic'])

    """=======  <DISBALANCE MANAGMENT> ======="""
    #   --there are approx. 140 000 non toxic comments, while 16 000 are toxic
    #   --one way to deal with this is to remove (plenty of) non toxic comments
    #     to make the count roughly equal

    num_toxic_comments = np.sum(y)
    num_non_toxic_comments = y.shape[0] - num_toxic_comments

    non_toxic_indices = np.argwhere(y == 0).flatten()

    comments_to_be_removed = np.random.choice(
                             non_toxic_indices,
                             size=num_non_toxic_comments-num_toxic_comments+1,
                             replace=False)

    X = np.delete(X, comments_to_be_removed)
    y = np.delete(y, comments_to_be_removed)
    """=======  </DISBALANCE MANAGMENT> ======="""

    return X, y


def _kfold_cv(clf, param_grid, X, y, k, verbose=0):
    inner = KFold(n_splits=k)

    gs = GridSearchCV(clf, param_grid, cv=inner, verbose=verbose)
    gs.fit(X, y)

    return gs.best_estimator_, gs.best_params_


def run(param_grid, classifier, k_folds=5, comments_file='../../data/train_binary_labels.csv'):
    comments_X, comments_y = _load_comments(comments_file)


    clf = Pipeline([
                ('bag_of_words', TfidfVectorizer()),
                ('dim_reduct', TruncatedSVD()),
                ('normalizer', Normalizer()),
                ('classifier', classifier)
    ])

    comments_X_train, comments_X_test, comments_y_train, comments_y_test = train_test_split(comments_X, comments_y, train_size=0.7, random_state=1)

    best_estimator, best_params = _kfold_cv(clf, param_grid, comments_X_train, comments_y_train, k_folds, verbose=1)

    print('=================  Classification report  =================')
    print(classification_report(comments_y_test, best_estimator.predict(comments_X_test)))

    pprint('=================     Best parameters     =================')
    print(best_params)

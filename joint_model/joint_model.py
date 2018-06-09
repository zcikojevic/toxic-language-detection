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
from comments_preprocess import CommentsPreprocesser

class JointModel:
    def __init__(self, binary_clf, multilabel_clf, binary_param_grid, multilabel_param_grid):
        self.binary_clf = binary_clf
        self.multilabel_clf = multilabel_clf
        self.binary_param_grid = binary_param_grid
        self.multilabel_param_grid = multilabel_param_grid


    def fit(self, X, y, store_model=None, k_folds=5, verbose=1):
        cp = CommentsPreprocesser()
        X, y = cp.remove_multilabel_imabalances(X, y)
        X, y = cp.remove_binary_imbalances(X, y)

        X_binary, y_binary = cp.prepare_binary_comments(X, y)
        X_multi, y_multi = X, y

        # ============== #
        # Fit binary classifier
        # ============== #
        binary_best_estimator, binary_best_params = self._kfold_cv(
                                            self.binary_clf,
                                            self.binary_param_grid,
                                            X_binary,
                                            y_binary,
                                            k_folds,
                                            verbose=verbose)
        # ============== #
        # Fit binary classifier
        # ============== #
        mutlilabel_best_estimator, multilabel_best_params = self._kfold_cv(
                                                                self.multilabel_clf,
                                                                self.multilabel_param_grid,
                                                                X_multi,
                                                                y_multi,
                                                                k_folds,
                                                                verbose=verbose)

        self.binary_clf = binary_best_estimator
        self.multilabel_clf = mutlilabel_best_estimator

        return binary_best_estimator, binary_best_params, mutlilabel_best_estimator, multilabel_best_params


    def predict(self, X):
        binary_predict = self.binary_clf.predict(X)
        mulit_predict = self.multilabel_clf.predict(X)

        final_predictions = np.full_like(mulit_predict, -1)

        non_toxic_indices = np.argwhere(binary_predict == 0).flatten()
        toxic_indices = np.argwhere(binary_predict == 1).flatten()

        # place binary classifier's prediction of clean comments
        final_predictions[non_toxic_indices] = np.array([0, 0, 0, 0, 0, 0])

        final_predictions[toxic_indices] = mulit_predict[toxic_indices]

        return final_predictions


    def _kfold_cv(self, clf, param_grid, X, y, k, verbose=0):
        inner = KFold(n_splits=k)

        gs = GridSearchCV(clf, param_grid, cv=inner, verbose=verbose)
        gs.fit(X, y)

        return gs.best_estimator_, gs.best_params_
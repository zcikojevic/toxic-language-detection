import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from joint_model import JointModel

train_set_path = os.path.join('../', 'data/train.csv')
test_set_path = os.path.join('../', 'data/test_clean.csv')

binary_param_grid = {
            'bag_of_words__stop_words': ['english'],
            'bag_of_words__ngram_range': [(1, 2)],
            'bag_of_words__max_features': [500],
            'dim_reduct__n_components': [300],
            'normalizer__norm': ['l2'],
            'classifier__C': [5., 10.]
}

multilabel_param_grid = multilabel_param_grid  = [{
            'estimator__bag_of_words__stop_words': ['english'],
            'estimator__bag_of_words__ngram_range': [(1, 2)],
            'estimator__bag_of_words__max_features': [500],
            'estimator__dim_reduct__n_components': [300],
            'estimator__normalizer__norm': ['l2'],
            'estimator__classifier__C': [5., 10.]
}]

binary_clf = LogisticRegression()
multilabel_clf = LogisticRegression()

clf = JointModel(binary_clf, multilabel_clf, binary_param_grid, multilabel_param_grid)

train_comments = pd.read_csv(train_set_path)
test_comments = pd.read_csv(test_set_path)

X_train, y_train = np.array(train_comments['comment_text']), np.array(train_comments[train_comments.columns[2:]])
X_test, y_test = np.array(test_comments['comment_text']), np.array(test_comments[test_comments.columns[2:]])

clf.fit(X_train, y_train)
predict = clf.predict(X_test)

print('=================  Classification report  =================')
print(classification_report(y_test, predict))
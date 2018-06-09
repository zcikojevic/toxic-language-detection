import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import run_binary_classifier
import run_multilabel_classifier

train_binary = os.path.join('../../../', 'data/train_binary.csv')
test_binary = os.path.join('../../../', 'data/test_clean_binary.csv')

train_multilabel = os.path.join('../../../', 'data/train.csv')
test_multilabel = os.path.join('../../../', 'data/test_clean.csv')

binary_param_grid = {
        'bag_of_words__stop_words': ['english'],
        'bag_of_words__ngram_range': [(1, 2)],
        'bag_of_words__max_features': [500],
        'dim_reduct__n_components': [300],
        'normalizer__norm': ['l2'],
        'classifier__C': [5., 10.]
}


multilabel_param_grid  = [{
        'estimator__bag_of_words__stop_words': ['english'],
        'estimator__bag_of_words__ngram_range': [(1, 2)],
        'estimator__bag_of_words__max_features': [500],
        'estimator__dim_reduct__n_components': [300],
        'estimator__normalizer__norm': ['l2'],
        'estimator__classifier__C': [5., 10.]
}]


# =========================== #
# TRAIN
# BINARY CLASSIFIER
# =========================== #
binary_clf = run_binary_classifier.run(binary_param_grid, LogisticRegression(), comments_file=train_binary)

with open('./saved_models/log_reg_joint_binary.pkl', 'wb') as saved_model:
	pickle.dump(binary_clf, file=saved_model)

# =========================== #
# TRAIN
# MULTILABEL CLASSIFIER
# =========================== #
multilabel_clf = run_multilabel_classifier.run(multilabel_param_grid, LogisticRegression(), comments_file=train_multilabel)
with open('./saved_models/log_reg_joint_multilabel.pkl', 'wb') as saved_model:
	pickle.dump(binary_clf, file=saved_model)


# =========================== #
# PREDICT
# BINARY CLASSIFIER
# =========================== #
print('Binary prediction')

X_binary_test, y_binary_test = run_binary_classifier.load_comments(test_binary)
y_binary_test_predict = binary_clf.predict(X_binary_test)

print(classification_report(y_binary_test, y_binary_test_predict))


# =========================== #
# PREDICT
# MULTILABEL CLASSIFIER
# =========================== #
print('Multilabel predict')

X_multilabel_test, y_multilabel_test = run_multilabel_classifier.load_comments(test_multilabel)
y_multilabel_test_predict = multilabel_clf.predict(X_multilabel_test)

print(classification_report(y_multilabel_test, y_multilabel_test_predict))


# =========================== #
# FINAL JOINT PREDICTION
# =========================== #
final_predictions = np.full_like(y_multilabel_test_predict, -1)


non_toxic_indices = np.argwhere(y_binary_test_predict == 0).flatten()
toxic_indices = np.argwhere(y_binary_test_predict == 1).flatten()

# place binary classifier's prediction of clean comments
final_predictions[non_toxic_indices] = np.array([0, 0, 0, 0, 0, 0])

multilabel_toxic_predictions = y_multilabel_test_predict[toxic_indices]
final_predictions[toxic_indices] = multilabel_toxic_predictions

print(classification_report(y_multilabel_test, final_predictions))
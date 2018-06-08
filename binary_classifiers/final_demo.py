import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from run_binary_classifier import _load_comments, run

train_comments_path = os.path.join('../../../', 'data/train_binary.csv')
test_comments_path = os.path.join('../../../', 'data/test_clean_binary.csv')

param_grid = {
        'bag_of_words__stop_words': ['english'],
        'bag_of_words__ngram_range': [(1, 2)],
        'bag_of_words__max_features': [500],
        'dim_reduct__n_components': [300],
        'normalizer__norm': ['l2'],
        'classifier__C': [5., 10.]
}


clf = LogisticRegression()

trained_clf = run(param_grid, clf, comments_file=train_comments_path)

with open('./saved_models/log_reg_trained_binary.pkl', 'wb') as saved_model:
	pickle.dump(trained_clf, file=saved_model)

with open('./saved_models/log_reg_trained_binary.pkl', 'rb') as saved_model:
	loaded_clf = pickle.load(saved_model)

	X_test, y_test = _load_comments(test_comments_path)
	y_test_predict = loaded_clf.predict(X_test)

	print(classification_report(y_test, y_test_predict))
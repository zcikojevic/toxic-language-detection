from sklearn.linear_model import LogisticRegression
from run_multilabel_classifier import run

param_grid = [{
        'bag_of_words__stop_words': ['english'],
        'bag_of_words__ngram_range': [(1, 2)],
        'bag_of_words__max_features': [500],
        'dim_reduct__n_components': [300],
        'normalizer__norm': ['l2'],
        'classifier__C': [1., 0.1, 0.001]
}]


clf = LogisticRegression()

run(param_grid, clf)
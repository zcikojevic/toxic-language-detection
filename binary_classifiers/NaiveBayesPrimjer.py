from sklearn.naive_bayes import BernoulliNB
from run_binary_classifier import run

param_grid = {
        'bag_of_words__stop_words': ['english'],
        'bag_of_words__ngram_range': [(1, 2)],
        'bag_of_words__max_features': [500],
        'dim_reduct__n_components': [300],
        'normalizer__norm': ['l2'],
        'classifier__alpha': [1.0],
        'classifier__binarize': [0.0]
}

clf = BernoulliNB()

run(param_grid, clf)

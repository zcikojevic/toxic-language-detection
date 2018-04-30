from sklearn.svm import SVC
from run_binary_classifier import run

param_grid = {
        'bag_of_words__stop_words': ['english'],
        'bag_of_words__ngram_range': [(1, 2)],
        'bag_of_words__max_features': [500],
        'bag_of_words__lowercase': [True, False],
        'dim_reduct__n_components': [100],
        'normalizer__norm': ['l2']
}

clf = SVC()

run(param_grid, clf)

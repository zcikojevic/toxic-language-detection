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


def remove_toxic_imbalances(X, y):
    num_classes = y.shape[1]

    toxic_indices = {i:[] for i in range(num_classes)}
    for example_count, label in enumerate(y):
        for class_index, label in enumerate(label):
            if label == 1:
                toxic_indices[class_index].append(example_count)

    for key, value in toxic_indices.items():
        print(key, len(value))

    classes_counts = np.sum(y, axis=0)
    most_common_class, avg_class_count = np.argmax(classes_counts), int(np.mean(classes_counts))
    stdev = np.std(classes_counts)
    print(most_common_class, avg_class_count)
    print(classes_counts)

    comments_to_resample = {}
    for class_index, all_class_indices in toxic_indices.items():
        # ako je broj instanci tog razreda pola standardne devijacije iznad srednjeg broja razreda
        if abs(len(all_class_indices) - avg_class_count) > 0.5 * stdev:
            comments_to_resample[class_index] = np.random.choice(all_class_indices,
                                                size=abs(avg_class_count - classes_counts[class_index]),
                                                replace=True)
        else:
            comments_to_resample[class_index] = all_class_indices

    print('===================================')
    for class_index, indices_to_resample in comments_to_resample.items():
        print(class_index, len(indices_to_resample))
    print('===================================')

    resampled_X = np.array([])
    resampled_y = np.array([])
    for class_index, indices_to_resample in comments_to_resample.items():
        resampled_X = np.append(resampled_X, X[indices_to_resample])
        resampled_y = np.append(resampled_y, y[indices_to_resample])

    binary_labels = np.max(y, axis=1)
    non_toxic_indices = np.argwhere(binary_labels == 0)
    resampled_X = np.append(resampled_X, X[non_toxic_indices])
    resampled_y = np.append(resampled_y, y[non_toxic_indices])
    return resampled_X, resampled_y.reshape((-1, num_classes)).astype(np.int32)


def remove_toxic_nontoxic_imbalances(X, y):

    binary_labels = np.max(y, axis=1)
    num_toxic_comments = binary_labels.sum()
    num_non_toxic_comments = y.shape[0] - num_toxic_comments

    non_toxic_indices = np.argwhere(binary_labels == 0).flatten()

    print(num_non_toxic_comments)
    print(num_toxic_comments)

    comments_to_be_removed = np.random.choice(
                             non_toxic_indices,
                             size=num_non_toxic_comments-num_toxic_comments+1,
                             replace=False)

    X = np.delete(X, comments_to_be_removed)
    y = np.delete(y, comments_to_be_removed, axis=0)

    binary_labels = np.max(y, axis=1)
    num_toxic_comments = binary_labels.sum()
    num_non_toxic_comments = y.shape[0] - num_toxic_comments

    return X, y


def load_comments(comments_file):
    data = pd.read_csv(comments_file, sep=',')

    X, y = np.array(data['comment_text']), np.array(data[data.columns[2:]])

    X, y = remove_toxic_imbalances(X, y)
    X, y = remove_toxic_nontoxic_imbalances(X, y)

    return X, y


def _kfold_cv(clf, param_grid, X, y, k, verbose=0):
    inner = KFold(n_splits=k)

    gs = GridSearchCV(clf, param_grid, cv=inner, verbose=verbose)
    gs.fit(X, y)

    return gs.best_estimator_, gs.best_params_


def run(param_grid, classifier, multilabel=False, k_folds=5, comments_file='../../../data/train.csv'):
    comments_X, comments_y = load_comments(comments_file)

    # tu se moraju maknut klin komentari
    toxic_indices = np.argwhere(comments_y == 1).flatten()
    comments_X = comments_X[toxic_indices]
    comments_y = comments_y[toxic_indices]

    clf = OneVsRestClassifier(
            Pipeline([
                ('bag_of_words', TfidfVectorizer()),
                ('dim_reduct', TruncatedSVD()),
                ('normalizer', Normalizer()),
                ('classifier', classifier)
    ]))

    comments_X_train, comments_X_test, comments_y_train, comments_y_test = train_test_split(comments_X, comments_y, train_size=0.7, random_state=1)

    best_estimator, best_params = _kfold_cv(clf, param_grid, comments_X_train, comments_y_train, k_folds, verbose=1)

    print('=================  Classification report  =================')
    print(classification_report(comments_y_test, best_estimator.predict(comments_X_test)))

    print('=================     Best parameters     =================')
    pprint(best_params)
    return best_estimator
from run_multilabel_classifier import _load_comments
import os
import numpy as np
from pprint import pprint
import pandas as pd

data_path = os.path.join('../../..', 'data/train.csv')
data = pd.read_csv(data_path, sep=',')

X, y = np.array(data['comment_text']), np.array(data[data.columns[2:]])

labels = np.array(list(data.columns.values[2:]))

subsets = np.unique(y, axis=0, return_counts=True)
subsets, subsets_counts = subsets[0], subsets[1]

print(f"['clean'] {subsets_counts[0]}")
for subset, subset_count in zip(subsets[1:], subsets_counts[1:]):
    int_to_str_labels = np.argwhere(subset == 1).reshape((-1, ))
    print(labels[int_to_str_labels], subset_count)
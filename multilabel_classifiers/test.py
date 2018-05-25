import os
from run_multilabel_classifier import _load_comments
import numpy as np
from pprint import pprint
from collections import Counter

data_path = os.path.join('../../..', 'data/train.csv')
X, y = _load_comments(data_path)
comments_to_be_resampled, classes = _load_comments(data_path)
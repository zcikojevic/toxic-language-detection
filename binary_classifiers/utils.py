import argparse
import os
from pprint import pprint

import numpy as np
import pandas as pd


def multilabel_to_binary(df_input_file, df_output_file):
    multilabel_df = pd.read_csv(df_input_file, sep=',')

    #labels start at 3rd column
    labels = multilabel_df[multilabel_df.columns[2:]]
    labels = np.array(labels)

    #if there is atleast one label (of 6) that is 1 - the whole comment
    #is considered toxic
    #if however all labels are 0, np.max will return 0 - non toxic comment
    binary_labels = np.max(labels, axis=1)

    #merge comments with their binary labels: toxic vs non toxic
    outpout_df_data = np.stack((multilabel_df['comment_text'], binary_labels), axis=1)

    #the final result - all the comments are mapped to 2 classes
    header=['comment_text', 'is_toxic']
    df_binary_labels = pd.DataFrame(data=outpout_df_data, columns=header)

    df_binary_labels.to_csv(df_output_file)


def construct_test_set(test_set_file, test_set_labels, output_file):
    #columns: id, comment_text
    test_set_df = pd.read_csv(test_set_file, sep=',')

    #columns: id, toxic, severe_toxic...
    test_set_labels_df = pd.read_csv(test_set_labels, sep=',')

    #columns: id, comment_text, toxic, severe_toxic...
    #contains unlabeled comments, those labels are = -1
    joint = test_set_df.merge(test_set_labels_df)

    labels = joint.values[:, 2:]
    #retain all the labels that do not contain -1, i.e. zeros or ones
    labeled_comments_indices = np.argwhere(np.min(labels, axis=1) != -1).flatten()
    properly_labeled_comments = joint.iloc[labeled_comments_indices]
    properly_labeled_comments.to_csv(output_file, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_set',
                        type=str,
                        default=os.path.join('..', '..', '..', 'data', 'test.csv'))

    parser.add_argument('--test_set_labels',
                        type=str,
                        default=os.path.join('..', '..', '..', 'data', 'test_labels.csv'))

    parser.add_argument('--clean_multilabel_test_set',
                        type=str,
                        default=os.path.join('..', '..', '..', 'data', 'test_clean.csv'))

    parser.add_argument('--clean_binary_test_set',
                        type=str,
                        default=os.path.join('..', '..', '..', 'data', 'test_clean_binary.csv'))

    args = parser.parse_args()

    return args.test_set, args.test_set_labels, args.clean_multilabel_test_set, args.clean_binary_test_set


if __name__ == '__main__':
    test_set, test_set_labels, clean_multilabel_test_set, clean_binary_test_set = parse_arguments()

    construct_test_set(test_set, test_set_labels, clean_multilabel_test_set)
    multilabel_to_binary(clean_multilabel_test_set, clean_binary_test_set)
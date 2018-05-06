import pandas as pd
import numpy as np
from pprint import pprint
from sys import argv

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

if __name__ == '__main__':
    multilabel_to_binary(argv[1], argv[2])

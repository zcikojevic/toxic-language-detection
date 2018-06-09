import numpy as np


class CommentsPreprocesser:
    def __init__(self):
        pass


    def prepare_binary_comments(self, X, y):
        y_binary = np.max(y, axis=1)
        return X, y_binary


    def prepare_multilabel_toxic_comments(self, file_path):
        pass


    def remove_binary_imbalances(self, X, y):
        binary_labels = np.max(y, axis=1)
        num_toxic_comments = binary_labels.sum()
        num_non_toxic_comments = y.shape[0] - num_toxic_comments

        non_toxic_indices = np.argwhere(binary_labels == 0).flatten()

        comments_to_be_removed = np.random.choice(
                                 non_toxic_indices,
                                 size=num_non_toxic_comments-num_toxic_comments+1,
                                 replace=False)

        X = np.delete(X, comments_to_be_removed)
        y = np.delete(y, comments_to_be_removed, axis=0)

        binary_labels = np.max(y, axis=1)
        num_toxic_comments = binary_labels.sum()
        num_non_toxic_comments = y.shape[0] - num_toxic_comments

        #retained_comments_indices = set(range(num_toxic_comments + num_non_toxic_comments)) - set(comments_to_be_removed)

        return X, y#, retained_comments_indices


    def remove_multilabel_imabalances(self, X, y):
        num_classes = y.shape[1]

        toxic_indices = {i:[] for i in range(num_classes)}
        for example_count, label in enumerate(y):
            for class_index, label in enumerate(label):
                if label == 1:
                    toxic_indices[class_index].append(example_count)

        # for key, value in toxic_indices.items():
        #     print(key, len(value))

        classes_counts = np.sum(y, axis=0)
        most_common_class, avg_class_count = np.argmax(classes_counts), int(np.mean(classes_counts))
        stdev = np.std(classes_counts)
        # print(most_common_class, avg_class_count)
        # print(classes_counts)

        comments_to_resample = {}
        for class_index, all_class_indices in toxic_indices.items():
            # ako je broj instanci tog razreda pola standardne devijacije iznad srednjeg broja razreda
            if abs(len(all_class_indices) - avg_class_count) > 0.5 * stdev:
                comments_to_resample[class_index] = np.random.choice(all_class_indices,
                                                    size=abs(avg_class_count - classes_counts[class_index]),
                                                    replace=True)
            else:
                comments_to_resample[class_index] = all_class_indices

        # print('===================================')
        # for class_index, indices_to_resample in comments_to_resample.items():
        #     print(class_index, len(indices_to_resample))
        # print('===================================')

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


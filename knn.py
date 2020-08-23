import argparse
import numpy as np


class KNN:

    def __init__(self):
        self.__k = None
        self.__X = []
        self.__Y = []

    def fit(self, X, Y, k=5):
        self.__X = X
        self.__Y = Y
        self.__k = k

    def predict(self, X):

        Y = []
        for i in X:

            dist = []
            for idx, j in enumerate(self.__X):
                dist.append((np.sqrt(np.sum(np.square(np.subtract(i, j)))), self.__Y[idx]))
            k_neighbours = sorted(dist, key=lambda x: x[0])[:self.__k]

            count_1 = 0
            for k in k_neighbours:
                count_1 += k[1]

            Y.append(1) if count_1 > len(k_neighbours)//2 else Y.append(0)

        return Y


def _parse_data(fil_path): # same as assignment 1
    """
    read and parse dataset

    :param fil_path: path to dataset
    :return: the parsed dataset
    """
    with open(fil_path, 'r') as fil:
        data = fil.read()
    dataset = []
    for line in data.split():
        dataset.append(line.split(','))
    return dataset


def shuffle(X, Y):  # same as assignment 1
    combined = list(zip(X, Y))
    np.random.shuffle(combined)
    return zip(*combined)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--k', required=False, help='Method to reduce loss', default=5, type=int)
    parser.add_argument('--train_size', required=False, help='Fraction of total training set. Is between 0 and 1 and '
                                                             'is usually greater than 0.6', default=0.8)
    parser.add_argument('--shuffle', required=False, help='Randomly shuffle the dataset', nargs='?', const=True,
                        default=False)

    options, _ = parser.parse_known_args()
    dataset = _parse_data(options.dataset)

    X = []
    Y = []
    for data in dataset[1:]:
        x, y = data[:-1], data[-1]
        X.append(list(map(float, x))), Y.append(float(y))

    train_size = int(float(options.train_size) * len(X))

    if bool(options.shuffle):
        X, Y = shuffle(X, Y)

    print("Training data: ", train_size, "Total data points: ", len(X))

    knn = KNN()
    knn.fit(X[:train_size], Y[:train_size], options.k)
    err = np.sum(np.where(knn.predict(X[train_size:]) != np.array(Y[train_size:]), 1, 0))
    print("k: ", options.k, "ERR: ", err)

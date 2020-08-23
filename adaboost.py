import argparse
import numpy as np
from sys import exit
import matplotlib.pyplot as plt


class AdaBoost:

    def __init__(self):
        self.__distributions = []
        self.__epsilon = []
        self.__decision_stumps = []
        self.__weights = []
        self.__boosters = 0

    @property
    def distributions(self):
        return self.__distributions

    @property
    def weights(self):
        return self.__weights

    @property
    def stumps(self):
        return self.__decision_stumps

    @property
    def epsilon(self):
        return self.__epsilon

    def __calculate_stumps(self, X, y):
        """
        calculate best decision stump
        :param X: training data
        :param y: testing data
        :return: best feature 'd' for current distribution, stump value for feature 'd'
        """
        _f = float("inf")
        _theta = 0
        _j = 0

        for j in range(len(X[0])):

            st = list(zip(X[:, j], y, self.__distributions))
            st = sorted(st, key=lambda z: z[0])

            f = sum(map(lambda w: w[2] if w[1] == 1 else 0, st))

            if f < _f:
                _f = f
                _theta = st[0][0] - 1
                _j = j

            for idx, row in enumerate(st[:-1]):
                f -= row[1] * row[2]
                if f < _f and row[0] != st[idx + 1][0]:
                    _f = f
                    _theta = 0.5 * (row[0] + st[idx + 1][0])
                    _j = j

        return _j, _theta

    def fit(self, X, y, boosters=15):
        """

        :param X:
        :param y:
        :param boosters:
        :return:
        """
        _l = len(X)
        self.__weights = np.zeros(boosters)
        self.__epsilon = np.ones(boosters)
        self.__distributions = [1 / _l for _ in range(_l)]
        self.__decision_stumps = []
        self.__boosters = boosters

        for t in range(self.__boosters):
            j, theta = self.__calculate_stumps(X, y)
            self.__decision_stumps.append((j, theta))

            predictions = self.__predict(X[:, j], t)
            identity = list(map(lambda w, z: 0 if w == z else 1, y, predictions))
            self.__epsilon[t] = np.dot(self.__distributions, identity)

            self.__weights[t] = 0.5 * np.log(
                (1 - self.__epsilon[t]) / self.__epsilon[t])

            exp = np.exp(-1 * self.__weights[t] * y * predictions)
            denominator = np.dot(self.__distributions, exp)
            self.__distributions = np.multiply(self.__distributions, exp) / denominator

    def __predict(self, X, iteration):
        """

        :param X:
        :return:
        """
        return np.where(X > self.__decision_stumps[iteration][1], 1, -1)

    def predict(self, X):
        mapper = map(lambda x, y: np.multiply(x, y), self.__weights,
                     [self.__predict(X[:, self.__decision_stumps[iteration][0]], iteration) for iteration in range(self.__boosters)])

        return np.where(sum(mapper) > 0, 1, -1)


def shuffle(X, Y):  # referred to https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
    combined = list(zip(X, Y))
    np.random.shuffle(combined)
    return zip(*combined)


def calculate_erm(predict, X, Y, train_size=None):
    """
    calculate the ERM of the hypothesis class

    :param predict: method to predict the values
    :param X: features on which the model should predict the output
    :param Y: actual values to compare after prediction
    :param train_size: factor to multiply with dataset to obtain sample
    :return: total error
    """
    err = 0
    # print(predict(X[train_size:]))
    if train_size == len(X):
        train_size = 0
    for idx, i in enumerate(predict(X[train_size:])):
        if i != Y[idx]:
            err += 1
    return err


def train_validate(model, X, Y, train_size, n_iterations):
    """
    train the model and validate

    :param model: model to train the dataset
    :param X: features of the dataset
    :param Y: values to be predicted
    :param train_size: factor to multiply with dataset to obtain sample
    :param n_iterations: maximum number of iterations to run the model
    :return: total error
    """
    model.fit(np.array(X[:train_size]), np.array(Y[:train_size]), n_iterations)
    err = calculate_erm(model.predict, np.array(X), np.array(Y), train_size)

    return err


def k_fold(model, X, Y, n_folds, n_iterations):
    """
    perform k-fold validation on dataset

    :param model: model to train the dataset
    :param X: features of the dataset
    :param Y: values to be predicted
    :param n_folds: number of folds
    :param n_iterations: maximum number of iterations to run the model
    :return: ERMs and Weights over all folds
    """
    X = np.array_split(X, n_folds)
    Y = np.array_split(Y, n_folds)

    train_erm_err = []
    validation_erm_err = []
    weights = []
    stumps = []

    for i in range(n_folds):
        validation_set = Y[i]
        x = np.concatenate(X[:i] + X[i + 1:])
        y = np.concatenate(Y[:i] + Y[i + 1:])
        model.fit(x, y, n_iterations)
        err = calculate_erm(model.predict, X[i], validation_set)
        train_err = calculate_erm(model.predict, x, y)
        train_erm_err.append(train_err/len(x))
        validation_erm_err.append(err / len(X[i]))
        weights.append(list(model.weights))
        stumps.append(list(model.stumps))
        # print("Fold-{0} iterations run: {1}".format(i+1, model.total_iterations))
    return train_erm_err, validation_erm_err, weights, stumps


def _parse_data(fil_path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--mode', required=False, help='Method to reduce loss', default='erm')
    parser.add_argument('--boosters', required=False, help='Total number of boosters in the algorithm',
                        default=15)
    parser.add_argument('--train_size', required=False, help='Fraction of total training set. Is between 0 and 1 and '
                                                             'is usually greater than 0.6', default=1)
    parser.add_argument('--folds', required=False, help='Total folds while performing cross-validation', default=10)
    parser.add_argument('--shuffle', required=False, help='Randomly shuffle the dataset', nargs='?', const=True, default=False)
    parser.add_argument('--plot', required=False, help="Plot the training vs Validation error on k-folds only", nargs='?', const=True ,default=False)

    options, _ = parser.parse_known_args()
    dataset = _parse_data(options.dataset)

    X = []
    Y = []
    for data in dataset[1:]:
        x, y = data[:-1], data[-1]
        X.append(list(map(float, x))), Y.append(float(y))

    Y = np.where(np.array(Y) == 0, -1, Y)

    if bool(options.shuffle):
        X, Y = shuffle(X, Y)

    total = len(X)

    booster = AdaBoost()
    boosters = int(options.boosters)
    if options.mode == 'erm':
        err = train_validate(booster, X, Y, int(total * float(options.train_size)), boosters)
        print("Weights: {0} \nStumps: {1} \nError: {2}".format(booster.weights, booster.stumps, err/total))
    else:
        folds = int(options.folds)
        print("N-FOLDS", folds)

        mean_train_err = []
        mean_validation_err = []
        if options.plot:
            for boost in range(1, boosters+1):
                print("Step: ", boost)
                train_erm_err, validation_erm_err, weights, stumps = k_fold(booster, X, Y, folds, boost)

                for i in range(folds):
                    print('Fold-{0}:\n\t Error: {1}\n\t Weights: {2}\n\t Decision Stumps: {3}'
                          .format(i + 1, validation_erm_err[i], weights[i], stumps[i]))

                print("\nMean Error: ", np.mean(validation_erm_err))
                print("Mean Training Error: ", np.mean(train_erm_err))
                mean_train_err.append(np.mean(train_erm_err))
                mean_validation_err.append(np.mean(validation_erm_err))

            _, ax = plt.subplots()
            ax.plot([x for x in range(1, boosters+1)], mean_validation_err, label='Validation Error')
            ax.plot([x for x in range(1, boosters+1)], mean_train_err, label='Training Error')
            ax.legend()
            plt.show()
        else:
            train_erm_err, validation_erm_err, weights, stumps = k_fold(booster, X, Y, folds, boosters)
            print("Training Error: {}\n\tValidation Error: {}\n\tWeights: {},\n\tDecision Stumps: {}  ".format(
                  train_erm_err, validation_erm_err, weights, stumps))
            print("\n\nMean training Error: ", np.mean(train_erm_err),"Mean Validation Error: ", np.mean(validation_erm_err))

    return 0


if __name__ == '__main__':
    exit(main())
    from shutil import copy2
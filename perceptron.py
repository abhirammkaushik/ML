import argparse
import numpy as np
import time


def timeit(method):  # referred to https://stackoverflow.com/questions/889900/accurate-timing-of-functions-in-python
    """
    time the runtime of a method

    :param method: function to be timed
    :return: function object
    """
    def get_run_time(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)

        print(method.__qualname__, 'runtime: %2.2f seconds' % (time.time()-ts))
        return result

    return get_run_time


class Perceptron:
    """
    Perceptron learning algorithm
    """
    def __init__(self, epsilon=0.05, bias_factor=0.05):
        """
        initialization constructor

        :param bias_factor: constant factor to update bias
        """
        self.__bias_factor = bias_factor
        self.__weights = np.array([])
        self.__epsilon = epsilon
        self.__iteration = 0

    @property
    def weights(self):
        return self.__weights[1:]

    @property
    def bias(self):
        return self.__weights[0]

    @property
    def total_iterations(self):
        return self.__iteration

    def fit(self, X, y, max_iter=10000):
        """
        train the dataset

        :param X: the features of the dataset
        :param y: the value to be predicted
        :param max_iter: maximum number of iterations to run the perceptron algorithm
        :return: None
        """
        self.__iteration = 0
        self.__weights = np.zeros(len(X[0]) + 1)
        _length = len(X)
        err = float("inf")

        if max_iter == 0:
            max_iter = float("inf")

        while self.__iteration < max_iter and err > self.__epsilon:
            i = 0
            errors = 0
            while i < len(X):
                pred = self.predict(X[i])
                if pred != y[i]:
                    errors += 1
                    error = y[i] - pred   # when pred is 1, y[i] will be 0 and vice-versa. Therefore multiplying the
                    self.__weights[1:] += np.multiply(X[i], error)  # weights with error helps increase or decrease
                    self.__weights[0] += error * self.__bias_factor  # the weights appropriately
                i += 1
            err = errors/_length
            self.__iteration += 1

    def predict(self, X):
        """
        predict the outcome of X

        :param X: input or array of input
        :return: single prediction or array of predictions
        """
        return np.where(np.dot(X, self.__weights[1:]) + self.__weights[0] > 0, 1, 0)


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
    if train_size == len(X):
        train_size = 0
    for idx, i in enumerate(predict(X[train_size:])):
        if i != Y[idx]:
            err += 1
    return err


@timeit
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
    err = calculate_erm(model.predict, X, Y, train_size)

    return err


@timeit
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

    erm_err = []
    weights = []
    bias = []
    for i in range(n_folds):
        validation_set = Y[i]
        x = np.concatenate(X[:i] + X[i + 1:])
        y = np.concatenate(Y[:i] + Y[i + 1:])
        model.fit(x, y, n_iterations)
        err = calculate_erm(model.predict, X[i], validation_set)
        erm_err.append(err / len(X[i]))
        weights.append(list(model.weights))
        bias.append(model.bias)
        print("Fold-{0} iterations run: {1}".format(i+1, model.total_iterations))

    return erm_err, bias, weights


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--mode', required=False, help='Method to reduce loss', default='erm')
    parser.add_argument('--iterations', required=False, help='Total iterations for the perceptron to run', default=10000)
    parser.add_argument('--train_size', required=False, help='Fraction of total training set. Is between 0 and 1 and '
                                                             'is usually greater than 0.6', default=1)
    parser.add_argument('--folds', required=False, help='Total folds while performing cross-validation', default=10)
    parser.add_argument('--shuffle', required=False, help='Randomly shuffle the dataset', nargs='?', const=True, default=False)
    parser.add_argument('--epsilon', required=False, help='Error value upon reaching the training will stop', default=0.0)
    parser.add_argument('--b-factor', required=False, help='Rate at which only the bias changes while updating weights', default=1)

    options, _ = parser.parse_known_args()
    dataset = _parse_data(options.dataset)

    X = []
    Y = []
    for data in dataset[1:]:
        x, y = data[:-1], data[-1]
        X.append(list(map(float, x))), Y.append(float(y))

    if bool(options.shuffle):
        X, Y = shuffle(X, Y)

    total = len(X)

    perceptron = Perceptron(epsilon=float(options.epsilon), bias_factor=float(options.b_factor))
    iterations = int(options.iterations)

    if options.mode == 'erm':
        err = train_validate(perceptron, X, Y, int(total * float(options.train_size)), iterations)
        print("Bias: {0}, Weights: {1}, Error: {2}".format(perceptron.bias, perceptron.weights, err/total))
        print("Total iterations run: ", perceptron.total_iterations)
    else:
        folds = int(options.folds)
        print("N-FOLDS", folds)

        erm_err, bias, weights = k_fold(perceptron, X, Y, folds, iterations)

        for i in range(folds):
            print('Fold-{0}:\n\t Error: {1}\n\t Bias: {2}\n\t Weights: {3}'.format(i+1,erm_err[i], bias[i], weights[i]))
        print("\nMean Error: ", np.mean(erm_err))

    return 0


if __name__ == '__main__':
    exit(main())

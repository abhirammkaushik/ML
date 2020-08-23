import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs


class SVM:
    """
    Support Vector Machine
    """

    def __init__(self, T, _lambda=0.001):
        """

        :param T:
        :param _lambda:
        """
        self.iterations = T
        self.__lambda = _lambda
        self.__weights = np.array([])
        self.__eta = None

    @property
    def weights(self):
        """

        :return:
        """
        return self.__weights[-1]

    @property
    def eta(self):
        return self.__eta

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        sample_size = X.shape[0]
        self.__weights = [np.zeros(X.shape[1])]
        sign = 1
        for t in range(self.iterations):
            weights = self.__weights[-1][1:].copy()
            bias = self.__weights[-1][0]
            self.__eta = 1 / (t + 1)

            i = np.random.randint(sample_size)
            while y[i] != sign:
                i = np.random.randint(sample_size)
            sign *= -1

            if y[i] * np.dot(X[i], self.__weights[-1]) < 1:
                weights -= self.__eta * (2 * self.__lambda * weights - y[i] * X[i][1:])
                bias += self.__eta * (y[i])
            else:
                weights -= self.__eta * (2 * self.__lambda * weights)

            self.__weights.append(np.insert(weights, 0, bias))

    def predict(self, X):
        """

        :param X:
        :return:
        """
        return np.where(np.dot(X, self.__weights[-1]) > 0, 1, -1)


def train(X, y, model):
    model.fit(X, y)


def test(X, y, model):
    print("Test Errors ", np.where(model.predict(X) != y, 1, 0))


def draw(X, weights, positive, negative):
    plt.figure(figsize=(13, 7))
    plt.grid(True)

    plt.plot(positive[:, 0], positive[:, 1], 'o', alpha=0.75, label="+1", color='#1f77b4')
    plt.plot(negative[:, 0], negative[:, 1], 'o', alpha=0.75, label="-1", color='#ff7f0e')

    slope = -(weights[1] / weights[2])
    intercept = -(weights[0]) / weights[2]

    x, line = zip(*[(i, (slope * i + intercept)) for i in np.linspace(np.amin(X[:, :1]), np.amax(X[:, :1]))])
    sv1, sv2 = zip(*[(i + (1 / np.linalg.norm(weights[1:])), i - (1 / np.linalg.norm(weights[1:]))) for i in line])

    plt.plot(x, line, color='r')
    plt.plot(x, sv1, linestyle='--', color='k')
    plt.plot(x, sv2, linestyle='--', color='k')
    plt.title('Decision Boundary Graph for SVM', fontdict={'fontsize': 15})
    plt.legend()
    plt.show()


if __name__ == '__main__':

    n_samples = 300
    X0, y = make_blobs(n_samples=n_samples, n_features=2, centers=2, cluster_std=1.05, random_state=10)
    y = np.where(y == 0, -1, y)
    X1 = np.c_[np.ones((X0.shape[0])), X0]  # add one to the x-values to incorporate bias

    positive_x = []
    negative_x = []
    for i, label in enumerate(y):
        if label == -1:
            negative_x.append(X0[i])
        else:
            positive_x.append(X0[i])

    train_size = int(n_samples * 0.8)
    svm = SVM(1000, _lambda=0.75)
    train(X1[:train_size], y[:train_size], svm)

    print("Weights", svm.weights)
    print("final learning rate", svm.eta)

    test(X1[train_size:], y[train_size:], svm)
    draw(X0, svm.weights, np.array(positive_x), np.array(negative_x))

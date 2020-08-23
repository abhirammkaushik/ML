import argparse
import numpy as np


def euclidean(x1, x2):
    return np.sqrt(np.sum(np.square(np.subtract(x1, x2))))


def manhattan(x1, x2):
    return np.sum(np.abs(np.subtract(x1, x2)))


class KMeans:
    def __init__(self, clusters=3, epsilon=0.0001, distance_function=euclidean):
        self.__clusters = clusters
        self.__centroids = []
        self.__sets = [0] * clusters
        self.__aggregates = [0] * clusters
        self.__distance_function = distance_function
        self.__iterations = 0
        self.__epsilon = epsilon
        self.__labels = []
        self.__X = None

    @property
    def labels(self):
        return self.__labels

    @property
    def sets(self):
        return self.__sets

    @property
    def centroids(self):
        return self.__centroids

    @property
    def iterations(self):
        return self.__iterations

    def __random_selection(self):
        prev = None
        for i in range(self.__clusters):
            i = np.random.randint(len(self.__X))
            while i == prev:
                i = np.random.randint(len(self.__X))
            self.__centroids.append(self.__X[i])
            prev = i

    def __choose_clusters(self, method='random'):
        self.__random_selection()

    def fit(self, X, max_iter=200):
        self.__X = X
        prev_centroids = [0] * self.__clusters

        self.__choose_clusters('random')

        diff = 1
        self.__labels = [-1] * len(X)
        while diff > self.__epsilon and self.__iterations < max_iter:
            diff = 0
            for point_idx in range(len(X)):
                nearest = float("inf")
                prev_idx = self.__labels[point_idx]
                for idx, centroid in enumerate(self.__centroids):
                    dist = self.__distance_function(centroid, X[point_idx])
                    if dist < nearest:
                        nearest = dist
                        if prev_idx != idx:
                            self.__sets[idx] += 1
                            self.__aggregates[idx] = np.add(self.__aggregates[idx], X[point_idx])
                            if prev_idx == self.__labels[point_idx] and prev_idx != -1:
                                self.__sets[prev_idx] -= 1
                                self.__aggregates[prev_idx] = np.subtract(self.__aggregates[prev_idx], X[point_idx])
                            prev_idx = idx
                            self.__labels[point_idx] = prev_idx

            for idx, _ in enumerate(self.__centroids):
                self.__centroids[idx] = self.__aggregates[idx] / self.__sets[idx] if self.__sets[idx] != 0 else self.__centroids[idx]
                diff += euclidean(self.__centroids[idx], prev_centroids[idx])

            prev_centroids = self.__centroids.copy()
            # print(diff)
            self.__iterations += 1


def _parse_data(fil_path):  # same as assignment 1
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


def get_cluster_distributions(k, labels, y, centroids):
    cluster_distribution = {j: {0: 0, 1: 0} for j in range(k)}
    for i in range(len(Y)):
        label = labels[i]
        cluster_distribution[label][y[i]] += 1

    print("Cluster distributions\ty = 0 \t y = 1 \t % of positive diagnosis \t \t \t \t \t \tcluster centroid")
    for cluster, value_dict in cluster_distribution.items():
        c0 = value_dict[0]
        c1 = value_dict[1]

        print("\t   Cluster ", cluster, ":  ", c0, "\t ", c1, "\t\t", round(c1/(c1+c0) * 100, 5), "\t\t  ", list(centroids[cluster]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--k', required=False, help='number of clusters', default=3, type=int)
    parser.add_argument('--distance', required=False, help='Appropriate distance function', default='euclidean')
    parser.add_argument('--epsilon', required=False, help='Error value upon reaching the training will stop',
                        default=0.0001, type=float)
    parser.add_argument('--iterations', required=False, help='Total iterations for the model to run',
                        default=200, type=int)
    parser.add_argument('--normalize', required=False, help='Normalize the input data', nargs='?', const=True,
                        default=False)
    options, _ = parser.parse_known_args()
    dataset = _parse_data(options.dataset)

    X = []
    Y = []
    for data in dataset[1:]:
        x, y = data[:-1], data[-1]
        X.append(list(map(float, x))), Y.append(float(y))

    distance_functions = {'euclidean': euclidean, 'manhattan': manhattan}

    k_means = KMeans(options.k, options.epsilon, distance_functions[options.distance])

    if options.normalize:
        std = np.std(X, axis=0)
        mean = np.mean(X, axis=0)
        X = (X-mean)/std

    k_means.fit(X, options.iterations)
    print("Total iterations: ", k_means.iterations)

    get_cluster_distributions(options.k, k_means.labels, Y, k_means.centroids if not options.normalize else (k_means.centroids * std) + mean)

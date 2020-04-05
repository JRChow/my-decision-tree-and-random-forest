from random import randrange
import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    """The Random Forest ensemble."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=42, max_features="auto"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.tree_ensemble = []

    def train(self, data, labels):
        """Train the Random Forest with data bagging and feature baggging."""
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            sample_data, sample_labels = RandomForest._data_bagging(data, labels, len(labels))
            tree = DecisionTree(max_depth=self.max_depth,
                                max_features=self.max_features,
                                random_state=self.random_state)
            tree.train(sample_data, sample_labels)
            self.tree_ensemble.append(tree)

    def predict(self, data):
        pass

    @staticmethod
    def _data_bagging(data, labels, sample_size):
        """Bootstrapping the data to create a sub-sample."""
        sample_data = []
        sample_labels = []
        for _ in range(sample_size):
            rand_idx = randrange(len(data))
            sample_data.append(data[rand_idx])
            sample_labels.append(labels[rand_idx])
        return np.vstack(sample_data), sample_labels


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randint(5, size=(10, 3))
    y = np.random.randint(2, size=10)
    print(RandomForest._data_bagging(X, y, 3))
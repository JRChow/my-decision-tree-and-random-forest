from random import randrange
import numpy as np
from decision_tree import DecisionTree
from scipy.stats import mode


class RandomForest:
    """The Random Forest ensemble."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=42, max_features="auto"):
        # Hyper-parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        # Class member variables
        self.tree_ensemble = None
        self.label_set = None

    def train(self, data, labels):
        """Train the Random Forest with data bagging and feature bagging."""
        # Set class member variables
        self.tree_ensemble = []
        self.label_set = set(labels)

        # Set random seed for data bagging & feature bagging
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            sample_data, sample_labels = RandomForest._data_bagging(data, labels, len(labels))
            tree = DecisionTree(max_depth=self.max_depth,
                                max_features=self.max_features,
                                random_state=self.random_state)
            tree.train(sample_data, sample_labels)
            self.tree_ensemble.append(tree)

    def predict(self, data):
        """Predict output by taking the majority vote."""
        ensemble_votes = []
        for tree in self.tree_ensemble:
            ensemble_votes.append(tree.predict(data))
        ensemble_votes = np.vstack(ensemble_votes)
        return mode(ensemble_votes, axis=0)[0].ravel()

    @staticmethod
    def _data_bagging(data, labels, sample_size):
        """Bootstrapping the data to create a sub-sample."""
        sample_data = []
        sample_labels = []
        for _ in range(sample_size):
            rand_idx = randrange(len(data))
            sample_data.append(data[rand_idx])
            sample_labels.append(labels[rand_idx])
        return np.vstack(sample_data), np.array(sample_labels)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randint(5, size=(10, 3))
    y = np.random.randint(2, size=10)
    rf = RandomForest(max_depth=10, n_estimators=1000)
    rf.train(X, y)
    pred = rf.predict(X)
    print(f"pred = {pred}")
    print(f"true = {y}")
    print(f"acc = {sum(pred == y) / len(y)}")

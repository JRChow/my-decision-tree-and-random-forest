from random import randrange
import numpy as np

class RandomForest:
    """The Random Forest ensemble."""

    def __init__(self, n_estimators=100, max_depth=None, random_state=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.max_features = max_features
        self.trees = []

    def train(self, data, labels):
        for i in range(self.n_estimators):
            subsample = RandomForest._data_bagging(data, len(data))
        pass

    def predict(self, data):
        pass

    @staticmethod
    def _data_bagging(data, sample_size):
        """Bootstrapping the data to create a sub-sample."""
        samples = []
        for i in range(sample_size):
            rand_idx = randrange(len(data))
            samples.append(data[rand_idx])
        return np.vstack(samples)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randint(5, size=(10, 3))
    y = np.random.randint(2, size=10)
    print(RandomForest._data_bagging(X, 3))
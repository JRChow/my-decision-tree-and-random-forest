from collections import Counter
import numpy as np


class DecisionTree:
    """The Decision Tree classifier."""

    class Node:
        """Inner class: tree node."""

        def __init__(self, split_rule=None, left=None, right=None, label=None):
            self.split_rule = split_rule
            self.left = left
            self.right = right
            self.label = label

        def is_leaf(self):
            return self.label is not None

    # def __init__(self):

    # def train(self, X, y):

    # def predict(self, X):

    @staticmethod
    def _entropy(y):
        """Takes in the labels of data and compute the entropy."""
        class_counts = Counter(y).values()
        class_freq = [n_c / len(y) for n_c in class_counts]
        return -sum([p_c * np.log2(p_c) for p_c in class_freq])

    @staticmethod
    def _information_gain(feature_col, y, threshold):
        """Takes in a feature column, the labels, and a threshold, then computes the information gain of splitting."""
        entropy_before = DecisionTree._entropy(y)
        left_idx = feature_col < threshold
        right_idx = ~left_idx
        left_size = np.sum(left_idx)
        right_size = np.sum(right_idx)
        left_entropy = DecisionTree._entropy(y[left_idx])
        right_entropy = DecisionTree._entropy(y[right_idx])
        entropy_after = (left_size * left_entropy + right_size * right_entropy) / (left_size + right_size)
        return entropy_before - entropy_after


if __name__ == "__main__":
    clf = DecisionTree()
    y = np.array([0] * 20 + [1] * 10)
    x = np.array([25] * 10 + [75] * 10 + [25] * 9 + [75])
    beta = 50
    print(DecisionTree._information_gain(x, y, beta))

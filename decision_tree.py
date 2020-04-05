from collections import Counter
import numpy as np
from math import sqrt


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

    def __init__(self, max_depth=None, max_features=None, random_state=42):
        # Hyper-parameters
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.root = None

    def train(self, data, labels):
        """Train the decision tree by growing the tree."""
        # Set number of features to consider when looking for best fit
        if self.max_features is None:
            self.max_features = data.shape[1]
        elif self.max_features == "auto":
            self.max_features = int(sqrt(data.shape[1]))
        # Set random seed
        np.random.seed(self.random_state)

        # Grow tree
        self.root = self._grow_tree(data, labels, 1)

    def _grow_tree(self, data, labels, depth):
        """Grows tree recursively by splitting on the best feature and threshold."""
        if self.max_depth and depth >= self.max_depth:  # Base case: max depth reached
            return DecisionTree.Node(label=DecisionTree._mode(labels))

        feature_idx, threshold = self._find_best_split(data, labels)
        # If it's best not to split
        if feature_idx is None or threshold is None:
            return DecisionTree.Node(label=DecisionTree._mode(labels))

        # Split and grow left and right recursively
        left_idx = data[:, feature_idx] < threshold
        right_idx = ~left_idx
        return DecisionTree.Node(
            split_rule=(feature_idx, threshold),
            left=self._grow_tree(data[left_idx], labels[left_idx], depth + 1),
            right=self._grow_tree(data[right_idx], labels[right_idx], depth + 1)
        )

    def _find_best_split(self, data, labels):
        """Finds the best split rule for a node. Returns None if it's best not to split."""
        n, d = data.shape
        best_split_rule = (None, None)
        # Record the entropy before any splitting
        class_counts = dict(Counter(labels))
        min_entropy = DecisionTree._entropy(class_counts)
        # For each candidate feature
        feature_candidates = np.random.choice(range(d), self.max_features, replace=False)
        for feature_idx in feature_candidates:
            # Sort the feature column
            col_sorted, y_sorted = zip(*sorted(zip(data[:, feature_idx], labels)))
            # Initialize left and right class counts
            left_class_counts = dict.fromkeys(class_counts.keys(), 0)
            right_class_counts = class_counts.copy()
            # For each split within a feature
            for i in range(1, n):
                # Update class counts at every split
                cls = y_sorted[i - 1]
                left_class_counts[cls] += 1
                right_class_counts[cls] -= 1
                # Skip equal consecutive values
                if col_sorted[i] == col_sorted[i - 1]:
                    continue
                # Calculate average entropy at this split
                left_entropy = DecisionTree._entropy(left_class_counts)
                right_entropy = DecisionTree._entropy(right_class_counts)
                wa_entropy = (i * left_entropy + (n - i) * right_entropy) / n
                # Update best split
                if wa_entropy < min_entropy:
                    min_entropy = wa_entropy
                    threshold = (col_sorted[i - 1] + col_sorted[i]) / 2
                    best_split_rule = (feature_idx, threshold)
        return best_split_rule

    def predict(self, data):
        return [self._predict_one_point(point) for point in data]

    def _predict_one_point(self, data_point):
        """Given a data point, traverse the tree to find the best label."""
        node = self.root
        while not node.is_leaf():
            feature_idx, threshold = node.split_rule
            if data_point[feature_idx] < threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    @staticmethod
    def _entropy(class_counts):
        """Calculate entropy based on class counts."""
        counts_ls = class_counts.values()
        class_freq = [n_c / sum(counts_ls) for n_c in counts_ls]
        return -sum([p_c * np.log2(p_c) if p_c != 0 else 0 for p_c in class_freq])

    @staticmethod
    def _mode(ls):
        """Get the first mode of a list."""
        return Counter(ls).most_common(1)[0][0]


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randint(5, size=(10, 3))
    y = np.random.randint(2, size=10)
    clf = DecisionTree(max_depth=5)
    clf.train(X, y)
    pred = clf.predict(X)
    print(f"pred={pred}")
    print(f"true={y}")
    print(f"acc = {sum(y == pred) / len(y)}")

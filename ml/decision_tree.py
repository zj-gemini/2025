import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, List, Optional, Tuple
import argparse


def train_test_split_custom(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Custom implementation of train_test_split."""
    if random_state:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_samples = int(n_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def accuracy_score_custom(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Custom implementation of accuracy_score."""
    return np.sum(y_true == y_pred) / len(y_true)


# --- Decision Tree Implementation ---


class Node:
    """Represents a node in the decision tree."""

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        *,
        value: Optional[int] = None,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Prediction value if it's a leaf node

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTreeFromScratch:
    """A simple Decision Tree Classifier implemented from scratch."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "entropy",
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        # For this implementation, we'll stick to entropy as requested
        self.criterion = criterion
        self.root: Optional[Node] = None
        self.random_state = random_state

    def _entropy(self, y: np.ndarray) -> float:
        """Calculates the entropy of a set of labels."""
        # Note: np.bincount is an efficient way to get a frequency histogram,
        # but it requires the input array 'y' to contain only non-negative integers.
        if len(y) == 0:
            return 0.0
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _calculate_child_entropy(
        self, y: np.ndarray, left_idxs: np.ndarray, right_idxs: np.ndarray
    ) -> float:
        """
        Calculates the weighted average entropy of the children for a given split.
        """
        # Weighted average of children's entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return child_entropy

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Finds the best feature and threshold to split the data."""
        best_gain = -1.0
        split_idx: Optional[int] = None
        split_thresh: Optional[float] = None
        n_features = X.shape[1]

        # Calculate parent entropy once per node split to avoid redundant calculations.
        parent_entropy = self._entropy(y)

        # Create a list of feature indices to iterate over.
        # Sklearn shuffles features to break ties, even with splitter='best'.
        # We mimic this behavior to align results more closely.
        feature_indices = list(range(n_features))
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(feature_indices)

        for feat_idx in feature_indices:
            X_column = X[:, feat_idx]
            unique_values = np.unique(X_column)
            # Use midpoints between unique values as potential thresholds for a more robust split.
            # This is a common practice in CART implementations like sklearn's.
            if len(unique_values) > 1:
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            else:
                thresholds = unique_values

            for threshold in thresholds:
                # Generate split indices
                left_idxs = np.argwhere(X_column <= threshold).flatten()
                right_idxs = np.argwhere(X_column > threshold).flatten()

                # If split is not valid (e.g., doesn't meet min_samples_leaf), skip.
                if (
                    len(left_idxs) < self.min_samples_leaf
                    or len(right_idxs) < self.min_samples_leaf
                ):
                    continue

                # Calculate information gain
                child_entropy = self._calculate_child_entropy(y, left_idxs, right_idxs)
                gain = parent_entropy - child_entropy

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh, best_gain

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively builds the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feat, best_thresh, best_gain = self._best_split(X, y)

        # If no useful split is found, create a leaf
        if best_gain <= 0 or best_feat is None or best_thresh is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Recurse
        left_idxs = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idxs = np.argwhere(X[:, best_feat] > best_thresh).flatten()

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _most_common_label(self, y: np.ndarray) -> Optional[int]:
        """Finds the most common label in a set of labels."""
        counter = Counter(y)
        if not counter:
            return None
        most_common = counter.most_common(1)[0][0]
        return most_common

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Trains the decision tree."""
        self.root = self._grow_tree(X, y)

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Optional[int]:
        """Traverses the tree to predict a single sample."""
        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions for a set of samples."""
        return np.array([self._traverse_tree(x, self.root) for x in X])


def print_tree(
    node: Optional[Node], feature_names: Optional[List[str]] = None, indent: str = ""
) -> None:
    """Prints the structure of the decision tree."""
    if not node or node.is_leaf_node():
        print(f"{indent}class: {node.value}")
        return

    feature_name = (
        feature_names[node.feature_index]
        if feature_names
        else f"feature_{node.feature_index}"
    )
    print(f"{indent}|--- {feature_name} <= {node.threshold:.2f}")
    print_tree(node.left, feature_names, indent + "|   ")

    print(f"{indent}|--- {feature_name} > {node.threshold:.2f}")
    print_tree(node.right, feature_names, indent + "|   ")


def run_from_scratch_implementation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    **kwargs,
) -> None:
    """Runs the from-scratch decision tree implementation."""
    print("--- Running From-Scratch Implementation ---")

    clf = DecisionTreeFromScratch(**kwargs)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score_custom(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    print("\nDecision Tree Structure:\n")
    print_tree(clf.root, feature_names=feature_names)


def run_sklearn_implementation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    **kwargs,
) -> None:
    """Runs the scikit-learn decision tree implementation."""
    print("--- Running scikit-learn Implementation ---")
    # Local imports for sklearn to keep dependencies encapsulated
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score

    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = sklearn_accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    print("\nDecision Tree Structure:\n")
    tree_text = export_text(clf, feature_names=feature_names)
    print(tree_text)


def main() -> None:
    # 1. Set up argument parser to choose implementation
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Decision Tree classifier."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use the from-scratch (raw) implementation instead of scikit-learn.",
    )
    args = parser.parse_args()

    # 2. Create and prepare the dataset
    data = {
        "age": [22, 35, 47, 19, 31, 53, 41, 27, 38, 60, 25, 45, 33, 50, 29],
        "income": [40, 70, 90, 25, 80, 120, 95, 50, 85, 130, 42, 88, 76, 110, 54],
        "score": [88, 65, 70, 92, 60, 55, 80, 90, 62, 50, 85, 68, 59, 53, 91],
        "buys_product": [
            "No",
            "No",
            "Yes",
            "No",
            "No",
            "Yes",
            "No",
            "No",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "No",
            "Yes",
            "Yes",
        ],
    }
    df = pd.DataFrame(data)

    # By convention, 'X' (uppercase) is used for the feature matrix (inputs)
    # and 'y' (lowercase) is for the target vector (labels).
    X = df[["age", "income", "score"]]
    y_encoded = df["buys_product"].apply(lambda x: 1 if x == "Yes" else 0)
    feature_names = list(X.columns)

    # Convert to NumPy arrays for both implementations
    X_np = X.to_numpy()
    y_np = y_encoded.to_numpy()

    # 3. Define shared hyperparameters for both models
    # These align with the key parameters of sklearn's DecisionTreeClassifier
    hyperparameters = {
        "criterion": "entropy",
        "max_depth": 3,
        "random_state": 42,
        "min_samples_split": 2,  # Default for both
        "min_samples_leaf": 1,  # Default for sklearn
    }

    # 4. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_custom(
        X_np, y_np, test_size=0.3, random_state=42
    )

    # 5. Run the selected implementation
    if args.raw:
        run_from_scratch_implementation(
            X_train, X_test, y_train, y_test, feature_names, **hyperparameters
        )
    else:
        run_sklearn_implementation(
            X_train, X_test, y_train, y_test, feature_names, **hyperparameters
        )


if __name__ == "__main__":
    main()

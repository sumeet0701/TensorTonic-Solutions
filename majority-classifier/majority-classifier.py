import numpy as np
from collections import Counter
def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    test_data_set = len(X_test)

    count = Counter(y_train)
    max_class = max(count, key=count.get)
    max_count = count[max_class]

    return [max_class] * test_data_set
    
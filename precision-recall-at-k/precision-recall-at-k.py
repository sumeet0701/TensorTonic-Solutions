def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """

    # Edge cases
    if k == 0:
        return [0, 0]

    # Take top-k recommendations
    top_k = recommended[:k]

    # Convert to sets for efficient intersection
    top_k_set = set(top_k)
    relevant_set = set(relevant)

    # Compute intersection
    intersection = top_k_set.intersection(relevant_set)
    num_hits = len(intersection)

    # Precision@k
    precision = num_hits / k

    # Recall@k
    recall = num_hits / len(relevant_set) if len(relevant_set) > 0 else 0

    return [precision, recall]
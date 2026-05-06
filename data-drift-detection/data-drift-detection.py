def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    import numpy as np

    ref_array = np.array(reference_counts, dtype=float)
    prod_array = np.array(production_counts, dtype=float)

    # Normalize to probability distributions
    p = ref_array / ref_array.sum()
    q = prod_array / prod_array.sum()

    # Compute TVD
    tvd = 0.5 * np.sum(np.abs(p - q))
    return {
        "score": float(tvd),
        "drift_detected": bool(tvd > threshold)
    }
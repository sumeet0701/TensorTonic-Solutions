from numpy import dot
from numpy.linalg import norm

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    cos_sim = dot(x1, x2)/(norm(x1)*norm(x2))

    if label == 1:
        return 1-cos_sim
    else:
        return max(0, (cos_sim-margin))


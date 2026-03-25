import numpy as np

def broadcast_ops(X: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Computes (X + b) * w using broadcasting.
    
    Args:
        X: Input matrix of shape (N, D)
        b: Bias vector of shape (D,)
        w: Weight vector of shape (N,)
        
    Returns:
        Resulting matrix of shape (N, D)
    """
    w = w.reshape(-1, 1)
    result = (X + b) * w
    
    return result
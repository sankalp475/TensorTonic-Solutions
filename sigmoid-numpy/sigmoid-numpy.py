import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    sig_z = lambda z: 1/(1+np.exp(-z))
    np_x = np.asarray(x, dtype=float)
    return sig_z(np_x)
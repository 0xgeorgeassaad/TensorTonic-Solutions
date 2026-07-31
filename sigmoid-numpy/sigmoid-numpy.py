import numpy as np

def stable_sigmoid(x):
    x = np.asarray(x, dtype=float)

    out = np.empty_like(x)

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1 + exp_x)

    return out

def unstable_sigmoid(x):
    # numerically unstable when x << 0, as e^(-x) overflows
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # return unstable_sigmoid(x)
    return stable_sigmoid(x)
    

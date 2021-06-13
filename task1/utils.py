# utils.py

import numpy as np



def f_to_m(f):
    """Convert hz to mel."""
    m = 2595. * np.log10(1 + f / 700.)
    return m

def m_to_f(m):
    """Convert mel to hz."""
    f = 700. * (np.power(10., m / 2595.) - 1)
    return f
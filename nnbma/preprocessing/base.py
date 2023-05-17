import numpy as np

__all__ = ['id', 'log10', 'pow10']

def id(t): return t

def log10(t): return np.log10(t)

def pow10(t): return 10**t

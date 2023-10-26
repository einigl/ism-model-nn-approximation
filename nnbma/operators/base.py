import numpy as np

__all__ = ["id", "log10", "pow10", "asinh", "sinh"]


def id(t: np.ndarray):
    return t


def log10(t: np.ndarray):
    return np.log10(t)


def pow10(t: np.ndarray):
    return 10**t


def asinh(t: np.ndarray, a: float = 1.0):
    return a * np.arcsinh(t / a)


def sinh(t: np.ndarray, a: float = 1.0):
    return a * sinh(t / a)

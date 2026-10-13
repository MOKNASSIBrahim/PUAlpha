
import numpy as np


def radial_grid(R: float, Nr: int, zeta: float = 0.95) -> np.ndarray:
    """Hyperbolic-tangent stretched radial grid from axis (r=0) to wall (r=R)."""
    j    = np.arange(1, Nr + 1, dtype=np.float64)
    beta = 0.5 * np.log((1.0 + zeta) / (1.0 - zeta))
    return (R / zeta) * np.tanh(beta * (j - 1.0) / (Nr - 1.0))


def radial_metrics(R: float, Nr: int, zeta: float = 0.95):
    """Return (r, dr, rmid, xi_mid) for the stretched radial grid."""
    r    = radial_grid(R, Nr, zeta)
    dr   = r[1:] - r[:-1]
    rmid = 0.5 * (r[1:] + r[:-1])
    return r, dr, rmid, rmid / R

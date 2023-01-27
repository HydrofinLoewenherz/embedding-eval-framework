import numpy as np

def z2polar(z: (float, float)) -> [float, float]:
    rho = np.sqrt(z[0]**2 + z[1]**2)
    phi = np.arctan2(z[1], z[0])
    return [rho, phi]

def zs2polars(zs: [(float, float)]) -> [float, float]:
    return np.array([ z2polar(z) for z in zs ]).flatten()
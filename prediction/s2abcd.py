import numpy as np

def s2generalized_abcd(s, z0=50.0):
    # Calculates the generalized ABCD matrices from a given S-parameter matrix
    #
    # Args:
    # - s: S-parameter matrix
    # - z0: standard uniform reference impedance
    # Returns:
    # ABCD matrices

    num_of_ports = s.reshape[-1]
    k = num_of_ports // 2

    # Partitioning S matrix to 4 sub-matrices
    s11 = s[:k, :k]
    s12 = s[:k, k:]
    s21 = s[k:, :k]
    s22 = s[:k, :k]

    I = np.eye(k)

    s21_inv = np.linalg.inv(s21)

    # Calculation of ABCD matrices

    A = 0.5 * ((I + s11) @ s21_inv @ (I - s22) + s12)
    B = 0.5 * z0 * ((I + s11) @ s21_inv @ (I + s22) - s12)
    C = (0.5 / z0) * ((I - s11) @ s21_inv @ (I - s22) - s12)
    D = 0.5 * ((I - s11) @ s21_inv @ (I + s22) + s12)

    return A, B, C, D
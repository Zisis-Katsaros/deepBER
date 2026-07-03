import numpy as np

def s2generalized_abcd(s, z0=50.0):
    # Calculates the generalized ABCD matrices from a given S-parameter matrix
    #
    # Args:
    # - s: S-parameter matrix
    # - z0: standard uniform reference impedance
    # Returns:
    # ABCD matrices

    num_of_ports = s.shape[-1]
    k = num_of_ports // 2

    # Partitioning S matrix to 4 sub-matrices
    s11 = s[:, :k, :k]
    s12 = s[:, :k, k:]
    s21 = s[:, k:, :k]
    s22 = s[:, k:, k:]

    I = np.eye(k)

    s21_inv = np.linalg.pinv(s21)

    # Calculation of ABCD matrices

    A = 0.5 * ((I + s11) @ s21_inv @ (I - s22) + s12)
    B = 0.5 * z0 * ((I + s11) @ s21_inv @ (I + s22) - s12)
    C = (0.5 / z0) * ((I - s11) @ s21_inv @ (I - s22) - s12)
    D = 0.5 * ((I - s11) @ s21_inv @ (I + s22) + s12)

    return A, B, C, D

def s_dict2mat(s_dict, expected_ports=18):
    first_val = next(iter(s_dict.values()))
    num_samples = len(first_val)
    s_matrices = np.zeros((num_samples, expected_ports, expected_ports), dtype=np.complex64)

    for i in range(expected_ports):
        for j in range(i, expected_ports):
            key = f"S{i+1}{j+1}"
            if np.iscomplexobj(s_dict[key]):
                Sij = s_dict[key]
            else:
                Sij_real = s_dict[key][:, 0]
                Sij_imag = s_dict[key][:, 1]
                Sij = Sij_real + 1j * Sij_imag

            s_matrices[:, i, j] = Sij
            s_matrices[:, j, i] = Sij
    return s_matrices


def trans_param_mat2dict(matrices, prefix):
    out_dict = {}
    for i in range(matrices.shape[1]):
        for j in range(matrices.shape[2]):
            MATij = matrices[:, i, j]
            
            key = f"{prefix}{i+1}{j+1}"
            out_dict[key] = MATij
    return out_dict


def s2abcd_dict(s_dict, expected_ports=18, z0=50.0):
    s_matrices = s_dict2mat(s_dict, expected_ports=expected_ports)
    A, B, C, D = s2generalized_abcd(s_matrices, z0=z0)

    a_dict = trans_param_mat2dict(A, "A")
    b_dict = trans_param_mat2dict(B, "B")
    c_dict = trans_param_mat2dict(C, "C")
    d_dict = trans_param_mat2dict(D, "D")
    return a_dict, b_dict, c_dict, d_dict
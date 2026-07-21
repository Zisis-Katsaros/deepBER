import numpy as np
import re

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

def trans_param_dict2mat(data_dict):
    first_key = list(data_dict.keys())[0]
    match = re.match(r"^([a-zA-Z]+)", first_key)
    if not match:
        raise ValueError(f"Could not extract a string prefix from the key: {first_key}")
    prefix = match.group(1)
    
    # Dynamically determine N (expected_ports) and Matrix Format
    num_keys = len(data_dict)
    
    # Try full matrix assumption: K = N^2
    n_full = int(np.sqrt(num_keys))
    is_full = (n_full * n_full == num_keys)
    
    # Try symmetric upper-triangular assumption: K = N(N+1)/2
    n_tri = int((np.sqrt(1 + 8 * num_keys) - 1) / 2)
    is_tri = (n_tri * (n_tri + 1) // 2 == num_keys)
    
    # Resolve the format
    if is_full and is_tri:
        # Rare edge cases (like K=1 or K=36). Tie-break by checking which max key exists.
        if f"{prefix}{n_tri}{n_tri}" in data_dict:
            expected_ports, matrix_format = n_tri, "upper_triangular"
        else:
            expected_ports, matrix_format = n_full, "full"
    elif is_full:
        expected_ports, matrix_format = n_full, "full"
    elif is_tri:
        expected_ports, matrix_format = n_tri, "upper_triangular"
    else:
        raise ValueError(f"Dictionary length ({num_keys}) doesn't match a valid NxN full or upper-triangular matrix.")
    
    # Get the number of samples from the first array in the dictionary
    num_samples = data_dict[first_key].shape[0]

    matrices = np.zeros((num_samples, expected_ports, expected_ports), dtype=np.complex64)

    # Reconstruct the NxN matrices based on the detected format
    if matrix_format == "upper_triangular":
        for i in range(expected_ports):
            for j in range(i, expected_ports):
                key = f"{prefix}{i+1}{j+1}"
                val = data_dict[key]
                matrices[:, i, j] = np.squeeze(val)
                matrices[:, j, i] = np.squeeze(val)  # Mirror to lower triangle
    else: # matrix_format == "full"
        for i in range(expected_ports):
            for j in range(expected_ports):
                key = f"{prefix}{i+1}{j+1}"
                val = data_dict[key]
                matrices[:, i, j] = np.squeeze(val)
    return matrices


def trans_param_mat2dict(matrices, prefix, symmetric=False):
    out_dict = {}
    if symmetric or prefix in ["L", "C", "S"]:
        for i in range(matrices.shape[1]):
            for j in range(i, matrices.shape[2]):
                MATij = matrices[:, i, j]
                
                key = f"{prefix}{i+1}{j+1}"
                out_dict[key] = MATij
    else:
        for i in range(matrices.shape[1]):
            for j in range(matrices.shape[2]):
                MATij = matrices[:, i, j]
                
                key = f"{prefix}{i+1}{j+1}"
                out_dict[key] = MATij
    return out_dict


def s2abcd_dict(s_dict, expected_ports=18, z0=50.0):
    s_matrices = trans_param_dict2mat(s_dict)
    A, B, C, D = s2generalized_abcd(s_matrices, z0=z0)

    a_dict = trans_param_mat2dict(A, "A")
    b_dict = trans_param_mat2dict(B, "B")
    c_dict = trans_param_mat2dict(C, "C")
    d_dict = trans_param_mat2dict(D, "D")
    return a_dict, b_dict, c_dict, d_dict


def s2rlcg(s, freq, lengths, z0=50.0):
    """
    # s2rlcg()
    ## Converts S-parameters to R, L, C, and G matrices for the specified frequency

    ## Args:
    - s: S-parameter matrices
    - freq: Frequency value in Hz
    - lengths: Array of lengths of the transmission lines in meters
    - z0: Reference impedance (default: 50.0 Ohms)
    ## Returns:
    - L: Inductance matrices
    - C: Capacitance matrices
    - R: Resistance matrices
    - G: Conductance matrices
    """
    num_ports = s.shape[1]
    N = num_ports // 2 # dimention of L, C matrices

    freqs = np.array([freq] * s.shape[0])  # Create an array of the same frequency for each sample
    lengths = np.asarray(lengths)[:, np.newaxis] # Ensure lengths is a column vector
    omega = 2 * np.pi * freqs[:, np.newaxis, np.newaxis]

    # Convert S-parameters to Z-parameters
    I_2N = np.eye(num_ports, dtype=np.complex64)
    Z0_mat = z0 * I_2N

    I_plus_S = I_2N + s
    I_minus_S_inv = np.linalg.pinv(I_2N - s)
    Z = Z0_mat @ I_plus_S @ I_minus_S_inv

    # Extract submatrices
    Z21 = Z[:, N:, :N]
    Z22 = Z[:, N:, N:]

    Z21_inv = np.linalg.pinv(Z21)
    A = Z21_inv @ Z22
    lambdas, V = np.linalg.eig(A)

    # Calculate the diagonal elements
    gamma_hat = np.arccosh(lambdas) / lengths
    gamma_hat = np.where(gamma_hat.real < 0, -gamma_hat, gamma_hat) # real part must be positive

    # Reconstruct the full gamma matrix
    V_inv = np.linalg.pinv(V)

    gamma_mat = (V * gamma_hat[:, np.newaxis, :]) @ V_inv

    sinh_gl = np.sinh(gamma_hat * lengths)
    sinh_gamma_l_mat = (V * sinh_gl[:, np.newaxis, :]) @ V_inv

    # Compute Characteristic Impedance Matrix Zc
    Zc = Z21 @ sinh_gamma_l_mat
    Zc_inv = np.linalg.pinv(Zc)

    # Calculate LRCG matrices
    R = np.real(Zc @ gamma_mat)
    L = np.imag(Zc @ gamma_mat) / omega
    C = np.imag(gamma_mat @ Zc_inv) / omega   
    G = np.real(gamma_mat @ Zc_inv)

    # Enforce symmetry 
    R_sym = 0.5 * (R + np.transpose(R, axes=(0, 2, 1)))
    L_sym = 0.5 * (L + np.transpose(L, axes=(0, 2, 1)))
    C_sym = 0.5 * (C + np.transpose(C, axes=(0, 2, 1)))
    G_sym = 0.5 * (G + np.transpose(G, axes=(0, 2, 1)))

    # float32 conversion
    R = R_sym.astype(np.float32)
    L = L_sym.astype(np.float32)
    C = C_sym.astype(np.float32)
    G = G_sym.astype(np.float32)

    return R, L, C, G


def s2rlcg_dict(s, freq, lengths, z0=50.0):
    """
    # s2rlcg_dict()
    ## Converts S-parameters to R, L, C, and G matrices and returns them as dictionaries for the specified frequency

    ## Args:
    - s: S-parameter matrices
    - freq: Frequency value in Hz
    - lengths: Array of lengths of the transmission lines in meters
    - z0: Reference impedance (default: 50.0 Ohms)
    ## Returns:
    - r_dict: Dictionary of resistance matrices
    - l_dict: Dictionary of inductance matrices
    - c_dict: Dictionary of capacitance matrices
    - g_dict: Dictionary of conductance matrices
    """
    R, L, C, G = s2rlcg(s, freq, lengths, z0=z0)
    
    r_dict = trans_param_mat2dict(R, "R")
    l_dict = trans_param_mat2dict(L, "L")
    c_dict = trans_param_mat2dict(C, "C")
    g_dict = trans_param_mat2dict(G, "G")
    
    return r_dict, l_dict, c_dict, g_dict


def s_param_imag_part_hilbert_construction(s: np.ndarray, num_og_freq: int, K: int=1):
    Nd, Dy, seq_len = s.shape
    NM = seq_len - 1
    N = num_og_freq - 1
    
    s_rev = np.flip(s[:, :, 1:], axis=-1)  # Reverse the sequence dimension
    s_double_sided = np.concatenate((s, s_rev), axis=-1)

    Y_tilde = np.fft.fft(s_double_sided, axis=-1)
    og_length = Y_tilde.shape[-1]

    padded_length = og_length * K

    Z = np.zeros((Nd, Dy, padded_length), dtype=np.complex64)
    Z[:, :, 0] = Y_tilde[:, :, 0]
    Z[:, :, 1:seq_len] = 2 * Y_tilde[:, :, 1:seq_len]

    # IFFT to transform back and extract real and imaginary parts
    z_analytic = np.fft.ifft(Z, axis=-1)

    s_cel_full_real = K * np.real(z_analytic)
    s_cel_full_imag = -K * np.imag(z_analytic)

    truncation_idx = N*K + 1

    s_cell_real = s_cel_full_real[:, :, :truncation_idx]
    s_cell_imag = s_cel_full_imag[:, :, :truncation_idx]

    return s_cell_real + 1j * s_cell_imag

    


    


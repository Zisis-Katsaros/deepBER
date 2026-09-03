import numpy as np
import re

def s2generalized_abcd(s, z0=50.0):
    """
    # s2generalized_abcd()
    ##  Calculates the generalized ABCD matrices from a given S-parameter matrix

    ## Args:
    - s: S-parameter matrix dimensions: (num_samples, 2k, 2k)
    - z0: standard uniform reference impedance
    ## Returns:
    - ABCD matrices
    """

    num_of_ports = s.shape[-1]
    k = num_of_ports // 2

    # Partitioning S matrix to 4 sub-matrices
    s11 = s[:, :k, :k]
    s12 = s[:, :k, k:]
    s21 = s[:, k:, :k]
    s22 = s[:, k:, k:]

    I = np.eye(k)

    s21_inv = np.linalg.inv(s21) 

    # Calculation of ABCD matrices
    A = 0.5 * ((I + s11) @ s21_inv @ (I - s22) + s12)
    B = 0.5 * z0 * ((I + s11) @ s21_inv @ (I + s22) - s12)
    C = (0.5 / z0) * ((I - s11) @ s21_inv @ (I - s22) - s12)
    D = 0.5 * ((I - s11) @ s21_inv @ (I + s22) + s12)

    return A, B, C, D


def abcd2s(A, B, C, D, z0=50.0):
    """
    # abcd2s()
    ## Calculates the S-parameter matrix from generalized ABCD block matrices.

    ## Args:
    - A, B, C, D: Sub-matrices of the ABCD (transmission) matrix. Expected shape: (num_samples, k, k) or (k, k)
    - z0: Standard uniform reference impedance
    ## Returns:
    - S: Full assembled S-parameter matrix. 
         Output shape: (num_samples, 2k, 2k) or (2k, 2k).
    """
    # Normalize B and C by the reference impedance
    Bn = B / z0
    Cn = C * z0
    
    # Calculate the common inverse term E = (A + B_n + C_n + D)^-1
    # We use np.linalg.inv which naturally handles batched 3D arrays
    sum_ABCD = A + Bn + Cn + D
    E = np.linalg.inv(sum_ABCD)
    
    # Calculate the 4 S-parameter sub-matrices using multi-port generalized formulas
    # Note: The order of matrix multiplication (@) is critical here.
    S11 = (A + Bn - Cn - D) @ E
    S21 = 2 * E
    S22 = E @ (-A + Bn - Cn + D)
    S12 = (A - Bn) - (A + Bn) @ E @ (A - Bn + Cn - D)
    
    # Assemble the full S-matrix
    # Concatenate horizontally (axis=-1) to form [S11 S12] and [S21 S22]
    row1 = np.concatenate((S11, S12), axis=-1)
    row2 = np.concatenate((S21, S22), axis=-1)
    
    # Concatenate vertically (axis=-2) to form the complete 18x18 S-matrix
    S = np.concatenate((row1, row2), axis=-2)
    return S



def s_to_t(s_matrices: np.ndarray) -> np.ndarray:
    """Converts standard S-parameters to Cascading T-parameters."""
    N = s_matrices.shape[1]
    n = N // 2
    t_matrices = np.zeros_like(s_matrices)
    
    S11, S12 = s_matrices[:, :n, :n], s_matrices[:, :n, n:]
    S21, S22 = s_matrices[:, n:, :n], s_matrices[:, n:, n:]
    
    S21_inv = np.linalg.inv(S21)
    
    t_matrices[:, :n, :n] = S12 - S11 @ S21_inv @ S22
    t_matrices[:, :n, n:] = S11 @ S21_inv
    t_matrices[:, n:, :n] = -S21_inv @ S22
    t_matrices[:, n:, n:] = S21_inv
    return t_matrices

def t_to_s(t_matrices: np.ndarray) -> np.ndarray:
    """Converts Cascading T-parameters back to standard S-parameters."""
    N = t_matrices.shape[1]
    n = N // 2
    s_matrices = np.zeros_like(t_matrices)
    
    T11, T12 = t_matrices[:, :n, :n], t_matrices[:, :n, n:]
    T21, T22 = t_matrices[:, n:, :n], t_matrices[:, n:, n:]
    
    T22_inv = np.linalg.inv(T22)
    
    s_matrices[:, :n, :n] = T12 @ T22_inv
    s_matrices[:, :n, n:] = T11 - T12 @ T22_inv @ T21
    s_matrices[:, n:, :n] = T22_inv
    s_matrices[:, n:, n:] = -T22_inv @ T21
    return s_matrices

def cascade_s_matrices(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> np.ndarray:
    """Cascades 3 S-parameter matrices from left to right (Unshield -> Shield -> Unshield)."""
    t1, t2, t3 = s_to_t(s1), s_to_t(s2), s_to_t(s3)
    t_total = t1 @ t2 @ t3
    return t_to_s(t_total)


def trans_param_dict2mat(data_dict):
    """
    # trans_param_dict2mat()
    ## Converts a dictionary of transmission parameters (S, R, L, C, G, A, B, C, D) into a 3D matrix format
    ## Args:
    - data_dict: Dictionary with keys '*11', '*12', ..., '*NN' where * is the prefix and values are 2D arrays of shape (num_samples, 1) or (num_samples,)
    ## Returns:
    - matrices: 3D numpy array of shape (num_samples, N, N)
    """

    # Extract prefix from the first key
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
    """
    # trans_param_mat2dict()
    ## Converts a 3D matrix of transmission parameters (S, R, L, C, G, A, B, C, D) into a dictionary format
    ## Args:
    - matrices: 3D numpy array of shape (num_samples, N, N)
    - prefix: String to be used as the prefix for the dictionary keys
    - symmetric: Boolean indicating if the matrix is symmetric
    ## Returns:
    - out_dict: Dictionary with keys '*11', '*12', ..., '*NN' where * is the prefix and values are 2D arrays. Output dictionary contains only unique elements if input issymmetric
    """
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
    """
    # s2abcd_dict()
    ## Converts S-parameter dictionary to ABCD parameter dictionaries
    ## Args:
    - s_dict: Dictionary with keys '*11', '*12', ..., '*NN' where * is the prefix and values are 2D arrays of shape (num_samples, 1) or (num_samples,)
    - expected_ports: Number of ports of the equivalent circuit
    - z0: Standard uniform reference impedance
    ## Returns:
    - a_dict, b_dict, c_dict, d_dict: Dictionaries containing the ABCD parameters
    """
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
    - z0: Standard uniform reference impedance
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
    - z0: Standard uniform reference impedance
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


def combine_shielded_and_unshielded_portions(s_dict_shielded, s_dict_unshielded, cascade_using="T"):
    # Convert the S-parameter dictionaries to matrices
    s_mat_shielded = trans_param_dict2mat(s_dict_shielded)
    s_mat_unshielded = trans_param_dict2mat(s_dict_unshielded)

    if cascade_using == "T":
        # Calculate the total T matrix: Unshielded - Shielded - Unshielded
        s_mat_total = cascade_s_matrices(s_mat_unshielded, s_mat_shielded, s_mat_unshielded)
    elif cascade_using == "ABCD":
        # Convert the S-parameter matrices to ABCD matrices
        A_shielded, B_shielded, C_shielded, D_shielded = s2generalized_abcd(s_mat_shielded)
        A_unshielded, B_unshielded, C_unshielded, D_unshielded = s2generalized_abcd(s_mat_unshielded)

        # Create the transmission matrices
        ABCD_shielded_row1 = np.concatenate((A_shielded, B_shielded), axis=-1)
        ABCD_shielded_row2 = np.concatenate((C_shielded, D_shielded), axis=-1)
        ABCD_shielded = np.concatenate((ABCD_shielded_row1, ABCD_shielded_row2), axis=-2)

        ABCD_unshielded_row1 = np.concatenate((A_unshielded, B_unshielded), axis=-1)
        ABCD_unshielded_row2 = np.concatenate((C_unshielded, D_unshielded), axis=-1)
        ABCD_unshielded = np.concatenate((ABCD_unshielded_row1, ABCD_unshielded_row2), axis=-2)

        # Calculate the total transmission matrix: Unshielded - Shielded - Unshielded
        ABCD_total = np.matmul(ABCD_unshielded, ABCD_shielded)
        ABCD_total = np.matmul(ABCD_total, ABCD_unshielded)

        # Convert the total transmission matrix back to S-parameters
        s_mat_total = abcd2s(ABCD_total[:, :9, :9], ABCD_total[:, :9, 9:], ABCD_total[:, 9:, :9], ABCD_total[:, 9:, 9:])

    # Convert the total S-parameter matrix back to a dictionary
    s_dict_total = trans_param_mat2dict(s_mat_total, "S")
    return s_dict_total
    


    


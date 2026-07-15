import numpy as np
from scipy.linalg import expm
from prediction.s2abcd import trans_param_dict2mat

def calculate_s_coarse_matrices(l_matrices, c_matrices, freqs_ghz, lengths, z0=50.0, r_matrices=None, g_matrices=None):
    num_samples, N, _ = l_matrices.shape
    num_freqs = len(freqs_ghz)

    # Prepare 4D Broadcasting Shapes
    freqs_hz = np.atleast_1d(freqs_ghz) * 1e9
    
    # Reshape omega to (1, num_freqs, 1, 1)
    omega = 2 * np.pi * freqs_hz.reshape(1, num_freqs, 1, 1)
    
    # Reshape lengths to (num_samples, 1, 1, 1)
    lengths = np.asarray(lengths).reshape(num_samples, 1, 1, 1)

    # Reshape L, C, R, G to (num_samples, 1, N, N)
    l_reshaped = l_matrices[:, np.newaxis, :, :]
    c_reshaped = c_matrices[:, np.newaxis, :, :]
    
    if r_matrices is None:
        r_reshaped = np.zeros_like(l_reshaped, dtype=complex)
    else:
        r_reshaped = r_matrices[:, np.newaxis, :, :]
        
    if g_matrices is None:
        g_reshaped = np.zeros_like(c_reshaped, dtype=complex)
    else:
        g_reshaped = g_matrices[:, np.newaxis, :, :]
    
    # Calculate Z and Y matrices (num_samples, num_freqs, N, N)
    Z = r_reshaped + 1j * omega * l_reshaped
    Y = g_reshaped + 1j * omega * c_reshaped

    # Construct block matrix A (num_samples, num_freqs, 2N, 2N)
    A = np.zeros((num_samples, num_freqs, 2*N, 2*N), dtype=complex)
    A[:, :, 0:N, N:2*N] = -Z
    A[:, :, N:2*N, 0:N] = -Y

    # Calculate chain parameter matrix phi = e^(A*l)
    A_l = A * lengths
    
    A_l_flat = A_l.reshape(-1, 2*N, 2*N)
    phi_flat = np.array([expm(mat) for mat in A_l_flat])
    phi = phi_flat.reshape(num_samples, num_freqs, 2*N, 2*N)

    # Slice phi into sub-matrices (num_samples, num_freqs, N, N)
    phi11 = phi[:, :, 0:N, 0:N]
    phi12 = phi[:, :, 0:N, N:2*N]
    phi21 = phi[:, :, N:2*N, 0:N]
    phi22 = phi[:, :, N:2*N, N:2*N]

    # Define reference impedance matrix Z0
    Z0_mat = z0 * np.eye(N, dtype=complex)
    Z0_inv = np.eye(N, dtype=complex) / z0

    # Denominator calculation
    D = phi11 + (phi12 @ Z0_inv) + (phi21 @ Z0_mat) + phi22
    D_inv = np.linalg.pinv(D)

    # Calculate S-parameter sub-matrices
    I_N = np.eye(N, dtype=complex)
    SAA = (phi11 + (phi12 @ Z0_inv) - (phi21 @ Z0_mat) - phi22) @ D_inv

    if r_matrices is None and g_matrices is None:
        SBB = 2 * I_N @ D_inv
        SCC = 2 * I_N @ D_inv
    else:
        SBB = 2 * ((phi11 @ phi22) - (phi12 @ phi21)) @ D_inv
        SCC = 2 * I_N @ D_inv
    SDD = (-phi11 + (phi12 @ Z0_inv) - (phi21 @ Z0_mat) + phi22) @ D_inv

    # Combine into final S-parameter matrix
    s_matrices = np.zeros((num_samples, num_freqs, 2*N, 2*N), dtype=complex)
    s_matrices[:, :, 0:N, 0:N] = SAA
    s_matrices[:, :, 0:N, N:2*N] = SBB
    s_matrices[:, :, N:2*N, 0:N] = SCC
    s_matrices[:, :, N:2*N, N:2*N] = SDD

    # Enforce symmetry
    s_matrices = 0.5 * (s_matrices + np.transpose(s_matrices, axes=(0, 1, 3, 2)))

    return s_matrices


# Might remove
def get_pki_dict(l_matrices, c_matrices, freqs_ghz, lengths, z0=50.0, r_matrices=None, g_matrices=None):
    s_matrices = calculate_s_coarse_matrices(l_matrices, c_matrices, freqs_ghz, lengths, z0=z0, r_matrices=r_matrices, g_matrices=g_matrices)
    pki_dict = trans_param_dict2mat(s_matrices)
    return pki_dict
    
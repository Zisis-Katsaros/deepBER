import numpy as np
import skrf as rf
from skrf.vectorFitting import VectorFitting

def fit_vtf_auto(freq, response, max_poles=48):
    s_data = np.asarray(response, dtype=complex).reshape(-1, 1, 1)
    ntwk = rf.Network(f=freq, s=s_data, f_unit='Hz')
    
    vf = VectorFitting(ntwk)
    vf.auto_fit(
        n_poles_init_real=2,
        n_poles_init_cmplx=4,
        n_poles_add=2,
        model_order_max=max_poles
    )
    return vf


def s_params2vtf_models(network, tx, rx, next1, next2, bit_rate):
    """
    Expects 1-based port indices:
      tx, next1, next2 in 1..N_channels
      rx in (N_channels + 1)..(2 * N_channels)
    """
    num_channels = network.number_of_ports // 2
    if not (1 <= tx <= num_channels):
        raise ValueError(f"Expected 1-based tx port in [1, {num_channels}], got {tx}")
    if not (num_channels < rx <= 2 * num_channels):
        raise ValueError(f"Expected 1-based rx port in [{num_channels + 1}, {2 * num_channels}], got {rx}")
    for name, agg in (("next1", next1), ("next2", next2)):
        if agg is not None and not (1 <= agg <= num_channels):
            raise ValueError(f"Expected 1-based {name} port in [1, {num_channels}] or None, got {agg}")

    # Configure termination and capacitance values for Tx and Rx
    R_tx = 30
    R_rx = 1e6 if bit_rate <= 12e9 else 50
    C_tx = 125e-15
    C_rx = 125e-15

    freq = network.f

    # Convert raw S-params to VTF using Nodal Analysis (0-based matrix)
    VTF_matrix = calculate_vtf_from_sparameters(network, R_tx, C_tx, R_rx, C_rx)

    # 1-based to 0-based conversion for array indexing
    tx_idx = tx - 1
    rx_idx = rx - 1

    fit_main = fit_vtf_auto(freq, VTF_matrix[:, rx_idx, tx_idx])
    fit_fext1 = fit_vtf_auto(freq, VTF_matrix[:, rx_idx, next1 - 1]) if next1 is not None else None
    fit_fext2 = fit_vtf_auto(freq, VTF_matrix[:, rx_idx, next2 - 1]) if next2 is not None else None

    return fit_main, fit_fext1, fit_fext2


def calculate_vtf_from_sparameters(network, R_tx, C_tx, R_rx, C_rx, tx_ports=None, rx_ports=None):
    freq = network.f
    num_freq = len(freq)
    num_ports = network.number_of_ports
    
    # Default to standard N-port layout (0-indexed in Python)
    if tx_ports is None:
        tx_ports = list(range(num_ports // 2))
    if rx_ports is None:
        rx_ports = list(range(num_ports // 2, num_ports))
        
    Y_matrix = network.y  # shape: (num_freq, num_ports, num_ports)
    VTF_matrix = np.zeros((num_freq, num_ports, num_ports), dtype=complex)
    
    # print(f'Applying {R_tx}-Ohm/{C_tx*1e15:.2f}-fF TX and {R_rx}-Ohm/{C_rx*1e15:.2f}-fF RX terminations...')
            
    for k in range(num_freq):
        omega = 2 * np.pi * freq[k]
        
        # Calculate admittances
        Y_tx = (1 / R_tx) + 1j * omega * C_tx
        Y_rx = (1 / R_rx) + 1j * omega * C_rx
        
        # Create a diagonal matrix of the termination admittances
        Y_term = np.zeros((num_ports, num_ports), dtype=complex)
        for p in tx_ports:
            Y_term[p, p] = Y_tx
        for p in rx_ports:
            Y_term[p, p] = Y_rx
            
        # Augment the original Y-matrix and convert back to Z
        Y_aug = Y_matrix[k] + Y_term
        Z_aug = np.linalg.inv(Y_aug)
        
        VTF_matrix[k] = Z_aug / R_tx
        
    return VTF_matrix
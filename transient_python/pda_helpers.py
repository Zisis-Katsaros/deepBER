import numpy as np
from scipy import signal
from scipy.linalg import block_diag

def timeresp(fit, V_pulse, Ts):
    """
    Simulates the transient time response of a scikit-rf VectorFitting model.
    """
    if fit is None:
        return np.zeros_like(V_pulse)

    # Flatten 1-port arrays: skrf stores residues as shape (1, n_poles)
    poles = np.ravel(fit.poles)
    residues = np.ravel(fit.residues)
    d = (
        float(np.real(np.ravel(fit.constant_coeff)[0]))
        if getattr(fit, "constant_coeff", None) is not None
        else 0.0
    )

    A_blocks = []
    B_blocks = []
    C_blocks = []

    visited = np.zeros(len(poles), dtype=bool)

    for i in range(len(poles)):
        if visited[i]:
            continue

        p = poles[i]
        c = residues[i]

        # Real pole
        if np.abs(np.imag(p)) < 1e-8:
            A_blocks.append(np.array([[np.real(p)]]))
            B_blocks.append(np.array([[1.0]]))
            C_blocks.append(np.array([[np.real(c)]]))
            visited[i] = True
        else:
            # Complex conjugate pole pair (skrf stores one pole per pair with imag > 0)
            conj_indices = np.where(
                ~visited & np.isclose(poles, np.conj(p), rtol=1e-3, atol=1e-3)
            )[0]
            if len(conj_indices) > 0 and conj_indices[0] != i:
                visited[i] = True
                visited[conj_indices[0]] = True
            else:
                visited[i] = True

            alpha, beta = np.real(p), np.imag(p)
            sigma, tau = np.real(c), np.imag(c)

            A_k = np.array([[alpha, beta], [-beta, alpha]])
            B_k = np.array([[1.0], [0.0]])
            C_k = np.array([[2.0 * sigma, 2.0 * tau]])

            A_blocks.append(A_k)
            B_blocks.append(B_k)
            C_blocks.append(C_k)

    if len(A_blocks) == 0:
        return d * V_pulse

    A = block_diag(*A_blocks)
    B = np.vstack(B_blocks)
    C = np.hstack(C_blocks)
    D = np.array([[d]])

    sys = signal.StateSpace(A, B, C, D)
    t_sim = np.arange(len(V_pulse)) * Ts

    _, y_out, _ = signal.lsim(sys, U=V_pulse, T=t_sim)
    return y_out


def perform_pda(fit_main, xtalk_fits, Ts, bit_rate, Vhi, mask_height, mask_width):
    samples_per_bit = round((1 / bit_rate) / Ts)
    num_bits = 200 
    
    # Create Unit Pulse
    t_sim = np.arange(0, num_bits * samples_per_bit) * Ts
    rise_time = 15e-12
    V_pulse = generate_trapezoidal_pulse(t_sim, Vhi, 1/bit_rate, rise_time)
    
    # Get Unit Pulse Responses
    y_main = timeresp(fit_main, V_pulse, Ts)

    y_main_aligned, shift_samples = align_unit_response(y_main, samples_per_bit)

    # Fold the aligned response into a matrix of 1 UI columns
    y_main_fold = y_main_aligned.reshape((samples_per_bit, num_bits), order='F')
    
    # Isolate the Main Cursor
    cursor_idx = np.unravel_index(np.argmax(y_main_fold), y_main_fold.shape)[1]
    y_cursor = y_main_fold[:, cursor_idx]
    
    # Calculate Intersymbol Interference (ISI)
    y_isi = y_main_fold.copy()
    y_isi[:, cursor_idx] = 0 
    
    isi_neg = np.sum(np.minimum(0, y_isi), axis=1) 
    isi_pos = np.sum(np.maximum(0, y_isi), axis=1) 
    
    # Calculate Cochannel Interference (Crosstalk)
    xtalk_neg = np.zeros(samples_per_bit)
    xtalk_pos = np.zeros(samples_per_bit)
    
    for fit in xtalk_fits:
        if fit is not None:
            y_xtalk = timeresp(fit, V_pulse, Ts)
            y_xtalk_aligned = np.roll(y_xtalk, shift_samples)
            y_xtalk_fold = y_xtalk_aligned.reshape((samples_per_bit, num_bits), order='F')
            
            xtalk_neg += np.sum(np.minimum(0, y_xtalk_fold), axis=1)
            xtalk_pos += np.sum(np.maximum(0, y_xtalk_fold), axis=1)
            
    # Construct Worst-Case Eye Edges
    s1 = y_cursor + isi_neg + xtalk_neg
    s0 = isi_pos + xtalk_pos
    
    # Extract Metrics
    metrics = {}
    metrics['eye_height'] = max(0, np.max(s1 - s0)) 
    metrics['eye_amp'] = np.max(y_cursor)
    
    # Calculate Eye Width and Jitter at V_ref
    if metrics['eye_height'] > 0:
        idx_eh = np.argmax(s1 - s0) 
        Vref = (s1[idx_eh] + s0[idx_eh]) / 2
        
        margin_to_vref = np.minimum(s1 - Vref, Vref - s0)
        
        cross_idx = []
        for i in range(len(margin_to_vref) - 1):
            y1 = margin_to_vref[i]
            y2 = margin_to_vref[i+1]
            
            if np.sign(y1) != np.sign(y2) and y1 != 0:
                slope = y2 - y1
                cross_idx.append(i - (y1 / slope))
            elif y1 == 0:
                cross_idx.append(i)
                
        # Add boundaries and find unique crossing points
        all_idx = np.unique([0] + cross_idx + [len(margin_to_vref) - 1])
        continuous_width_idx = 0
        
        for i in range(len(all_idx) - 1):
            idx1 = all_idx[i]
            idx2 = all_idx[i+1]
            idx_mid = int(round((idx1 + idx2) / 2))
            
            if margin_to_vref[idx_mid] > 0:
                continuous_width_idx += (idx2 - idx1)
                
        metrics['eye_width'] = continuous_width_idx * Ts
        metrics['jitter'] = (1 / bit_rate) - metrics['eye_width']
    else:
        metrics['eye_width'] = 0
        metrics['jitter'] = 1 / bit_rate
        
    # Check the UCIe Target Mask Rectangle
    win_samples = round(mask_width / Ts)
    metrics['passes_mask'] = False
    
    max_margin = -np.inf
    best_win_start = 0
    best_min_s1 = 0
    best_max_s0 = 0
    
    # Sliding window to see if the mask rectangle fits inside the eye
    for i in range(samples_per_bit - win_samples + 1):
        local_min_s1 = np.min(s1[i:i+win_samples])
        local_max_s0 = np.max(s0[i:i+win_samples])
        margin = local_min_s1 - local_max_s0
        
        if margin > max_margin:
            max_margin = margin
            best_win_start = i
            best_min_s1 = local_min_s1
            best_max_s0 = local_max_s0
            
    if max_margin >= mask_height:
        metrics['passes_mask'] = True
        
    metrics['mask_center_idx'] = best_win_start + round(win_samples / 2)
    metrics['mask_center_v'] = (best_min_s1 + best_max_s0) / 2
    
    return s1, s0, metrics


def generate_trapezoidal_pulse(t, Vhi, bit_period, rise_time):
    V_in = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] <= rise_time:
            V_in[i] = Vhi * (t[i] / rise_time)
        elif t[i] <= bit_period:
            V_in[i] = Vhi
        elif t[i] <= bit_period + rise_time:
            V_in[i] = Vhi * (1 - (t[i] - bit_period) / rise_time)
    return V_in


def align_unit_response(y, samples_per_bit):
    peak_idx = np.argmax(y)
    half_amp = y[peak_idx] / 2
    
    # Rising edge
    search_rise = np.arange(max(0, peak_idx - samples_per_bit), peak_idx + 1)
    rise_offset = np.argmin(np.abs(y[search_rise] - half_amp))
    rise_idx = search_rise[rise_offset]
    
    # Falling edge
    search_fall = np.arange(peak_idx, min(len(y), peak_idx + 2 * samples_per_bit + 1))
    fall_offset = np.argmin(np.abs(y[search_fall] - half_amp))
    fall_idx = search_fall[fall_offset]
    
    eye_center_idx = round((rise_idx + fall_idx) / 2)
    target_idx = round(samples_per_bit / 2)
    
    current_idx_in_ui = eye_center_idx % samples_per_bit
    shift_samples = int(target_idx - current_idx_in_ui)
    
    y_aligned = np.roll(y, shift_samples)
    return y_aligned, shift_samples
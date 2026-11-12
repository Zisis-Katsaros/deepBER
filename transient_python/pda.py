import numpy as np
import skrf as rf
from transient_python.pda_helpers import perform_pda
from transient_python.freq2time import s_params2vtf_models
from transient_python.transient_visualization import plot_pda_eye

def run_pda_evaluation(s_matrices: np.ndarray, freqs_hz: np.ndarray, title_str: str = "PDA Evaluation", show_plots: bool = True, single_channel: bool = True, bit_rate: float = 16e9, 
                       fs: float = 2e12, Vhi: float = 0.625):
    """
    Runs Peak Distortion Analysis (PDA) on S-parameters.
    Identifies the worst-case channel and evaluates pass/fail against the UCIe mask.

    s_matrices: 3D numpy array of S-parameters (freqs, ports, ports)
    """
    Ts = 1 / fs

    # UCIe Specification Mask based on transmission rate
    if bit_rate <= 16e9:
        mask_height = 40e-3  # 40 mV
        mask_width = 0.75 * 1 / bit_rate  # 0.75 UI
    else:
        mask_height = 40e-3  # 40 mV
        mask_width = 0.65 * 1 / bit_rate  # 0.65 UI

    print(f"\n[PDA] Starting Peak Distortion Analysis: {title_str}")

    worst_port = None
    min_eye_height = np.inf
    min_eye_width = np.inf
    verdict = True

    # 0-based channel indices (0..8 for 9-bit MTL; index 4 = Channel 5)
    num_channels = s_matrices.shape[1] // 2
    start_port = 4 if single_channel else 0
    end_port = 4 if single_channel else (num_channels - 1)

    frequency = rf.Frequency.from_f(freqs_hz, unit="hz")
    network = rf.Network(frequency=frequency, s=s_matrices, name="MTL_9bit")

    for port in range(start_port, end_port + 1):
        # Convert to 1-based physical port numbers (Tx: 1..9, Rx: 10..18)
        tx = port + 1
        rx = tx + num_channels
        next1 = tx - 1 if tx > 1 else None
        next2 = tx + 1 if tx < num_channels else None

        print(f"\n[PDA] Evaluating Channel {tx} (Tx={tx}, Rx={rx}, FEXT={next1}, {next2})...")

        # Extract VTF rational models
        fit_main, fit_fext1, fit_fext2 = s_params2vtf_models(network, tx, rx, next1, next2, bit_rate)

        # Group crosstalk models
        xtalk_fits = [fit_fext1, fit_fext2]

        # Run PDA
        s1, s0, wire_metrics = perform_pda(fit_main, xtalk_fits, Ts, bit_rate, Vhi, mask_height, mask_width)

        if wire_metrics["eye_height"] < min_eye_height:
            min_eye_height = wire_metrics["eye_height"]
            worst_port = tx

        if wire_metrics["eye_width"] < min_eye_width:
            min_eye_width = wire_metrics["eye_width"]

        wire_verdict = wire_metrics["passes_mask"]
        verdict = verdict and wire_verdict

        if show_plots:
            plot_pda_eye(Ts, s1, s0, wire_metrics, mask_height, mask_width, plot_title=f"{title_str} - Channel {tx}")

    eye_height = min_eye_height
    eye_width = min_eye_width

    print("[PDA] Evaluation Complete")
    print("[PDA] Final Results:")
    print(f' > Verdict = {"PASS" if verdict else "FAIL"}')
    print(f" > Worst Channel = {worst_port}")
    print(f" > Min. Eye Height = {eye_height:.3f} V")
    print(f" > Min. Eye Width = {eye_width:.3e} s")

    return verdict, eye_height, eye_width
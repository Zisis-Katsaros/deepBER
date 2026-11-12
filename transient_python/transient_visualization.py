import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_pda_eye(Ts, s1, s0, metrics, mask_height, mask_width, plot_title="PDA Worst-Case Eye Diagram"):
    if s1 is None or s0 is None or len(s1) == 0 or len(s0) == 0:
        print("[PDA plot] No valid PDA waveform data to plot.")
        return None, None

    s1 = np.asarray(s1).ravel()
    s0 = np.asarray(s0).ravel()

    # Create time vector in picoseconds (ps)
    t_ps = np.arange(len(s1)) * Ts * 1e12

    # Setup the Figure
    fig, ax = plt.subplots(facecolor="white")
    ax.grid(True)

    # Plot Worst-Case Eye Bounds (Solid Red lines)
    ax.plot(t_ps, s1, "r-", linewidth=2, label=r"$s_1$ (Worst 1)")
    ax.plot(t_ps, s0, "r-", linewidth=2, label=r"$s_0$ (Worst 0)")

    # Draw the UCIe Mask
    half_w_ps = (mask_width * 1e12) / 2.0
    half_h = mask_height / 2.0

    # Uses 0-based Python indexing for mask_center_idx
    center_time = t_ps[metrics["mask_center_idx"]]
    cv = metrics["mask_center_v"]

    mask_rect = patches.Rectangle(
        (center_time - half_w_ps, cv - half_h),
        width=mask_width * 1e12,
        height=mask_height,
        facecolor=(1.0, 0.0, 0.0, 0.1),  # 10% red fill while keeping border opaque
        edgecolor="red",
        linewidth=1.5,
        label=f"Mask ({round(mask_height * 1e3)}mV x {round(mask_width * 1e12)}ps)",
    )
    ax.add_patch(mask_rect)

    # Format the Plot
    ax.set_title(plot_title)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="best")
    ax.set_xlim(t_ps[0], t_ps[-1])

    plt.tight_layout()
    plt.show()

    return fig, ax
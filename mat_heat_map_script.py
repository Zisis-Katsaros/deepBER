import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from load_set import get_grouping
import re

def plot_matrix_heatmap(data_dict, avg_per_geom=True, avg_per_element=True, x_array=None, n_geom_feats: int=7, mask_large_values=False):
    """
    Plots a heatmap for any matrix dictionary (S, L, C, ABCD).
    Automatically detects if the dictionary contains a full NxN matrix 
    or just the upper-triangular elements of a symmetric matrix.
    """
    
    # Dynamically extract the string prefix (e.g., "S", "L", "C", "ABCD")
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
    num_of_samples = data_dict[first_key].shape[0]

    if avg_per_element:
        avg_per_geom = False  # Disable avg_per_geom if avg_per_element is True

    # Initialize the generalized matrices array
    matrices = np.zeros((num_of_samples, expected_ports, expected_ports), dtype=complex)

    # Reconstruct the NxN matrices based on the detected format
    if matrix_format == "upper_triangular":
        for i in range(expected_ports):
            for j in range(i, expected_ports):
                key = f"{prefix}{i+1}{j+1}"
                val = data_dict[key]
                matrices[:, i, j] = val
                matrices[:, j, i] = val  # Mirror to lower triangle
    else: # matrix_format == "full"
        for i in range(expected_ports):
            for j in range(expected_ports):
                key = f"{prefix}{i+1}{j+1}"
                val = data_dict[key]
                matrices[:, i, j] = val

    # Averaging Logic
    if avg_per_element:
        # Take the mean of the absolute values across all samples (axis=0)
        global_avg_matrix = np.mean(np.abs(matrices.real) + 1j * np.abs(matrices.imag), axis=0)
        matrices = np.array([global_avg_matrix])
        
    if avg_per_geom:
        if x_array is None:
            raise ValueError("You must provide 'x_array' if 'avg_per_geom' is True.")
        
        # Get the lists of indices for each geometry group
        grouping_indices, unique_feats = get_grouping(x_array, n_non_unique_feats=n_geom_feats)
        num_geometries = len(grouping_indices)
        averaged_matrices = np.zeros((num_geometries, expected_ports, expected_ports), dtype=complex)
        
        for geom_idx, indices_for_this_geom in enumerate(grouping_indices):
            matrices_to_average = matrices[indices_for_this_geom]
            averaged_matrices[geom_idx] = np.mean(
                np.abs(matrices_to_average.real) + 1j * np.abs(matrices_to_average.imag), 
                axis=0
            )
            
        matrices = averaged_matrices
        plot_title_prefix = f"Geometry (Mean of {prefix})"
    else:
        plot_title_prefix = f"Sample ({prefix} Matrix)"

    # Plotting Logic 
    for idx, matrix in enumerate(matrices):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Create labels for the axes
        labels = [str(x) for x in range(1, expected_ports + 1)]
        
        mask = None
        if mask_large_values:
            floor = 0.175
            mask = np.abs(matrix) > floor

        heatmap_kwargs = {
            'cmap': 'coolwarm',
            'center': 0,
            'xticklabels': labels,
            'yticklabels': labels,
            'mask': mask,
            'annot': False, 
            'cbar_kws': {'label': 'Amplitude'}
        }
        
        # Plot Real Part
        sns.heatmap(np.real(matrix), ax=axes[0], **heatmap_kwargs)
        axes[0].set_title(f"{plot_title_prefix} {idx+1} - Real Part", fontsize=15, pad=15)
        axes[0].set_xlabel("Port j", fontsize=12)
        axes[0].set_ylabel("Port i", fontsize=12)
        
        # Plot Imaginary Part
        sns.heatmap(np.imag(matrix), ax=axes[1], **heatmap_kwargs)
        axes[1].set_title(f"{plot_title_prefix} {idx+1} - Imaginary Part", fontsize=15, pad=15)
        axes[1].set_xlabel("Port j", fontsize=12)
        axes[1].set_ylabel("Port i", fontsize=12)
        
        plt.tight_layout()
        plt.show()

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
x_array = pred_arrays_dict["x_array"]
s_dict = pred_arrays_dict["s_dict"]

l_dict = torch.load("csv_files/s_params/pt/l_dict.pt", weights_only=False)
c_dict = torch.load("csv_files/s_params/pt/c_dict.pt", weights_only=False)

plot_matrix_heatmap(l_dict, x_array=x_array, mask_large_values=False)
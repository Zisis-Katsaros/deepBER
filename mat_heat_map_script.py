import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from load_set import get_grouping
from prediction.s2abcd import trans_param_dict2mat


def plot_matrix_heatmap(data_dict, avg_per_geom=True, avg_per_element=True, x_array=None, n_geom_feats: int=7, mask_large_values=False):
    """
    Plots a heatmap for any matrix dictionary (S, L, C, ABCD).
    Automatically detects if the dictionary contains a full NxN matrix 
    or just the upper-triangular elements of a symmetric matrix.
    """
    matrices = trans_param_dict2mat(data_dict)
    expected_ports = matrices.shape[1]
    prefix = list(data_dict.keys())[0][0]  # Get the first character of the first key to determine the prefix
    
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
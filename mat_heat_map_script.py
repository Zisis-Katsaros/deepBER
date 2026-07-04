import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from load_set import get_grouping

def s_mat_heatmap(s_dict, expected_ports=18, avg_per_geom=True, avg_per_element=True, x_array = None, n_geom_feats: int = 7, mask_large_values=False):
    s11_array = s_dict["S11"]
    num_of_samples = s11_array.shape[0] 

    if avg_per_element:
        avg_per_geom = False  # Disable avg_per_geom if avg_per_element is True

    s_matrices = np.zeros((num_of_samples, expected_ports, expected_ports), dtype=complex)

    # Create S dictionary (keys: "S11", "S12"... "Sij"... "SNN", values: Sij_labels, main diagonal and upper triangle only)
    # s_array to NxN S-matrix, where N=expected ports
    col_idx = 0
    for i in range(expected_ports):
        for j in range(i, expected_ports):
            key = f"S{i+1}{j+1}"
            Sij_complex = s_dict[key]
            s_matrices[:, i, j] = Sij_complex
            s_matrices[:, j, i] = Sij_complex
 
    if avg_per_element:
        # Take the mean of the absolute values across all samples (axis=0)
        global_avg_matrix = np.mean(np.abs(s_matrices.real) + 1j * np.abs(s_matrices.imag), axis=0)
        
        # Overwrite s_matrices with an array of length 1 containing our single matrix
        s_matrices = np.array([global_avg_matrix])
        
    if avg_per_geom:
        if x_array is None:
            raise ValueError("You must provide 'x_array' if 'avg_per_geom' is True.")
        
        # Get the lists of indices for each geometry group
        grouping_indices, unique_feats = get_grouping(x_array, n_non_unique_feats=n_geom_feats)
        
        num_geometries = len(grouping_indices)
        averaged_s_matrices = np.zeros((num_geometries, expected_ports, expected_ports), dtype=complex)
        
        for geom_idx, indices_for_this_geom in enumerate(grouping_indices):
            # Isolate the matrices for this specific geometry across all frequencies
            matrices_to_average = s_matrices[indices_for_this_geom]
            
            averaged_s_matrices[geom_idx] = np.mean(np.abs(matrices_to_average.real) + 1j*np.abs(matrices_to_average.imag), axis=0)
            
        # Overwrite s_matrices with our new condensed version for plotting
        s_matrices = averaged_s_matrices
        plot_title_prefix = "Geometry"
    else:
        plot_title_prefix = "Sample"

    for matrix in s_matrices:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
        # Create labels for the axes
        labels = [str(x) for x in range(1, expected_ports + 1)]
        
        mask = None
        if mask_large_values:
                floor = 0.175
                mask = np.abs(matrix) > floor

        # colour map
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
        axes[0].set_title(r"Real Part", fontsize=15, pad=15)
        axes[0].set_xlabel("Port j", fontsize=12)
        axes[0].set_ylabel("Port i", fontsize=12)
        
        # Plot Imaginary Part
        sns.heatmap(np.imag(matrix), ax=axes[1], **heatmap_kwargs)
        axes[1].set_title(r"Imaginary Part", fontsize=15, pad=15)
        axes[1].set_xlabel("Port j", fontsize=12)
        axes[1].set_ylabel("Port i", fontsize=12)
        
        plt.tight_layout()
        plt.show()

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)
x_array = pred_arrays_dict["x_array"]
s_dict = pred_arrays_dict["s_dict"]

s_mat_heatmap(s_dict, x_array=x_array, mask_large_values=True)
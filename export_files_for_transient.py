import numpy as np
import skrf as rf
from prediction.s2abcd import s_dict2mat
import os
import csv

def create_touchstone_file(s_matrices, freqs, filename="my_dnn_output.s18p"):
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    freqs_hz = freqs * 1e9 # GHz to Hz

    # Create frequency and network objects
    frequency = rf.Frequency.from_f(freqs_hz, unit='hz')
    network = rf.Network(frequency=frequency, s=s_matrices, name='MTL_9bit')

    network.write_touchstone(filename)
    print(f"Touchstone file saved successfully to: {filename}")


def create_geometry_mapping_file(geometries: list[np.ndarray], feature_names: list[str], save_dir: str = "csv_files/transient_input_files"):
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    csv_save_path = os.path.join(save_dir, f"geometry_mapping.csv")
    
    num_features = len(geometries[0]) 
    if feature_names is None:
        feature_names = [f"Feature_{j+1}" for j in range(num_features)]
    if num_features < len(feature_names):
        feature_names = feature_names[:num_features]
        
    with open(csv_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header row
        writer.writerow(["Geom_Index"] + feature_names)
        
        # Write the data rows
        for i, geom in enumerate(geometries):
            writer.writerow([i+1] + geom.tolist()) 
    print(f"Saved CSV mapping to {csv_save_path}")


def export_files_for_transient(geometries: list[np.ndarray], feature_names: list[str], labels_dict_per_geom: list[dict], preds_dict_per_geom: list[dict], freq_arrays_per_geom: list[np.ndarray], save_dir: str="csv_files/transient_input_files"):
    """
    # export_files_for_transient()
    ## Exports the actual and predicted S-parameters for each geometry into Touchstone files for transient simulations.

    ## Args:
    - geometries: List of arrays containing unique geometric features for each geometry
    - labels_dict_per_geom: List of dictionaries containing actual S-parameters for each geometry
    - preds_dict_per_geom: List of dictionaries containing predicted S-parameters for each geometry
    - freq_arrays_per_geom: List of frequency arrays for each geometry
    - save_dir: Directory to save the Touchstone files
    ## Returns:
    *none*
    """
    # Create output directories
    actual_save_dir = save_dir + f"/actuals"
    preds_save_dir = save_dir + f"/preds"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(actual_save_dir, exist_ok=True)
    os.makedirs(preds_save_dir, exist_ok=True)
    
    # Create geometry mapping
    create_geometry_mapping_file(geometries, feature_names, save_dir=save_dir)

    # Iterate through geometries and create touchstone files of actual and predicted s-parameter matrices for each
    for geom_idx, (s_labels_dict, s_preds_dict, geom_freq_array) in enumerate(zip(labels_dict_per_geom, preds_dict_per_geom, freq_arrays_per_geom), start=1):
        if not s_labels_dict or geom_freq_array is None:
            continue
    
        # Create individual save paths
        actual_save_path = os.path.join(actual_save_dir, f"geom{geom_idx}_actual.s18p")
        pred_save_path = os.path.join(preds_save_dir, f"geom{geom_idx}_pred.s18p")

        # Convert dictionaries to matrices and create touchstone files
        s_label_matrices = s_dict2mat(s_labels_dict)
        s_pred_matrices = s_dict2mat(s_preds_dict)
        create_touchstone_file(s_label_matrices, geom_freq_array, filename=actual_save_path)
        create_touchstone_file(s_pred_matrices, geom_freq_array, filename=pred_save_path)

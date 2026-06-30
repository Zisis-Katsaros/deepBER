import numpy as np
from numpy.typing import NDArray

def latin_hypercube_order(x_array: NDArray, sample_size: int, seed: int =42):
	"""
	# latin_hypercube_order()
    ## Builds a space-filling ordering of sample indices using Latin Hypercube Sampling
	
	## Args:
	- x_array: samples
	- sample_size: number of samples to be selected
	- seed: Random seed for reproducibility 
	## Returns:
	- selected_indices: indices of selected samples
    """
	
	x_view = np.asarray(x_array)
	id_mapping = np.arange(len(x_view))

	total_size = len(x_view)
	if sample_size > total_size:
		raise ValueError("sample_size cannot exceed the number of available samples.")

	if sample_size <= 0:
		raise ValueError("sample_size must be greater than 0.")

	
	x_min = x_view.min(axis=0)
	x_max = x_view.max(axis=0)
	x_range = np.where((x_max - x_min) == 0.0, 1.0, x_max - x_min)
	x_norm = (x_view - x_min) / x_range

	# Build LHS points
	rng = np.random.default_rng(seed)
	num_of_features = x_norm.shape[1]
	lhs_points = np.empty((sample_size, num_of_features))

	for feature_idx in range(num_of_features):
		permutation = rng.permutation(sample_size)
		lhs_points[:, feature_idx] = (permutation + rng.random(sample_size)) / sample_size

	selected_indices = []
	available_mask = np.ones(total_size, dtype=bool)

	# For each LHS point, find the closest available sample in the normalized feature space and select it
	for point in lhs_points:
		available_indices = np.flatnonzero(available_mask)
		available_points = x_norm[available_mask]

        # Calculate distances and map to the idx of the original array
		distances = np.sum((available_points - point) ** 2, axis=1)
		chosen_local_index = int(np.argmin(distances))
		chosen_global_index = available_indices[chosen_local_index]
	
		selected_indices.append(chosen_global_index)
		available_mask[chosen_global_index] = False # Mark as used
	return np.asarray(selected_indices)
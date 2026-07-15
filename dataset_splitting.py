import numpy as np
from numpy.typing import NDArray
from typing import Literal
import torch

def get_grouping(x_array: NDArray, n_non_unique_feats: int = 7, round_decimals: int = 5):
	"""
	# get_grouping()
	## For a given x_array containing mixed-up grouped samples, gets how samples are grouped and each set of unique features

	## Args:
	- x_array: 2D array of mixed-up grouped samples
	- n_non_unique_feats: Number of non unique features, it is assumed that these are the frist n features
	- round_decimal: Rounding will occur after this many digits after decimal point, if None no rounding will occur
	## Returns:
	- grouping_indices: List of lists, each containing the indices of samples belonging to the same group
	- unique_feats: 2D array of unique features, each row is a unique set of features
	"""
	geometries = x_array[:, :n_non_unique_feats]

	if round_decimals is not None:
		geometries = np.round(geometries, decimals=round_decimals)

	# Get each unique set of features and their positions
	unique_feats, inverse_indices = np.unique(geometries, axis=0, return_inverse=True)

	sort_idx = np.argsort(inverse_indices)
	sorted_group_ids = inverse_indices[sort_idx]

	# Find the boundries where group ID changes
	split_indices = np.flatnonzero(np.diff(sorted_group_ids)) + 1

	grouping_indices = np.split(sort_idx, split_indices)
	return grouping_indices, unique_feats


def split_dataset(x_array: NDArray, sample_percentage: float, sampling_method: Literal["random", "lhs"] = "lhs", seed: int = 42):
	"""
	# split_dataset()
	## Splits the dataset into a smaller subset based on the specified sample percentage and sampling method.

	## Args:
	- x_array: samples
	- sample_percentage: percentage of the dataset to be sampled (between 0.0 and 1.0)
	- sampling_method: method of sampling, either "random" or "lhs" (Latin Hypercube Sampling)
	- seed: Random seed for reproducibility
	## Returns:
	- x_array: sampled subset of the original dataset
	- selected_row_indices: indices of the selected rows in the original dataset
	"""
	if not 0.0 < sample_percentage <= 1.0:
		raise ValueError("sample_percentage must be within the interval (0.0, 1.0].")
	
	# Get grouping indices and unique geometries from x_array
	grouping, unique_geoms = get_grouping(x_array)
	
	total_size = len(grouping)
	sample_size = int(total_size * sample_percentage)
	sample_size = min(total_size, max(1, sample_size))

	if sampling_method == "random":
		generator = torch.Generator().manual_seed(seed)
		sampled_group_idxs = torch.randperm(total_size, generator=generator)[:sample_size].numpy()
	elif sampling_method == "lhs":
		sampled_group_idxs = latin_hypercube_order(unique_geoms, sample_size, seed=seed)
	else:
		raise ValueError("sampling_method must be 'random' or 'lhs'.")
	
	# Get indices of selected rows in x_array and sort to maintain original order
	selected_row_indices = np.concatenate([grouping[i] for i in sampled_group_idxs])
	selected_row_indices = np.sort(selected_row_indices)

	x_array = x_array[selected_row_indices]
	return x_array, selected_row_indices


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
import csv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset_manipulation import extend_features, exclude_columns
from classification.ber_to_class import ber_to_class
from prediction.s2abcd import s2generalized_abcd
from dataset_splitting import split_dataset, latin_hypercube_order, get_grouping
from typing import Literal

def create_arrays(csv_names, target_columns, thresholds, test_names, manipulate_features = None,  
				  binary_classification = False, sample_percentage=1.0, seed=42, sampling_method="random"):
	# Creates arrays for each test dataset
	#
	# Args:
	# - csv_names: List of lists of CSV file names for each test
	# - target_columns: List of target column names for each test
	# - thresholds: List of tuples (lower_thres, upper_thres) for filtering samples by BER range for each test
	# - test_names: List of test names for printing results
	# - manipulate_features: List of booleans indicating whether to apply feature manipulation for each test
	# - sampling_method: "random" or "lhs" for subsampling the loaded dataset
	# Returns:
	# - test_info_dict: Dictionary with test names as keys and tuples (x_array, y_array, y_array_log, y_classes, thresholds, 
	# 					feature_columns) as values
	
	if manipulate_features is None:
		manipulate_features = [True] * len(csv_names)
	elif len(manipulate_features) != len(csv_names):
		raise ValueError("Length of manipulate_features must match length of csv_names.")
	
	test_info_dict = {}
	for idx, csv_batch in enumerate(csv_names):
		x_array, y_array, feature_columns = load_csv_dataset(csv_batch, target_columns=target_columns[idx])

		eps = 10**-15 # To avoid log(0)
		y_array_log = np.log10(np.clip(y_array, eps, None)).astype(np.float32)

		class_lower_thres = np.log10(np.clip(thresholds[idx][0], eps, None)).astype(np.float32)
		class_upper_thres = np.log10(np.clip(thresholds[idx][1], eps, None)).astype(np.float32)

		y_classes = ber_to_class(y_array_log, class_lower_thres, class_upper_thres, logBER=False, 
						   binary_classification=binary_classification)
		
		if manipulate_features[idx]:
			# add derived features
			x_array, feature_columns = extend_features(x_array, feature_columns, "width", "space", "/", "width_space_ratio")
			x_array, feature_columns = extend_features(x_array, feature_columns, "width", "metal_thickness", "*", "cross_sectional_area")
			x_array, feature_columns = extend_features(x_array, feature_columns, "gnd_width", "width", "/", "gnd_width_width_ratio")
			x_array, feature_columns = exclude_columns(x_array, feature_columns, columns_to_exclude=["delay"])

		if not 0.0 < sample_percentage <= 1.0:
			raise ValueError("sample_percentage must be within the interval (0.0, 1.0].")

		total_size = len(y_array)
		sample_size = int(total_size * sample_percentage)
		sample_size = min(total_size, max(1, sample_size))

		if sampling_method == "random":
			generator = torch.Generator().manual_seed(seed)
			sample_indices = torch.randperm(total_size, generator=generator)[:sample_size].numpy()
		elif sampling_method == "lhs":
			sample_indices = latin_hypercube_order(x_array, sample_size, seed=seed)
		else:
			raise ValueError("sampling_method must be 'random' or 'lhs'.")

		x_array = x_array[sample_indices]
		y_array = y_array[sample_indices]
		y_array_log = y_array_log[sample_indices]
		y_classes = y_classes[sample_indices]

		test_info_dict[test_names[idx]] = (x_array, y_array, y_array_log, y_classes, thresholds[idx], feature_columns)
	return test_info_dict


def create_param_prediction_arrays(csv_names: list[str], expected_ports:int = 18, target_columns: list[str] = [], 
								   manipulate_features: bool = True, sample_percentage: float = 1.0, seed: int = 42, 
								   sampling_method: Literal["random", "lhs"] = "random", subfolder: str = None):
	"""
	# create_param_prediction_arrays()
	## Creates arrays features and labels arrays from given csv file(s)
	
	## Args:
	- csv_names: List of CSV file names as they appear inside csv_files/ 
	- expected_ports: Expected number of ports for the equivalent circuit model 
	- target_columns: List of target column names
	- manipulate_features: Whether to apply feature manipulation (adds width to space ratio, cross sectional area and gnd width to width ratio)
	- sampling_method: "random" or "lhs" for subsampling the loaded dataset
	- subfolder: Subfolder in csv_files/ where the datasets are located
	## Returns:
	- Tuple (x_array, s_dict, a_dict, b_dict, c_dict, d_dict, feature_columns)
	"""

	# Setting up default target columns
	if target_columns == []:
			for i in range(expected_ports):
				for j in range(i, expected_ports):
					target_columns.append(f"S{i+1}{j+1}_real")
					target_columns.append(f"S{i+1}{j+1}_imag")

	# Extract inputs, labels and feature names from csv files
	x_array, s_array, feature_columns = load_csv_dataset(csv_names, target_columns=target_columns, subfolder=subfolder)

	num_of_samples = len(s_array)	

	if manipulate_features:
		x_array, feature_columns = extend_features(x_array, feature_columns, "width", "space", "/", "width_space_ratio")
		x_array, feature_columns = extend_features(x_array, feature_columns, "width", "metal_thickness", "*", "cross_sectional_area")
		x_array, feature_columns = extend_features(x_array, feature_columns, "gnd_width", "width", "/", "gnd_width_width_ratio")
	
	s_dict = {}
	s_dict["all"] = s_array

	# Initialize S-matrices
	s_matrices = np.zeros((num_of_samples, expected_ports, expected_ports), dtype=complex)

	# Create S dictionary (keys: "S11", "S12"... "Sij"... "SNN", values: Sij_labels, main diagonal and upper triangle only)
	# s_array to NxN S-matrix, where N=expected ports
	col_idx = 0
	for i in range(expected_ports):
		for j in range(i, expected_ports):
			Sij_real = s_array[:, col_idx]
			Sij_imag = s_array[:, col_idx + 1]

			Sij_complex = Sij_real + 1j * Sij_imag

			key = f"S{i+1}{j+1}"
			s_dict[key] = Sij_complex

			s_matrices[:, i, j] = Sij_complex
			s_matrices[:, j, i] = Sij_complex

			# Move to next column
			col_idx +=  2

	# Calculate ABCD parameters from S-matrix
	A, B, C, D = s2generalized_abcd(s_matrices)

	# Create ABCD dictionaries
	a_dict = {}
	b_dict = {}
	c_dict = {}
	d_dict = {}
	col_idx = 0
	for submatrices, dict, name in zip([A,B,C,D], [a_dict, b_dict, c_dict, d_dict], ["A", "B", "C", "D"]):
		# Dictionary entry with all elements of the ABCD matrices together 
		MATflat = submatrices.reshape(num_of_samples, submatrices.shape[1]**2)

		all_array = np.empty((num_of_samples, 2*submatrices.shape[1]**2))
		all_array[:, 0::2] = MATflat.real
		all_array[:, 1::2] = MATflat.imag

		dict["all"] = all_array

		# Dictionary entries for each element of the ABCD matrices
		for i in range(submatrices.shape[1]):
			for j in range(submatrices.shape[2]):
				MATij = submatrices[:, i, j]

				key = f"{name}{i+1}{j+1}"
				dict[key] = MATij	

	x_array, selected_row_indices = split_dataset(x_array, sample_percentage=sample_percentage, sampling_method=sampling_method, seed=seed)

	for dict in [s_dict, a_dict, b_dict, c_dict, d_dict]:
		for key in dict.keys():
			dict[key] = dict[key][selected_row_indices]
			
	# Return
	return x_array, s_dict, a_dict, b_dict, c_dict, d_dict, feature_columns


def load_csv_dataset(csv_names: list[str], target_columns="BER", subfolder: str =None):
	""""
	# load_csv_dataset()
	## Loads dataset from CSV file(s)

	## Args:
	- csv_names: List of CSV file names
	- target_columns: List of target column names
	- subfolder: Subfolder in csv_files/ where the datasets are located 
	## Returns:
	- x_array: 2D array of features
	- y_array: 1D array of labels
	- feature_columns: List of feature column names 
	"""

	# If target_columns is given as a single string, convert it to a list for consistent processing
	if not isinstance(target_columns, list):
		target_columns = [target_columns]
		single_target = True
	else:
		single_target = False

	x_array = []
	y_array = []
	feature_columns = None
	
	# Iterate through csv files and extract inputs, labels and feature names from each one
	for idx, name in enumerate(csv_names):
		if subfolder is not None:
			dataset_path = Path(__file__).resolve().parent / "csv_files" / subfolder / name
		else:
			dataset_path = Path(__file__).resolve().parent / "csv_files" / name
		if not dataset_path.exists():
			raise FileNotFoundError(
				f"Dataset not found at {dataset_path}. Update the path in main.py or move the file."
			)

		with open(dataset_path, mode="r", newline="") as csv_file:
			reader = csv.DictReader(csv_file)
			headers = reader.fieldnames

			for target_column in target_columns:
					if headers is None or target_column not in headers:
						raise ValueError(f"Target column '{target_column}' was not found in {dataset_path}.")

			current_feature_columns = [column for column in headers if column not in target_columns]
			
			# Validate that all CSV files have the same columns
			if idx == 0:
				feature_columns = current_feature_columns
			else:
				if set(current_feature_columns) != set(feature_columns):
					missing_in_current = set(feature_columns) - set(current_feature_columns)
					extra_in_current = set(current_feature_columns) - set(feature_columns)
					error_msg = f"Column mismatch in {dataset_path}.\n"
					if missing_in_current:
						error_msg += f"Missing columns: {missing_in_current}\n"
					if extra_in_current:
						error_msg += f"Extra columns: {extra_in_current}"
					raise ValueError(error_msg)
				
				# Ensure column order matches the first file
				if current_feature_columns != feature_columns:
					raise ValueError(f"Column order mismatch in {dataset_path}. Expected order: {feature_columns}, got: {current_feature_columns}")

			features = []
			targets = []

			if not single_target:
				y = []

			for row in reader:
				try:
					x = [row[column] for column in feature_columns]

					if single_target:
						y = float(row[target_columns[0]])
					else:
						y = [float(row[target_column]) for target_column in target_columns]
				except (TypeError, ValueError, KeyError):
					continue

				features.append(x)
				targets.append(y)

		if not features:
			raise ValueError("No valid numeric rows were found in the dataset.")

		x_batch = np.asarray(features, dtype=np.float32)
		y_batch = np.asarray(targets, dtype=np.float32)

		x_array.extend(x_batch)
		y_array.extend(y_batch)

	# Convert lists back to numpy arrays with proper shape
	x_array = np.asarray(x_array, dtype=np.float32).reshape(-1, len(feature_columns))
	y_array = np.asarray(y_array, dtype=np.float32)
	
	return x_array, y_array, feature_columns


def create_dataloader(
	x_array,
 	y_array,
 	batch_size=64,
 	seed=42,
 	ber_interval=None,
 	logBER=False,
 	standard_scale=False,
	split_method="random",
):
	# Creates dataloader
	#
	# Args:
	# - x_array: 2D array of features
	# - y_array: 1D array of labels
	# - batch_size: Batch size for dataloader
	# - seed: Random seed for reproducibility
	# - ber_interval: Tuple (min_ber, max_ber) to filter samples by BER range
	# - logBER: If true labels are log10(BER)
	# - standard_scale: If true standard scaling is applied to features
	# - split_method: "random" or "lhs" for splitting the dataset
	# Returns:
	# - dataloader: [train_data, val_data, test_data] 

	if ber_interval is not None:
		if len(ber_interval) != 2:
			raise ValueError("ber_interval must be [min_ber, max_ber].")

		min_ber, max_ber = ber_interval
		mask = np.ones_like(y_array, dtype=bool)

		if min_ber is not None:
			mask &= y_array >= float(min_ber)
		if max_ber is not None:
			mask &= y_array <= float(max_ber)

		x_array = x_array[mask]
		y_array = y_array[mask]

		if len(y_array) == 0:
			raise ValueError("No samples found inside the provided ber_interval.")

	# Log10(BER)
	if logBER:
		eps = 1e-15  # To avoid log(0)
		y_array = np.log10(np.clip(y_array, eps, None)).astype(np.float32)

	# Set split percentages
	train_percent = 0.8
	val_percent = 0.1

	total_size = len(y_array)
	train_size = int(total_size * train_percent)
	val_size = int(total_size * val_percent)

	if split_method == "random":
		generator = torch.Generator().manual_seed(seed)
		split_indices = torch.randperm(total_size, generator=generator).numpy()
	elif split_method == "lhs":
		split_indices = latin_hypercube_order(x_array, total_size, seed=seed)
	else:
		raise ValueError("split_method must be 'random' or 'lhs'.")

	train_idx = split_indices[:train_size]
	val_idx = split_indices[train_size:train_size + val_size]
	test_idx = split_indices[train_size + val_size:]

	# Standard scaling
	if standard_scale:
		# Fit scaling parameters on train split only to avoid leakage.
		train_mean = x_array[train_idx].mean(axis=0)
		train_std = x_array[train_idx].std(axis=0)
		train_std = np.where(train_std == 0.0, 1.0, train_std)
		x_array = ((x_array - train_mean) / train_std).astype(np.float32)

	train_set = TensorDataset(
		torch.from_numpy(x_array[train_idx]),
		torch.from_numpy(y_array[train_idx]),
	)
	val_set = TensorDataset(
		torch.from_numpy(x_array[val_idx]),
		torch.from_numpy(y_array[val_idx]),
	)
	test_set = TensorDataset(
		torch.from_numpy(x_array[test_idx]),
		torch.from_numpy(y_array[test_idx]),
	)

	train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)
	test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	return [train_data, val_data, test_data]


def create_param_dataloader(x_array: NDArray, y_array: NDArray, batch_size: int =64, seed: int =42, standard_scale: tuple[bool, bool]| bool = False,
							split_method: Literal["random", "lhs"] = "random", split_percentages: list[float]=[0.8, 0.1]):
	"""
	# create_param_dataloader()
	## Creates train/val/test dataloader

	## Args:
	- x_array: 2D array of features
	- y_array: 1D array of labels
	- batch_size: Batch size for dataloader
	- seed: Random seed for reproducibility
	- standard_scale: If true standard scaling is applied to features and labels using mean and std of the training data
	- split_method: "random" or "lhs" for splitting the dataset
	- split_percentages: [percentage of samples for training set, percentage of samples for validation set]
	## Returns:
	- dataloader: [train_data, val_data, test_data]
	- x_scale_params: (x_train_mean, x_train_std)
	- y_scale_params: (y_train_mean, y_train_std)
	"""

	if isinstance(standard_scale, (bool, bool)):
		scale_features, scale_labels = standard_scale
	else:
		scale_features, scale_labels = standard_scale, standard_scale

	if len(y_array) == 0:
		raise ValueError("No samples found inside the provided label array.")

	if y_array.ndim == 1:
		if y_array.dtype == np.complex64:
			y_array = np.asarray(y_array).reshape(-1, 1)
		else:
			y_array = np.asarray(y_array, dtype=np.float32).reshape(-1, 1)

	# Set split percentages
	train_percent = split_percentages[0]
	val_percent = split_percentages[1]
	if train_percent + val_percent > 1.0:
		raise ValueError("train_percent + val_percent cannot exceed 1.0")

	# Get grouping indices and unique geometries from x_array
	grouping, unique_geoms = get_grouping(x_array)

	total_size = len(grouping)
	train_size = int(total_size * train_percent)
	val_size = int(total_size * val_percent)

	if split_method == "random":
		generator = torch.Generator().manual_seed(seed)
		split_group_ids = torch.randperm(total_size, generator=generator).numpy()
	elif split_method == "lhs":
		split_group_ids = latin_hypercube_order(unique_geoms, total_size, seed=seed)
	else:
		raise ValueError("split_method must be 'random' or 'lhs'.")

	# Split grouping ids into train/val/test
	train_group_ids = split_group_ids[:train_size]
	val_group_ids = split_group_ids[train_size:train_size + val_size]
	test_group_ids = split_group_ids[train_size + val_size:]

	# Map grouping ids to original row indices (accounting for the case that val and test sets are empty)
	train_idx = np.concatenate([grouping[i] for i in train_group_ids])
	val_idx = np.concatenate([grouping[i] for i in val_group_ids]) if len(val_group_ids) > 0 else np.array([], dtype=int)
	test_idx = np.concatenate([grouping[i] for i in test_group_ids]) if len(test_group_ids) > 0 else np.array([], dtype=int)

	# Sort to maintain original order
	train_idx = np.sort(train_idx)
	val_idx = np.sort(val_idx)
	test_idx = np.sort(test_idx)

	# Standard scaling
	if scale_features:
		# Fit scaling parameters on train split only to avoid leakage.
		x_train_mean = x_array[train_idx].mean(axis=0)
		x_train_std = x_array[train_idx].std(axis=0)
		x_train_std = np.where(x_train_std == 0.0, 1.0, x_train_std)
		x_array = ((x_array - x_train_mean) / x_train_std)

		x_scale_params = (x_train_mean, x_train_std)
	else:
		x_scale_params = (0, 1)

	if scale_labels:
		y_train_mean = y_array[train_idx].mean(axis=0)
		y_train_std = y_array[train_idx].std(axis=0)
		y_train_std = np.where(y_train_std == 0.0, 1.0, y_train_std)
		y_array = ((y_array - y_train_mean) / y_train_std)

		y_scale_params = (y_train_mean, y_train_std)
	else:
		y_scale_params = (0, 1)

	train_set = TensorDataset(
		torch.from_numpy(x_array[train_idx]),
		torch.from_numpy(y_array[train_idx]),
	)
	val_set = TensorDataset(
		torch.from_numpy(x_array[val_idx]),
		torch.from_numpy(y_array[val_idx]),
	)
	test_set = TensorDataset(
		torch.from_numpy(x_array[test_idx]),
		torch.from_numpy(y_array[test_idx]),
	)

	train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)
	test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	dataloader = [train_data, val_data, test_data]
		
	return dataloader, x_scale_params, y_scale_params
	

def create_param_forward_dataloader(x_array: NDArray, batch_size: int =64, standard_scale: bool =False, 
									x_scale_params: tuple[np.ndarray, np.ndarray] = None):
	"""
	# create_param_forward_dataloader()
	## Creates dataloader for forward pass

	## Args:
	- x_array: 2D array of features
	- batch_size: Batch size for dataloader
	- standard_scale: If true standard scaling is applied to features and labels using mean and std of the training data
	- x_scale_params: (x_train_mean, x_train_std) for standard scaling of features
	## Returns:
	- dataloader: DataLoader for forward pass
	"""
	# Standard scaling
	if standard_scale:
		# Fit scaling parameters on train split only to avoid leakage.
		x_train_mean = x_scale_params[0]
		x_train_std = x_scale_params[1]
		x_array = ((x_array - x_train_mean) / x_train_std)

	dataset = TensorDataset(torch.from_numpy(x_array))
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return dataloader


def organize_dataset_for_pi_stcnn(x_array: NDArray, s_dict: dict, feature_columns: list[str]):
	"""
	# organize_dataset_for_pi_stcnn()
	## Organizes the dataset for use alongside PI-STCNN model

	## Args:
	- x_array: 2D array of features
	- s_dict: Dictionary of S-parameters with keys as "Sij" and values as complex arrays
	- feature_columns: List of feature column names
	## Returns:
	- unique_x: 2D array of unique design geometries
	- new_feature_columns: List of feature column names excluding "frequency_in_ghz"
	- y_array: 3D array of complex S-parameters with shape (num_geoms, 2*num_channels, num_freqs)
	"""
	s_dict.pop("all", None)  # Remove the "all" key if it exists

	# Locate the frequency column
	try:
		freq_idx = feature_columns.index("frequency_ghz")
	except ValueError:
		raise ValueError("The exact string 'frequency_ghz' must be present in feature_columns.")

    # Extract frequency tracking information
	freq_col = x_array[:, freq_idx]
	unique_freqs = np.unique(freq_col)
	num_freqs = len(unique_freqs)

    # Map each frequency in the dataset to a spatial index (0 to num_freqs - 1)
	freq_indices = np.searchsorted(unique_freqs, freq_col)

    # Remove the frequency column from the input features
	x_no_freq, new_feature_columns = exclude_columns(x_array, feature_columns, columns_to_exclude=["frequency_ghz"])

    # Extract unique samples (collapse the dataset to one row per unique design geometry)
    # inverse_indices maps the flat original array back to the unique geometry index
	unique_x, inverse_indices = np.unique(x_no_freq, axis=0, return_inverse=True)
	num_geoms = len(unique_x)

    # Build Y
	channel_keys = list(s_dict.keys())
	num_channels = len(channel_keys)

    # Initialize the complex 3D output tensor: (Batch_Size, Channels, Frequency_Points)
	y_array = np.zeros((num_geoms, 2*num_channels, num_freqs), dtype=np.float32)

    # Populate the tensor using advanced numpy indexing for high performance
	for c_idx, key in enumerate(channel_keys):
        # inverse_indices dictates the geometry row, freq_indices dictates the depth/sequence step
		y_array[inverse_indices, c_idx, freq_indices] = s_dict[key].real
		y_array[inverse_indices, c_idx + num_channels, freq_indices] = s_dict[key].imag
	
	return unique_x, new_feature_columns, y_array
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataset_manipulation import extend_features, exclude_columns
from classification.ber_to_class import ber_to_class


def create_arrays(csv_names, target_columns, thresholds, test_names, manipulate_features=[True, True, True, True]):
	# Creates arrays for each test dataset
	#
	# Args:
	# - csv_names: List of lists of CSV file names for each test
	# - target_columns: List of target column names for each test
	# - thresholds: List of tuples (lower_thres, upper_thres) for filtering samples by BER range for each test
	# - test_names: List of test names for printing results
	# - manipulate_features: List of booleans indicating whether to apply feature manipulation for each test
	# Returns:
	# - test_info_dict: Dictionary with test names as keys and tuples (x_array, y_array, y_array_log, y_classes, thresholds, 
	# 					feature_columns) as values
	
	test_info_dict = {}
	for idx, csv_batch in enumerate(csv_names):
		x_array, y_array, feature_columns = load_csv_dataset(csv_batch, target_column=target_columns[idx])
        
		eps = 10**-15 # To avoid log(0)
		y_array_log = np.log10(np.clip(y_array, eps, None)).astype(np.float32)

		class_lower_thres = np.log10(np.clip(thresholds[idx][0], eps, None)).astype(np.float32)
		class_upper_thres = np.log10(np.clip(thresholds[idx][1], eps, None)).astype(np.float32)

		y_classes = ber_to_class(y_array_log, class_lower_thres, class_upper_thres, logBER=False)

		if manipulate_features[idx]:
			# add derived features
			x_array, feature_columns = extend_features(x_array, feature_columns, "width", "space", "/", "width_space_ratio")
			x_array, feature_columns = extend_features(x_array, feature_columns, "width", "metal_thickness", "*", "cross_sectional_area")
			x_array, feature_columns = extend_features(x_array, feature_columns, "gnd_width", "width", "/", "gnd_width_width_ratio")
			x_array, feature_columns = exclude_columns(x_array, feature_columns, columns_to_exclude=["delay"])

		test_info_dict[test_names[idx]] = (x_array, y_array, y_array_log, y_classes, thresholds[idx], feature_columns)
	return test_info_dict


def load_csv_dataset(csv_names, target_column="BER"):
	# Loads dataset from CSV file
	#
	# Args:
	# - csv_names: List of CSV file names
	# - target_column: Name of label column
	# - exclude_columns: Feature columns to be excluded
	# Returns:
	# - x_array: 2D array of features
	# - y_array: 1D array of labels
	# - feature_columns: List of feature column names 

	x_array = []
	y_array = []
	feature_columns = None
	
	for idx, name in enumerate(csv_names):
		dataset_path = Path(__file__).resolve().parent / "csv_files" / name
		if not dataset_path.exists():
			raise FileNotFoundError(
				f"Dataset not found at {dataset_path}. Update the path in main.py or move the file."
			)

		with open(dataset_path, mode="r", newline="") as csv_file:
			reader = csv.DictReader(csv_file)
			headers = reader.fieldnames

			if headers is None or target_column not in headers:
				raise ValueError(f"Target column '{target_column}' was not found in {dataset_path}.")

			current_feature_columns = [column for column in headers if column != target_column]
			
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

			for row in reader:
				try:
					x = [float(row[column]) for column in feature_columns]
					y = float(row[target_column])
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
	test_size = total_size - train_size - val_size

	generator = torch.Generator().manual_seed(seed)
	indices = torch.randperm(total_size, generator=generator).numpy()
	train_idx = indices[:train_size]
	val_idx = indices[train_size:train_size + val_size]
	test_idx = indices[train_size + val_size:]

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

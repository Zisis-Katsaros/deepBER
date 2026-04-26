import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def load_csv_dataset(csv_path, target_column="BER", exclude_columns=[]):
	# Loads dataset from CSV file
	#
	# Args:
	# - csv_path: Path to the CSV file
	# - target_column: Name of label column
	# - exclude_columns: Feature columns to be excluded
	# Returns:
	# - x_array: 2D array of features
	# - y_array: 1D array of labels
	# - feature_columns: List of feature column names 

	with open(csv_path, mode="r", newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		headers = reader.fieldnames

		if headers is None or target_column not in headers:
			raise ValueError(f"Target column '{target_column}' was not found in {csv_path}.")

		feature_columns = [column for column in headers if column != target_column and column not in exclude_columns]

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

	x_array = np.asarray(features, dtype=np.float32)
	y_array = np.asarray(targets, dtype=np.float32)

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
		eps = 1e-15 # To avoid log(0) 
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

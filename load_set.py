import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def load_csv_dataset(csv_path, target_column="BER"):
	with open(csv_path, mode="r", newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		headers = reader.fieldnames

		if headers is None or target_column not in headers:
			raise ValueError(f"Target column '{target_column}' was not found in {csv_path}.")

		feature_columns = [column for column in headers if column != target_column]

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

def create_dataloader(x_array, y_array, ber_interval=None, logBER=False, batch_size=64, seed=42):
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

	if logBER:
		eps = 1e-12
		y_array = np.log10(np.clip(y_array, eps, None)).astype(np.float32)

	tensor_x = torch.from_numpy(x_array)
	tensor_y = torch.from_numpy(y_array)
	dataset = TensorDataset(tensor_x, tensor_y)

	total_size = len(dataset)
	train_size = int(total_size * 0.8)
	val_size = int(total_size * 0.1)
	test_size = total_size - train_size - val_size

	generator = torch.Generator().manual_seed(seed)
	train_set, val_set, test_set = random_split(
		dataset,
		[train_size, val_size, test_size],
		generator=generator,
	)

	train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)
	test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	return [train_data, val_data, test_data]

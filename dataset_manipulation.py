import numpy as np


def exclude_columns(x_array, feature_names, columns_to_exclude):
	# Removes specified columns from x_array
	#
	# Args:
	# - x_array: 2D array of features
	# - feature_names: List of feature column names
	# - columns_to_exclude: List of column names to exclude
	# Returns:
	# - x_array: Updated 2D array with excluded columns removed
	# - feature_names: Updated list of feature column names

	# Validate that all columns to exclude exist
	for col in columns_to_exclude:
		if col not in feature_names:
			raise ValueError(f"Column '{col}' not found in feature_names.")

	# Find indices of columns to keep
	indices_to_keep = [i for i, col in enumerate(feature_names) if col not in columns_to_exclude]
	
	# Filter x_array and feature_names
	x_filtered = x_array[:, indices_to_keep]
	feature_names_filtered = [feature_names[i] for i in indices_to_keep]

	return x_filtered, feature_names_filtered

def extend_features(x_array, feature_names, first_column, second_column, operation, new_feature_name):
	# Extends features by applying an operation to two existing features
	#
	# Args:
	# - x_array: 2D array of features
	# - feature_names: List of feature column names
	# - first_column: Name of the first feature column
	# - second_column: Name of the second feature column
	# - operation: Function that takes two arrays and returns a new array (e.g. lambda a, b: a * b)
	# - new_feature_name: Name of the new feature column to be added
	# Returns:
	# - x_array: Updated 2D array of features with the new feature added
	# - feature_names: Updated list of feature column names with the new feature name added

	if first_column not in feature_names or second_column not in feature_names:
		raise ValueError(f"Both columns must be in feature_names.")

	first_idx = feature_names.index(first_column)
	second_idx = feature_names.index(second_column)

	if operation == "+":
		new_feature = (x_array[:, first_idx] + x_array[:, second_idx]).reshape(-1, 1)
	elif operation == "-":
		new_feature = (x_array[:, first_idx] - x_array[:, second_idx]).reshape(-1, 1)
	elif operation == "*":
		new_feature = (x_array[:, first_idx] * x_array[:, second_idx]).reshape(-1, 1)
	elif operation == "/":
		epsilon = 1e-15
		new_feature = (x_array[:, first_idx] / (x_array[:, second_idx] + epsilon)).reshape(-1, 1)
	else:
		raise ValueError("Unsupported operation. Use one of: '+', '-', '*', '/'.")

	x_array = np.hstack((x_array, new_feature))
	feature_names.append(new_feature_name)

	return x_array, feature_names


def mock_pki(x_array, feature_names, y_array, noise_factor=0.2):
	y_std = np.std(y_array)
	noise_std = noise_factor * y_std

	noise = np.random.normal(0, noise_std, y_array.shape)
	pki = y_array + noise
	pki = np.clip(pki).reshape(-1,1)

	x_array = np.hstack((x_array, pki))
	feature_names.append("pki")

	return x_array, feature_names
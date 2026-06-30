import numpy as np
from numpy.typing import NDArray
from typing import Literal
from scipy.interpolate import interp1d


def exclude_columns(x_array: NDArray, feature_names: list[str], columns_to_exclude: list[str]):
	"""
	# exclude_columns()
	## Removes specified columns from x_array

	## Args:
	- x_array: 2D array of features
	- feature_names: List of feature column names
	- columns_to_exclude: List of column names to exclude
	## Returns:
	- x_array: Updated 2D array with excluded columns removed
	- feature_names: Updated list of feature column names
	"""

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


def extend_features(x_array: NDArray, feature_names: list[str], first_column: str, second_column: str, operation: Literal["+", "-", "*", "/"], 
					new_feature_name: str):
	"""
	# extend_features()
	## Extends features by applying an operation to two existing features (e.g. "width" "/" "space" -> "width_space_ratio")
	
	## Args:
	- x_array: 2D array of features
	- feature_names: List of feature column names
	- first_column: Name of the first feature column
	- second_column: Name of the second feature column
	- operation: +, -, * or /
	- new_feature_name: Name of the new feature column to be added
	## Returns:
	- x_array: Updated 2D array of features with the new feature added
	- feature_names: Updated list of feature column names with the new feature name added
	"""

	if first_column not in feature_names or second_column not in feature_names:
		raise ValueError(f"Both columns must be in feature_names.")

	first_idx = feature_names.index(first_column)
	second_idx = feature_names.index(second_column)

	val1 = x_array[:, first_idx].astype(np.float32)
	val2 = x_array[:, second_idx].astype(np.float32)

	if operation == "+":
		new_feature = (val1 + val2).reshape(-1, 1)
	elif operation == "-":
		new_feature = (val1 - val2).reshape(-1, 1)
	elif operation == "*":
		new_feature = (val1 * val2).reshape(-1, 1)
	elif operation == "/":
		epsilon = 1e-15
		new_feature = (val1 / val2 + epsilon).reshape(-1, 1)
	else:
		raise ValueError("Unsupported operation. Use one of: '+', '-', '*', '/'.")

	x_array = np.hstack((x_array, new_feature))
	feature_names.append(new_feature_name)

	return x_array, feature_names


def mock_pki(freqs, s_param_actual, f0: float =10.0, alpha: float =0.012, beta: float =0.05):
	"""
	# mock_pki()
	## Simulates prior knowledge input by taking the actual s parameters and warping their phase and magnitude
	"""
	# Phase error
	freqs_warped = freqs * (1 + alpha *(freqs - f0))

	# Interpolation
	interp_real = interp1d(freqs, s_param_actual.real, kind='cubic', fill_value="extrapolate")
	interp_imag = interp1d(freqs, s_param_actual.imag, kind='cubic', fill_value="extrapolate")

	s_param_warped_real = interp_real(freqs_warped)
	s_param_warped_imag = interp_imag(freqs_warped)
	s_param_warped = s_param_warped_real + 1j * s_param_warped_imag

	# Amplitude error
	amplitude_error_envelope = 1 + beta * np.sqrt(np.abs(freqs - f0))

	s_param_mock = s_param_warped * amplitude_error_envelope
	return s_param_mock

def pki_extend(x_array: NDArray, feature_names: list[str], pki: NDArray, mode: Literal["real", "complex"] ="real"):
	"""
	# pki_extend()
	## Extends features by adding prior knowledge input, works for real and complex values
	
	## Args:
	- x_array: 2D array of features
	- feature_names: List of feature column names
	- pki: 1D array of prior knowledge input for each sample
	- mode: if "real", two new featuresget added: Re(pki) and Im(pki), if "complex"	one gets added: Re(pki) + 1j*Im(pki)
	## Returns:
	- x_array: Updated 2D array of features with the new pki feature added
	- feature_names: Updated list of feature column names with the new pki feature name(s) added
	"""
	pki = pki.reshape(-1, 1)
	if mode == "complex":
		x_array = np.hstack((x_array, pki))
		feature_names.append("pki")
	else:
		x_array = np.hstack((x_array, pki.real)).astype(np.float32)
		x_array = np.hstack((x_array, pki.imag)).astype(np.float32)
		feature_names.append("pki_real")
		feature_names.append("pki_imag")

	return x_array, feature_names
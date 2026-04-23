import torch
from torch import nn
from pathlib import Path
from load_set import create_dataloader, load_csv_dataset
from model import DeepBERModel
from test_config import test_configuration


def main():
	torch.manual_seed(42)

	dataset_path = Path(__file__).resolve().parent.parent / "Datasets" / "delay_csv_database2.csv"
	if not dataset_path.exists():
		raise FileNotFoundError(
			f"Dataset not found at {dataset_path}. Update the path in main.py or move the file."
		)

	x_array, y_array, feature_columns = load_csv_dataset(dataset_path, target_column="BER")
	batch_size = 64
	
	dataloader = create_dataloader(x_array, y_array, ber_interval=[10**(-5.5),10**(-2.5)], 
								logBER=False, batch_size=batch_size, seed=42)
	dataloaderLog = create_dataloader(x_array, y_array, ber_interval=[10**(-5.5),10**(-2.5)], 
                                logBER=True, batch_size=batch_size, seed=42)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = DeepBERModel(
		input_size=len(feature_columns),
		hidden=[64, 128, 64],
		activation_fn=nn.ReLU(),
		logBER=True,
		batch_norm=False,
		dropout=0.0,
	).to(device)

	learning_rate = 1e-4
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	print(f"Loaded dataset from: {dataset_path}")
	print(f"Samples: {len(y_array)} | Features: {len(feature_columns)}")

	test_configuration(
		title="DeepBER Baseline - Hidden [64, 128, 64]",
		device=device,
		model=model,
		dataloader=dataloaderLog,
		learning_rate=learning_rate,
		batch_size=batch_size,
		criterion=criterion,
		optimizer=optimizer,
		epochs=30,
		early_stopping=True,
		patience=5,
		visualize=True,
	)


if __name__ == "__main__":
	main()

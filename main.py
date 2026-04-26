import torch
from torch import nn
from pathlib import Path
from load_set import create_dataloader, load_csv_dataset
from model import DeepBERModel
from test_config import test_configuration

def main():
	torch.manual_seed(42)

	# Loading CSV fil
	dataset_path = Path(__file__).resolve().parent / "delay_csv_database2.csv"
	if not dataset_path.exists():
		raise FileNotFoundError(
			f"Dataset not found at {dataset_path}. Update the path in main.py or move the file."
		)
	x_array, y_array, feature_columns = load_csv_dataset(dataset_path, target_column="BER")

	batch_size = 16
	
	# Creare dataloader
	dataloader = create_dataloader(x_array, y_array, ber_interval=[10**(-5.5),10**(-2.5)], 
								logBER=False, batch_size=batch_size, seed=42, standard_scale=True)
	dataloaderLog = create_dataloader(x_array, y_array, ber_interval=[10**(-5.5),10**(-2.5)], 
                                logBER=True, batch_size=batch_size, seed=42, standard_scale=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	model = DeepBERModel(
		input_size=len(feature_columns),
		hidden=[64, 128, 64],
		activation_fn=nn.ReLU(),
		logBER=True,
		batch_norm=True,
		dropout=0.1,
	).to(device)

	learning_rate = 1e-3
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
														factor=0.1, patience=3)

	print(f"Loaded dataset from: {dataset_path}")
	print(f"Samples: {len(y_array)} | Features: {len(feature_columns)}")

	test_configuration(
		title="DeepBER Baseline",
		device=device,
		model=model,
		dataloader=dataloaderLog,
		learning_rate=learning_rate,
		batch_size=batch_size,
		criterion=criterion,
		optimizer=optimizer,
		epochs=240,
		early_stopping=True,
		patience=5,
		visualize=True,
	)


if __name__ == "__main__":
	main()

from prediction.optuna_tuner import run_optuna

import torch
from torch import nn
from pathlib import Path
from load_set import create_dataloader, load_csv_dataset
from dataset_manipulation import extend_features, exclude_columns
from classification.classifier import xgb_classifier
from classification.test_classifier_config import test_classifier_configuration
from prediction.predictor import DeepBERPredictor
from prediction.test_predictor_config import test_predictor_configuration
import numpy as np

def main():
    torch.manual_seed(42)

    # ============================================= Initializing Dataset ============================================= #
    # Loading CSV files
    csv_names = ["delay_snr_csv_database1.csv", "delay_snr_csv_database2.csv"]
    dataset_paths = []

    for name in csv_names:
        dataset_path = Path(__file__).resolve().parent / "csv_files" / name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. Update the path in main.py or move the file."
            )
        dataset_paths.append(dataset_path)
    x_array, y_array, feature_columns = load_csv_dataset(dataset_paths, target_column="snr")

    # Extra features
    # Width to space ratio:
    x_array, feature_columns = extend_features(x_array, feature_columns, "width", "space", "/", "width_space_ratio")
    
    # Cross-sectional area:
    x_array, feature_columns = extend_features(x_array, feature_columns, "width", "metal_thickness", "*", "cross_sectional_area")
    
    # Ground width to signal width ratio:
    x_array, feature_columns = extend_features(x_array, feature_columns, "gnd_width", "width", "/", "gnd_width_width_ratio")

    # Remove columns
    x_array, feature_columns = exclude_columns(x_array, feature_columns, columns_to_exclude=["delay"])

    batch_size = 8
    gray_area_interval = [5.71, 10.08]
    
    # Create dataloaders
    classifier_dataloader = create_dataloader(x_array, y_array, logBER=False, batch_size=batch_size, seed=42, standard_scale=True)
    
    predictor_dataloader = create_dataloader(x_array, y_array, ber_interval=gray_area_interval, 
                                    logBER=False, batch_size=batch_size, seed=42, standard_scale=True)
    

    # Load previous dataset for Optuna tuning
    previous_dataset_path = Path(__file__).resolve().parent / "csv_files" / "delay_csv_database2.csv"
    x_array_prev, y_array_prev, feature_columns_prev = load_csv_dataset([previous_dataset_path], target_column="BER")

    gray_area_interval_prev = [10**-5.5, 10**-2.5]

    eps = 1e-15 # To avoid log(0) 
    # y_array_prev = np.log10(np.clip(y_array_prev, eps, None)).astype(np.float32)

    # Extra features
    # Width to space ratio:
    x_array_prev, feature_columns_prev = extend_features(x_array_prev, feature_columns_prev, "width", "space", "/", "width_space_ratio")
    
    # Cross-sectional area:
    x_array_prev, feature_columns_prev = extend_features(x_array_prev, feature_columns_prev, "width", "metal_thickness", "*", "cross_sectional_area")
    
    # Ground width to signal width ratio:
    x_array_prev, feature_columns_prev = extend_features(x_array_prev, feature_columns_prev, "gnd_width", "width", "/", "gnd_width_width_ratio")

    # Remove columns
    x_array_prev, feature_columns_prev = exclude_columns(x_array_prev, feature_columns_prev, columns_to_exclude=["delay"])

    # Build previous-dataset dataloader after feature engineering so dimensions match model input.
    predictor_dataloader_prev = create_dataloader(
        x_array_prev,
        y_array_prev,
        logBER=True,
        batch_size=batch_size,
        seed=42,
        ber_interval=gray_area_interval_prev,
        standard_scale=True,
    )

    
    
    
    # =================================================== Classifier ================================================== #
    classifier = xgb_classifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.01,
        gamma=1.0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        seed=42,
        eval_metric="mlogloss"
    )
    lower_thres = gray_area_interval[0]
    upper_thres = gray_area_interval[1]
    """
    test_classifier_configuration(
        title="XGBoost Baseline",
        model=classifier,
        dataloader=classifier_dataloader,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
        confusion_matrix=True
    )
    """

    # =================================================== Predictor =================================================== #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DeepBERPredictor(
        input_size=len(feature_columns_prev),
        hidden=[128, 16, 96, 48, 256],
        activation_fn=nn.ELU(),
        logBER=True,
        batch_norm=False,
        dropout=0.2514,
    ).to(device)

    learning_rate = 0.0031
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=41)

    print(f"Loaded dataset from: {dataset_path}")
    print(f"Samples: {len(y_array)} | Features: {len(feature_columns)}")
    #"""
    test_predictor_configuration(
        title="DeepBER Baseline",
        device=device,
        model=model,
        dataloader=predictor_dataloader,
        learning_rate=learning_rate,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
        epochs=240,
        early_stopping=True,
        patience=10,
        training_curves=True,
        predicted_vs_actual=True,
        # error_distribution=True,
        # error_vs_feature=feature_columns,
        # feature_columns=feature_columns
    )
    #"""

    # run_optuna(x_array_prev, y_array_prev, feature_columns_prev, gray_area_interval=gray_area_interval_prev, n_trials=800, n_epochs=240, seed=42, study_name="deepber_optuna", cv_folds=3)
    run_optuna(x_array, y_array, feature_columns, gray_area_interval=gray_area_interval, n_trials=800, n_epochs=240, seed=42, study_name="deepber_optuna", cv_folds=3)


if __name__ == "__main__":
    main()
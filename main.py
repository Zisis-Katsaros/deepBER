import torch
from torch import nn
from pathlib import Path
from load_set import create_dataloader, load_csv_dataset, extend_features
from classification.classifier import xgb_classifier
from classification.test_classifier_config import test_classifier_configuration
from prediction.predictor import DeepBERPredictor
from prediction.test_predictor_config import test_predictor_configuration
import numpy as np

def main():
    torch.manual_seed(42)

    # ============================================= Initializing Dataset ============================================= #
    # Loading CSV file
    dataset_path = Path(__file__).resolve().parent / "delay_csv_database2.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Update the path in main.py or move the file."
        )
    x_array, y_array, feature_columns = load_csv_dataset(dataset_path, target_column="BER", exclude_columns=["delay"])

    # Extra features
    # Width to space ratio:
    x_array, feature_columns = extend_features(x_array, feature_columns, "width", "space", "/", "width_space_ratio")
    
    # Cross-sectional area:
    x_array, feature_columns = extend_features(x_array, feature_columns, "width", "metal_thickness", "*", "cross_sectional_area")
    
    # Ground width to signal width ratio:
    x_array, feature_columns = extend_features(x_array, feature_columns, "gnd_width", "width", "/", "gnd_width_width_ratio")
    
    # Trace aspect ratio:
    # x_array, feature_columns = extend_features(x_array, feature_columns, "metal_thickness", "width", "/", "aspect_ratio")

    batch_size = 16
    gray_area_interval = [10**(-5.5), 10**(-2.5)] # BER range where classification is most difficult, used for focused training and evaluation
    
    # Create dataloaders
    classifier_dataloader = create_dataloader(x_array, y_array, logBER=True, batch_size=batch_size, seed=42, standard_scale=True)
    
    predictor_dataloader = create_dataloader(x_array, y_array, ber_interval=gray_area_interval, 
                                    logBER=True, batch_size=batch_size, seed=42, standard_scale=True)
    
    
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
    lower_thres = np.log10(gray_area_interval[0])
    upper_thres = np.log10(gray_area_interval[1])

    test_classifier_configuration(
        title="XGBoost Baseline",
        model=classifier,
        dataloader=classifier_dataloader,
        lower_thres=lower_thres,
        upper_thres=upper_thres,
        confusion_matrix=True
    )


    # =================================================== Predictor =================================================== #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """
    model = DeepBERPredictor(
        input_size=len(feature_columns),
        hidden=[64, 128, 64],
        activation_fn=nn.ReLU(),
        logBER=True,
        batch_norm=True,
        dropout=0.2,
    ).to(device)

    learning_rate = 1e-3
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print(f"Loaded dataset from: {dataset_path}")
    print(f"Samples: {len(y_array)} | Features: {len(feature_columns)}")

    test_predictor_configuration(
        title="DeepBER Baseline",
        device=device,
        model=model,
        dataloader=predictor_dataloader,
        learning_rate=learning_rate,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        epochs=240,
        early_stopping=True,
        patience=5,
        training_curves=True,
        predicted_vs_actual=True,
        error_distribution=True,
        error_vs_feature=feature_columns,
        feature_columns=feature_columns
    )
    """


if __name__ == "__main__":
    main()

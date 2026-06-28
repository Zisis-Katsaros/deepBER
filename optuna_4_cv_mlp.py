import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prediction.param_pred_optuna import run_optuna
from load_set import create_param_dataloader
from prediction.predictor import DeepBER_Param_Predictor_Complex
from rmse import RMSELoss
from prediction.test_predictor_config import test_predictor_configuration
from complexNN import nn as cvnn


# ============================================= Initializing Dataset ============================================= #
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

pred_arrays_dict = torch.load("csv_files/s_params/pt/pred_arrays_dict.pt", weights_only=False)


x_array = pred_arrays_dict["param_prediction_test"][0]
s_dict = pred_arrays_dict["param_prediction_test"][1]
feature_columns = pred_arrays_dict["param_prediction_test"][6]

"""
hidden_map = {
    "funnel_small": [96, 64, 48, 32],
    "funnel_large": [128, 96, 64, 48, 32],
    "pyramid_small": [48, 64, 64, 48],
    "pyramid_large": [48, 64, 96, 64, 48]
}


storage_url = "sqlite:///cv_mlp_optuna_v2.db"
run_optuna("cv_mlp", x_array, s_dict, feature_columns, batch_size=128, hidden_map=hidden_map, n_trials=50, n_epochs=25, seed=42, 
            study_name="cv_mlp_optuna_v2", storage=storage_url)
"""

elements = ["S1616", "S55", "S78", "S39", "S217"]
for element in elements:
    y_array = s_dict[element]
    out_size = y_array.shape[1] if y_array.ndim > 1 else 1

    dataloader, y_scale_params = create_param_dataloader(
                    x_array,
                    y_array,
                    batch_size=128,
                    seed=42,
                    standard_scale=True,
                    split_method="lhs"
                    )
    
    predictor = DeepBER_Param_Predictor_Complex(
        input_size=len(feature_columns) - 1, 
        hidden=[128, 96, 64, 48, 32], 
        activation_fn=cvnn.cGelu(), 
        dropout=0.0
        ).to(device)
    
    criterion = RMSELoss()
    learning_rate = 9.906773918756151e-05
    weight_decay = 0.0007214881131497281
    optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    test_predictor_configuration(
    title=f"{element}",
    device=device,
    model=predictor,
    dataloader=dataloader,
    learning_rate=learning_rate,
    batch_size=128,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=300,
    early_stopping=True,
    patience=15,
    y_scale_params=y_scale_params,
    training_curves=True,
    predicted_vs_actual=True,
    test_out_dir = f"out_files/cv_mlp/{element}"
)
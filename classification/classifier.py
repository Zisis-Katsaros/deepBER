from xgboost import XGBClassifier
from torch import nn

def xgb_classifier(n_estimators=1000, max_depth=3, learning_rate=0.01, gamma=1.5, subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0, seed=42, eval_metric="mlogloss"):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=seed,
        eval_metric=eval_metric,
    )

class DeepBERClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden, activation_fn=nn.ReLU(), 
                 logBER=False, batch_norm=False, dropout=0.0):
        # Modular MLP architecture 
        #
        # Args:
        # - input_size: Number of input features
        # - num_classes: Number of output classes
        # - hidden: List of hidden layer sizes
        # - activation_fn: Activation function for each of the hidden layers
        # - logBER: If true model predicts log10(BER)
        # - batch_norm: If true batch normalization is applied after each hidden layer
        # - dropout: Dropout rate
        
        super(DeepBERClassifier, self).__init__()

        self.layers = nn.ModuleList() # create a list to hold the layers
        current_dim = input_size # initial input dimension

        for layer in range(len(hidden)):
            # Linear Layer
            self.layers.append(nn.Linear(current_dim, hidden[layer]))

            # Batch Normalization (optional)
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden[layer]))

            # Activation Function
            self.layers.append(activation_fn)

            # Dropout (optional)
            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))
            
            current_dim = hidden[layer] # update current dimension for next layer

        # Final output layer
        self.output_layer = nn.Linear(current_dim, num_classes)

    # Forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) # pass through each layer
        output = self.output_layer(x) # final output layer
        return output
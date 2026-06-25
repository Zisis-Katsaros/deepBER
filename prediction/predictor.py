import torch
from torch import nn
from complexNN import nn as cvnn

class DeepBERPredictor(nn.Module):
    def __init__(self, input_size, hidden, activation_fn=nn.ReLU(), 
                 logBER=False, batch_norm=False, dropout=0.0):
        # Modular MLP architecture 
        #
        # Args:
        # - input_size: Number of input features
        # - hidden: List of hidden layer sizes
        # - activation_fn: Activation function for each of the hidden layers
        # - logBER: If true model predicts log10(BER)
        # - batch_norm: If true batch normalization is applied after each hidden layer
        # - dropout: Dropout rate
        
        super(DeepBERPredictor, self).__init__()

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
        if logBER:
            self.output_layer = nn.Linear(current_dim, 1) # linear layer with single output for log10(BER)
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(current_dim, 1),
                nn.Sigmoid()
            ) # sigmoid activation for BER so that output is between 0 and 1

    # Forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) # pass through each layer
        output = self.output_layer(x) # final output layer
        return output
    

class DeepBER_Param_Predictor(nn.Module):
    def __init__(self, input_size, hidden, output_size, activation_fn=nn.ReLU(), batch_norm=False, dropout=0.0):
        # Modular MLP architecture for S-/ ABCD-Parameter prediction
        #
        # Args:
        # - input_size: Number of input features
        # - hidden: List of hidden layer sizes
        # - activation_fn: Activation function for each of the hidden layers
        # - batch_norm: If true batch normalization is applied after each hidden layer
        # - dropout: Dropout rate
        
        super(DeepBER_Param_Predictor, self).__init__()

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
        self.output_layer = nn.Linear(current_dim, output_size)
        
    # Forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) # pass through each layer
        output = self.output_layer(x) # final output layer
        return output
    

class DeepBER_Param_Predictor_Complex(nn.Module):
    def __init__(self, input_size, hidden, activation_fn=nn.ReLU(), batch_norm=False, dropout=0.0):
        # Modular Complex Valued MLP architecture for S-/ ABCD-Parameter prediction
        #
        # Args:
        # - input_size: Number of input features
        # - hidden: List of hidden layer sizes
        # - activation_fn: Activation function for each of the hidden layers
        # - batch_norm: If true batch normalization is applied after each hidden layer
        # - dropout: Dropout rate
        
        super(DeepBER_Param_Predictor_Complex, self).__init__()

        self.layers = nn.ModuleList() # create a list to hold the layers
        current_dim = input_size # initial input dimension
     
        for layer in range(len(hidden)):
            # Linear Layer
            self.layers.append(cvnn.cLinear(current_dim, hidden[layer]))

            # Batch Normalization (optional)
            if batch_norm:
                self.layers.append(cvnn.cBatchNorm1d(hidden[layer]))

            # Activation Function
            self.layers.append(activation_fn)

            # Dropout (optional)
            if dropout > 0.0:
                self.layers.append(cvnn.cDropout(dropout))
            
            current_dim = hidden[layer] # update current dimension for next layer

        # Final output layer 
        self.output_layer = cvnn.cLinear(current_dim, 1)        
    # Forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) # pass through each layer
        output = self.output_layer(x) # final output layer
        return output

 

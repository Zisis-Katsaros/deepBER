import torch
from torch import nn
from complexNN import nn as cvnn
from prediction.custom_layers import GaussianSmoothingLayer, CausalityEnforcementLayer, PassivityEnforcementLayer

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


class PI_STCNN(nn.Module):
    def __init__(self, input_size, mlp_hidden, mlp_activation_fn, mlp_dropout, tcnn_layer_params, tcnn_activation_fn, output_size, num_ports, N, M, K, varience_min=1.0, passivity_margin=1.03):
        """
        # Physics-Informed Transposed Convolutional Neural Network modular architecture for S-Parameter prediction

        ## Args:
        - input_size: Number of input features
        - mlp_hidden: List of hidden layer sizes for the base-model 
        - mlp_activation_fn: Activation function for each of the hidden layers in the base-model
        - mlp_dropout: Dropout rate for the base-model
        - tcnn_params: List of layer parameters for the 1D Transposed Conv Layers [out_channels, kernel_size, stride]
        - tcnn_activation_fn: Activation function for each of the transposed convolutional layers
        - output_size: Number of unique S-parameter elements
        - num_ports: Number of ports
        - N: Base number of frequency points in target data
        - M: Extrapolation factor
        - K: Interpolation/Truncation factor
        - passivity_margin: Margin for passivity enforcement (>1)
        """
        super(PI_STCNN, self).__init__()

        self.num_ports = num_ports
        self.Dy = output_size
        self.extrapolated_pts = int(N*M + 1)

        # Base-model
        self.mlp = nn.ModuleList()
        current_dim = input_size
        for hidden in mlp_hidden:
            self.mlp.append(nn.Linear(current_dim, hidden))
            self.mlp.append(mlp_activation_fn)
            if mlp_dropout > 0.0:
                self.mlp.append(nn.Dropout(mlp_dropout))
            current_dim = hidden

        # Linear layer mapping to the initial shape for 1D Transposed Convs
        self.initial_seq_len = 10 
        self.mlp_to_tcnn = nn.Linear(current_dim, tcnn_layer_params[0][0] * self.initial_seq_len)

        self.tcnn = nn.ModuleList()
        in_channels = tcnn_layer_params[0][0]

        for out_channels, kernel_size, stride in tcnn_layer_params[1:]:
            self.tcnn.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1))
            self.tcnn.append(tcnn_activation_fn)
            in_channels = out_channels

        # Final transposed conv mapping to output channels
        self.tcnn_final = nn.ConvTranspose1d(in_channels, self.Dy, kernel_size=tcnn_layer_params[-1][1], stride=tcnn_layer_params[-1][2], padding=1)

        # Adaptive pooling or interpolation to enforce exact N*M + 1 length dimension
        self.length_adjust = nn.AdaptiveAvgPool1d(self.extrapolated_pts)

        # CoordConv Layer: Maps (Dy channels + 1 coordinate channel) back down to Dy channels
        self.coord_layer = nn.Conv1d(self.Dy + 1, self.Dy, kernel_size=1)

        # Causality and Passivity enforcement layers
        self.smoothing_layer = GaussianSmoothingLayer(channels=self.Dy, varience_min=varience_min)
        self.cel = CausalityEnforcementLayer(N=N, M=M, K=K)
        self.pel = PassivityEnforcementLayer(num_ports=num_ports, passivity_margin=passivity_margin)
    
    def forward(self, x, bypass_pel=False, match_training_grid=True):
        # Forward pass through the base-model (MLP)
        for layer in self.mlp:
            x = layer(x)
        x = self.mlp_to_tcnn(x)

        # Unflatten to (batch_size, channels, seq_len)
        x = x.view(x.size(0), -1, self.initial_seq_len)

        # Forwar pass through the transposed convolutional layers
        for layer in self.tcnn:
            x = layer(x)
        x = self.tcnn_final(x)

        # Enforce exact length for extrapolated Re{S}
        y_bn = self.length_adjust(x)

        # CoordConv logic
        batch_size, _, current_k = y_bn.size()

        coords = torch.linspace(-1, 1, steps=current_k, device=y_bn.device) # -1 to 1 due to standard scaling
        coords = coords.view(1, 1, current_k).expand(batch_size, 1, current_k)

        y_bn_with_coords = torch.cat([y_bn, coords], dim=1)
        y_cc = self.coord_layer(y_bn_with_coords)

        # Apply smoothing layer
        y_sl = self.smoothing_layer(y_cc)

        # Causality enforcement
        S_cel_real, S_cel_imag = self.cel(y_sl)

        if bypass_pel:
            out = torch.cat([S_cel_real, S_cel_imag], dim=1)
        else:
            # Passivity enforcement
            S_final_real, S_final_imag = self.pel(S_cel_real, S_cel_imag)
            out = torch.cat([S_final_real, S_final_imag], dim=1)

        if match_training_grid:
            # Sub-sample to match the training grid
            out = out[:, :, ::self.cel.K][:, :, :self.cel.N]
        return out

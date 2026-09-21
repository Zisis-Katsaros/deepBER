function VTF_matrix = calculate_vtf_from_sparameters(S_data, R_tx, C_tx, R_rx, C_rx, tx_ports, rx_ports)
    %{
    Calculates the N-port Voltage Transfer Function (VTF) matrix from an S-parameter object.
    It embeds the TX and RX resistance and capacitance into the network.
    
    Inputs:
    - S_data: sparameters object
    - R_tx, C_tx: Transmitter termination resistance (Ohms) and capacitance (Farads)
    - R_rx, C_rx: Receiver termination resistance (Ohms) and capacitance (Farads)
    - tx_ports: (Optional) Array of TX port indices. Defaults to 1:(N/2).
    - rx_ports: (Optional) Array of RX port indices. Defaults to (N/2 + 1):N.
    
    Outputs:
    - VTF_matrix: N x N x Freq array representing the voltage transfer function
                  from an ideal Thevenin voltage source at port j to node i.
    %}
    arguments
        S_data (1,1) sparameters
        R_tx (1,1) double {mustBePositive}
        C_tx (1,1) double {mustBeNonnegative}
        R_rx (1,1) double {mustBePositive}
        C_rx (1,1) double {mustBeNonnegative}
        tx_ports (1,:) double {mustBeInteger, mustBePositive} = []
        rx_ports (1,:) double {mustBeInteger, mustBePositive} = []
    end
    freq = S_data.Frequencies;
    num_freq = length(freq);
    num_ports = size(S_data.Parameters, 1);
    
    % Default to standard N-port layout
    if isempty(tx_ports)
        tx_ports = 1:(num_ports/2);
    end
    if isempty(rx_ports)
        rx_ports = (num_ports/2 + 1):num_ports;
    end
    
    % Convert S-parameters to Y-parameters
    Y_matrix = s2y(S_data.Parameters, S_data.Impedance);
    
    VTF_matrix = zeros(num_ports, num_ports, num_freq);
    
    fprintf('Applying %d-Ohm/%.2f-fF TX and %d-Ohm/%.2f-fF RX terminations...\n', ...
            R_tx, C_tx*1e15, R_rx, C_rx*1e15);
            
    for k = 1:num_freq
        omega = 2 * pi * freq(k);
        
        % Calculate admittances of the TX and RX terminations
        Y_tx = (1 / R_tx) + 1i * omega * C_tx;
        Y_rx = (1 / R_rx) + 1i * omega * C_rx;
        
        % Create a diagonal matrix of the termination admittances
        Y_term = zeros(num_ports, num_ports);
        for p = tx_ports
            Y_term(p, p) = Y_tx;
        end
        for p = rx_ports
            Y_term(p, p) = Y_rx;
        end
        
        % Augment the original Y-matrix with the physical terminations
        Y_aug = Y_matrix(:,:,k) + Y_term;
        
        % Convert back to an Impedance matrix (Z = Y^-1)
        Z_aug = inv(Y_aug);
        
        % The Voltage Transfer Function (VTF) from an ideal source V_s at a TX port
        % to any node in the network is Z_aug / R_tx.
        % (Because injected Norton current I = V_s / R_tx, and V_node = Z_aug * I)
        VTF_matrix(:,:,k) = Z_aug / R_tx;
    end
end
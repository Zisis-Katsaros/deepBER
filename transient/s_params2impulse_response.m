function [fit_main, fit_next1, fit_fext1, fit_next2, fit_fext2] = s_params2impulse_response(filename, tx, rx, next1, fext1, next2, fext2, precision, R_tx, C_tx, R_rx, C_rx)
    %{
    Converts S-parameters to a Voltage Transfer Function (VTF) with specific 
    R/C terminations, then performs Vector Fitting to return the fitted models.
    
    Inputs:
    - filename: Path to the Touchstone file containing S-parameters.
    - tx, rx: Transmit and Receive port numbers 
    - next*, fext*: Near-end and Far-end crosstalk port numbers
    - precision: (Optional) Precision for rational fitting in dB (default: -40 dB)
    - R_tx, C_tx: (Optional) TX termination resistance and capacitance
    - R_rx, C_rx: (Optional) RX termination resistance and capacitance
    %}
    arguments 
        filename (1,1) string
        tx (1,1) double {mustBeInteger, mustBePositive}
        rx (1,1) double {mustBeInteger, mustBePositive}
        next1 (1,1) double {mustBeInteger, mustBePositive}
        fext1 (1,1) double {mustBeInteger, mustBePositive}
        next2 (1,1) double {mustBeInteger, mustBePositive}
        fext2 (1,1) double {mustBeInteger, mustBePositive}
        precision (1,1) double = -40; % Default precision is -40 dB
        % Default UCIe 32 GT/s Standard Package values:
        R_tx (1,1) double {mustBePositive} = 30; 
        C_tx (1,1) double {mustBeNonnegative} = 125e-15; 
        R_rx (1,1) double {mustBePositive} = 50; 
        C_rx (1,1) double {mustBeNonnegative} = 125e-15; 
    end 
    
    % Load S-parameter matrix (Touchstone)
    S_data = sparameters(filename);
    freq = S_data.Frequencies;

    % Convert raw S-parameters into a properly terminated Voltage Transfer Function (VTF)
    VTF_matrix = calculate_vtf_from_sparameters(S_data, R_tx, C_tx, R_rx, C_rx);

    % Extract specific frequency-domain VTF arrays for the requested paths
    VTF_main = squeeze(VTF_matrix(rx, tx, :));
    VTF_next1 = squeeze(VTF_matrix(next1, tx, :));
    VTF_fext1 = squeeze(VTF_matrix(fext1, tx, :));
    VTF_next2 = squeeze(VTF_matrix(next2, tx, :));
    VTF_fext2 = squeeze(VTF_matrix(fext2, tx, :));

    % Convert VTFs to time-domain impulse responses (Vector Fitting)
    fprintf('Fitting rational models. This may take a moment...\n');
    fit_main = rationalfit(freq, VTF_main, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_next1 = rationalfit(freq, VTF_next1, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_fext1 = rationalfit(freq, VTF_fext1, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_next2 = rationalfit(freq, VTF_next2, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_fext2 = rationalfit(freq, VTF_fext2, 'Tolerance', precision, 'NPoles', [0, 60]);
    fprintf('Models fitted successfully.\n');
end
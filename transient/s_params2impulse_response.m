function [fit_main, fit_next, fit_fext] = s_params2impulse_response(filename, tx, rx, next, fext, precision)
    %{
    Performs Vector fitting for given S-parameters and returns the fitted models for main, next, and fext channels.
    Inputs:
    - filename: Path to the Touchstone file containing S-parameters.
    - tx: Transmit port number (integer)
    - rx: Receive port number (integer)
    - next: Next port number (integer)
    - fext: Far-end crosstalk port number (integer)
    - precision: (Optional) Precision for rational fitting in dB (default: -40 dB)
    Outputs:
    - fit_main: Fitted model for the main channel
    - fit_next: Fitted model for the next channel
    - fit_fext: Fitted model for the far-end crosstalk channel
    %}
    arguments 
        filename (1,1) string
        tx (1,1) double {mustBeInteger, mustBePositive}
        rx (1,1) double {mustBeInteger, mustBePositive}
        next (1,1) double {mustBeInteger, mustBePositive}
        fext (1,1) double {mustBeInteger, mustBePositive}
        precision = -40; % Default precision is -40 dB
    end 
    % Load S-parameter matrix (Touchstone)
    S_data = sparameters(filename);
    freq = S_data.Frequencies;

    % Extract specific frequency-domain arrays
    S_main = squeeze(S_data.Parameters(rx, tx, :));
    S_next = squeeze(S_data.Parameters(next, tx, :));
    S_fext = squeeze(S_data.Parameters(fext, tx, :));

    % Convert S-parameters to time-domain impulse responses (Vector Fitting)
    fprintf('Fitting rational models. This may take a moment...\n');
    fit_main = rationalfit(freq, S_main, 'Tolerance', precision);
    fit_next = rationalfit(freq, S_next, 'Tolerance', precision);
    fit_fext = rationalfit(freq, S_fext, 'Tolerance', precision);
    fprintf('Models fitted successfully.\n');
end
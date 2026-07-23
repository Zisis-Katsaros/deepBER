function [fit_main, fit_next1, fit_fext1, fit_next2, fit_fext2] = s_params2impulse_response(filename, tx, rx, next1, fext1, next2, fext2, precision)
    %{
    Performs Vector fitting for given S-parameters and returns the fitted models for main, next, and fext channels.
    Inputs:
    - filename: Path to the Touchstone file containing S-parameters.
    - tx: Transmit port number 
    - rx: Receive port number 
    - next*: Next port numbers (above and bellow tx port)
    - fext: Far-end crosstalk port numbers (above and bellow rx port)
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
        next1 (1,1) double {mustBeInteger, mustBePositive}
        fext1 (1,1) double {mustBeInteger, mustBePositive}
        next2 (1,1) double {mustBeInteger, mustBePositive}
        fext2 (1,1) double {mustBeInteger, mustBePositive}
        precision = -40; % Default precision is -40 dB
    end 
    % Load S-parameter matrix (Touchstone)
    S_data = sparameters(filename);
    freq = S_data.Frequencies;

    % Extract specific frequency-domain arrays
    S_main = squeeze(S_data.Parameters(rx, tx, :));
    S_next1 = squeeze(S_data.Parameters(next1, tx, :));
    S_fext1 = squeeze(S_data.Parameters(fext1, tx, :));
    S_next2 = squeeze(S_data.Parameters(next2, tx, :));
    S_fext2 = squeeze(S_data.Parameters(fext2, tx, :));

    % Convert S-parameters to time-domain impulse responses (Vector Fitting)
    fprintf('Fitting rational models. This may take a moment...\n');
    fit_main = rationalfit(freq, S_main, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_next1 = rationalfit(freq, S_next1, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_fext1 = rationalfit(freq, S_fext1, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_next2 = rationalfit(freq, S_next2, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_fext2 = rationalfit(freq, S_fext2, 'Tolerance', precision, 'NPoles', [0, 60]);
    fprintf('Models fitted successfully.\n');
end
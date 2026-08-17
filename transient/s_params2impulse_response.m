function [fit_main, fit_next1, fit_fext1, fit_next2, fit_fext2] = s_params2impulse_response(filename, tx, rx, next1, fext1, next2, fext2, precision, R_tx, C_tx, R_rx, C_rx)
    arguments 
        filename (1,1) string
        tx (1,1) double {mustBeInteger, mustBePositive}
        rx (1,1) double {mustBeInteger, mustBePositive}
        next1 (1,1) double {mustBeInteger, mustBePositive}
        fext1 (1,1) double {mustBeInteger, mustBePositive}
        next2 (1,1) double {mustBeInteger, mustBePositive}
        fext2 (1,1) double {mustBeInteger, mustBePositive}
        precision (1,1) double = -40;
        R_tx (1,1) double {mustBePositive} = 30; 
        C_tx (1,1) double {mustBeNonnegative} = 125e-15; 
        R_rx (1,1) double {mustBePositive} = 50; 
        C_rx (1,1) double {mustBeNonnegative} = 125e-15; 
    end 
    
    S_data = sparameters(filename);
    freq = S_data.Frequencies;
    VTF_matrix = calculate_vtf_from_sparameters(S_data, R_tx, C_tx, R_rx, C_rx);

    VTF_main = squeeze(VTF_matrix(rx, tx, :));
    VTF_next1 = squeeze(VTF_matrix(next1, tx, :));
    VTF_fext1 = squeeze(VTF_matrix(fext1, tx, :));
    VTF_next2 = squeeze(VTF_matrix(next2, tx, :));
    VTF_fext2 = squeeze(VTF_matrix(fext2, tx, :));

    fprintf('Fitting rational models. This may take a moment...\n');
    fit_main = rationalfit(freq, VTF_main, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_next1 = rationalfit(freq, VTF_next1, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_fext1 = rationalfit(freq, VTF_fext1, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_next2 = rationalfit(freq, VTF_next2, 'Tolerance', precision, 'NPoles', [0, 60]);
    fit_fext2 = rationalfit(freq, VTF_fext2, 'Tolerance', precision, 'NPoles', [0, 60]);
    fprintf('Models fitted successfully.\n');
end
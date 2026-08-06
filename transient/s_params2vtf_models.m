function [fit_main, fit_next1, fit_fext1, fit_next2, fit_fext2] = s_params2vtf_models(filename, tx, rx, next1, fext1, next2, fext2)
    
    % UCIe 32 GT/s Standard Defaults
    R_tx = 30; C_tx = 125e-15; 
    R_rx = 50; C_rx = 125e-15; 
    precision = -40;

    S_data = sparameters(filename);
    
    % Convert raw S-params to VTF using Nodal Analysis
    VTF_matrix = calculate_vtf_from_sparameters(S_data, R_tx, C_tx, R_rx, C_rx);
    freq = S_data.Frequencies;

    % Safely extract paths and Vector Fit
    fit_main = rationalfit(freq, squeeze(VTF_matrix(rx, tx, :)), 'Tolerance', precision);
    
    if ~isempty(next1)
        fit_next1 = rationalfit(freq, squeeze(VTF_matrix(next1, tx, :)), 'Tolerance', precision);
        fit_fext1 = rationalfit(freq, squeeze(VTF_matrix(fext1, tx, :)), 'Tolerance', precision);
    else
        fit_next1 = []; fit_fext1 = [];
    end
    
    if ~isempty(next2)
        fit_next2 = rationalfit(freq, squeeze(VTF_matrix(next2, tx, :)), 'Tolerance', precision);
        fit_fext2 = rationalfit(freq, squeeze(VTF_matrix(fext2, tx, :)), 'Tolerance', precision);
    else
        fit_next2 = []; fit_fext2 = [];
    end
end
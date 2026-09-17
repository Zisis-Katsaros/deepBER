function [fit_main, fit_next1, fit_fext1, fit_next2, fit_fext2] = s_params2vtf_models(filename, tx, rx, next1, fext1, next2, fext2, bit_rate)
    
    % Configure termination and capacitance values for Tx and Rx as specified in the UCIe protocol for the standard package at the given bit rate
    if bit_rate <= 12e9
        R_tx = 30;
        R_rx = 1e6; % unterminated
        C_tx = 125e-15;
        C_rx = 125e-15;
    elseif bit_rate <= 16e9
         R_tx = 30;
        R_rx = 50; % reach dependent, simplified to a fixed 50 Ohm value
        C_tx = 125e-15;
        C_rx = 125e-15;
    else 
        R_tx = 30;
        R_rx = 50;
        C_tx = 125e-15;
        C_rx = 125e-15;
    end
     
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
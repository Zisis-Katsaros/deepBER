function [t_vec, z_pred, z_act] = run_tdr_evaluation(filename_preds, filename_actuals, geometry_title, show_plots)
    % Runs TDR Evaluation for a given set of S-parameters.
    
    tx_port = 1; % Port to launch step excitation
    Z0 = 50;     % Reference impedance (Ohms)
    
    % Load S-parameters
    S_pred = sparameters(filename_preds);
    S_act = sparameters(filename_actuals);
    
    % Extract Reflection coefficient (S11)
    S11_pred = squeeze(S_pred.Parameters(tx_port, tx_port, :));
    S11_act = squeeze(S_act.Parameters(tx_port, tx_port, :));
    
    % Fit rational models for time domain conversion
    fit_pred = rationalfit(S_pred.Frequencies, S11_pred, 'Tolerance', -40);
    fit_act = rationalfit(S_act.Frequencies, S11_act, 'Tolerance', -40);
    
    % Define time parameters
    num_samples = 1000;
    t_max = 200e-12; % 200 ps
    ts = t_max / (num_samples - 1); % Calculate time step
    
    % Calculate step response
    [rho_pred, t_vec] = stepresp(fit_pred, ts, num_samples);
    [rho_act, ~] = stepresp(fit_act, ts, num_samples);
    
    % Convert reflection coefficient to impedance
    z_pred = Z0 .* (1 + rho_pred) ./ (1 - rho_pred);
    z_act = Z0 .* (1 + rho_act) ./ (1 - rho_act);
    
    if show_plots
        plot_tdr(t_vec, z_pred, z_act, geometry_title);
    end
end
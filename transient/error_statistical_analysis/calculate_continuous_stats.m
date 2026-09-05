function continuous_stats = calculate_continuous_stats(pred, act)
    raw_error = pred - act;
    abs_error = abs(raw_error);

    % Aggregate Metrics
    rmse_val = sqrt(mean(raw_error.^2));
    mae_val = mean(abs_error);
    span_act = max(act) - min(act);
    nrmse_val = rmse_val / span_act; % Normalized RMSE

    % Correlation Coefficient
    R = corrcoef(act, pred);
    R_squared = R(1, 2)^2;

    % Descriptive Stats for absolute error
    err_median = median(abs_error);
    err_q25 = quantile(abs_error, 0.25);
    err_q75 = quantile(abs_error, 0.75);
    err_IQR = err_q75 - err_q25;
    err_max = max(abs_error);
    err_var = var(abs_error);
    err_std = std(abs_error);
    coeff_var = err_std / mean(abs_error);

    continuous_stats = struct('RMSE', rmse_val, 'MAE', mae_val, 'NRMSE', nrmse_val, ...
        'R_squared', R_squared, 'Median_Error', err_median, 'Q25_Error', err_q25, ...
        'Q75_Error', err_q75, 'IQR_Error', err_IQR, 'Max_Error', err_max, ...
        'Variance_Error', err_var, 'StdDev_Error', err_std, 'CoeffVar_Error', coeff_var);
end
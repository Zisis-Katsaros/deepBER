function [rmse_rt, rmse_ft, rmse_eye_height, rmse_eye_jitter, rmse_eye_amp, rmse_eye_width, mape_eye_height, mape_eye_width, ...
          eye_height_pred, eye_height_act, eye_width_pred, eye_width_act] = eye_metrics_pred_vs_act(Vout_pred, Vout_act, eye_matrix_pred, ...
    eye_matrix_act, fs, bit_rate)
    
    rt_act = mean(risetime(Vout_act, fs));
    rt_pred = mean(risetime(Vout_pred, fs));
    
    ft_act = mean(falltime(Vout_act, fs));
    ft_pred = mean(falltime(Vout_pred, fs));

    % The raw variables are generated right here:
    [eye_height_pred, eye_jitter_pred, eye_amp_pred, eye_width_pred] = calculate_eye_stats(eye_matrix_pred, fs, bit_rate);
    [eye_height_act, eye_jitter_act, eye_amp_act, eye_width_act] = calculate_eye_stats(eye_matrix_act, fs, bit_rate);

    rmse_rt = rmse(rt_pred, rt_act);
    rmse_ft = rmse(ft_pred, ft_act);
    rmse_eye_height = rmse(eye_height_pred, eye_height_act);
    rmse_eye_width = rmse(eye_width_pred, eye_width_act);
    rmse_eye_jitter = rmse(eye_jitter_pred, eye_jitter_act);
    rmse_eye_amp = rmse(eye_amp_pred, eye_amp_act);
    
    % MAPE safeguards
    % Safe MAPE for Eye Height
    if eye_height_act <= 1e-4 % Treat anything below 0.1mV as exactly 0
        if eye_height_pred <= 1e-4
            mape_eye_height = 0; % Both closed
        else
            mape_eye_height = 100; % Act closed, Pred open (Max error)
        end
    else
        mape_eye_height = abs((eye_height_act - eye_height_pred) / eye_height_act) * 100;
    end

    % Safe MAPE for Eye Width (Ignoring NaNs)
    mape_eye_width = safe_mape(eye_width_act, eye_width_pred, 1e-15);
end
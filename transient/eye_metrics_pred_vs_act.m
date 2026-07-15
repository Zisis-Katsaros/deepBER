function [rmse_rt, rmse_ft, rmse_eye_hight, rmse_eye_jitter, rmse_eye_amp] = eye_metrics_pred_vs_act(Vout_pred, Vout_act, eye_matrix_pred, ...
    eye_matrix_act, fs, bit_rate)
    % Rise Time
    rt_act = mean(risetime(Vout_act, fs));
    rt_pred = mean(risetime(Vout_pred, fs));
    
    % Fall Time
    ft_act = mean(falltime(Vout_act, fs));
    ft_pred = mean(falltime(Vout_pred, fs));

    [eye_hight_pred, eye_jitter_pred, eye_amp_pred] = calculate_eye_stats(eye_matrix_pred, fs, bit_rate);
    [eye_hight_act, eye_jitter_act, eye_amp_act] = calculate_eye_stats(eye_matrix_act, fs, bit_rate);

    rmse_rt = rmse(rt_pred, rt_act);
    rmse_ft = rmse(ft_pred, ft_act);
    rmse_eye_hight = rmse(eye_hight_pred, eye_hight_act);
    rmse_eye_jitter = rmse(eye_jitter_pred, eye_jitter_act);
    rmse_eye_amp = rmse(eye_amp_pred, eye_amp_act);
end
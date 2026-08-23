function V_out = compute_transient(model, V_in, Ts, method, t_imp)
    % Evaluates the model using standard rational timeresp or IFFT convolution
    if isempty(model)
        V_out = [];
        return;
    end
    
    if method == "rationalfit"
        V_out = timeresp(model, V_in, Ts);
    else
        % 'model' is the discrete impulse response array
        dt_native = t_imp(2) - t_imp(1);
        
        % Resample impulse response to match simulation sample rate (Ts)
        t_resampled = (0:Ts:t_imp(end))';
        h_resampled = interp1(t_imp, model, t_resampled, 'linear', 0);
        
        % Crucial: Scale the amplitude DOWN to preserve DC gain after upsampling
        h_resampled = h_resampled * (Ts / dt_native);
        
        % Compute transient via discrete convolution
        V_out = conv(V_in, h_resampled);
        
        % Truncate to match original input time vector length
        V_out = V_out(1:length(V_in));
    end
end
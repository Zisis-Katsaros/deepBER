function [ADM, FDM, GDM, ADMi, FDMi, GDMi, ADMc, FDMc, GDMc] = perform_fsv(pred_file, act_file, port_i, port_j)
    % Read Touchstone files
    act_obj = sparameters(act_file);
    pred_obj = sparameters(pred_file);
    
    % Extract specific S-parameter array (complex data)
    act_data = squeeze(act_obj.Parameters(port_i, port_j, :));
    pred_data = squeeze(pred_obj.Parameters(port_i, port_j, :));
    
    % Step 1 & 2: Zero-pad to next power of 2 and apply FFT
    N_orig = length(act_data);
    N_pad = 2^nextpow2(N_orig);
    act_fft = fft([act_data; zeros(N_pad - N_orig, 1)]);
    pred_fft = fft([pred_data; zeros(N_pad - N_orig, 1)]);
    
    % Calculate break-point for low/high pass separation
    bp1 = find_breakpoint(act_fft);
    bp2 = find_breakpoint(pred_fft);
    break_point = min(bp1, bp2) + 5; 
    
    % Step 3 & 4: Apply filters and IFFT to separate Low and High components
    [Lo1, Hi1] = apply_filters(act_fft, break_point);
    [Lo2, Hi2] = apply_filters(pred_fft, break_point);
    
    % Truncate back to original dataset length
    Lo1 = Lo1(1:N_orig); Hi1 = Hi1(1:N_orig);
    Lo2 = Lo2(1:N_orig); Hi2 = Hi2(1:N_orig);
    
    % Step 5 & 6: Calculate ADMi (Point-by-point) and ADM (Mean)
    ADMi = calc_ADMi(Lo1, Lo2);
    ADM = mean(ADMi);
    
    % Step 8-10: Calculate FDMi (Point-by-point) and FDM (Mean)
    FDMi = calc_FDMi(Lo1, Lo2, Hi1, Hi2);
    FDM = mean(FDMi);
    
    % Step 12 & 13: Calculate GDMi (Point-by-point) and GDM (Mean)
    GDMi = sqrt(ADMi.^2 + FDMi.^2);
    GDM = mean(GDMi);
    
    % Step 7, 11, 13: Calculate Confidence Histograms
    ADMc = calc_confidence(ADMi);
    FDMc = calc_confidence(FDMi);
    GDMc = calc_confidence(GDMi);
end
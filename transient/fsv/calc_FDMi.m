function FDMi = calc_FDMi(Lo1, Lo2, Hi1, Hi2)
    N = length(Lo1);
    % Calculate derivatives: Lo'(f), Hi'(f), and Hi''(f)
    Lo1_p = get_derivative(Lo1, 2); Lo2_p = get_derivative(Lo2, 2);
    Hi1_p = get_derivative(Hi1, 2); Hi2_p = get_derivative(Hi2, 2);
    Hi1_pp = get_derivative(Hi1_p, 3); Hi2_pp = get_derivative(Hi2_p, 3);
    
    denom1 = (2/N) * sum(abs(Lo1_p) + abs(Lo2_p));
    FDM1 = (abs(Lo1_p) - abs(Lo2_p)) ./ (denom1 + eps);
    
    denom2 = (6/N) * sum(abs(Hi1_p) + abs(Hi2_p));
    FDM2 = (abs(Hi1_p) - abs(Hi2_p)) ./ (denom2 + eps);
    
    denom3 = (7.2/N) * sum(abs(Hi1_pp) + abs(Hi2_pp));
    FDM3 = (abs(Hi1_pp) - abs(Hi2_pp)) ./ (denom3 + eps);
    
    FDMi = 2 * abs(FDM1 + FDM2 + FDM3);
end
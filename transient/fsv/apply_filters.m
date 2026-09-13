function [Lo, Hi] = apply_filters(fft_data, break_point)
    N = length(fft_data);
    H_LP = zeros(N, 1);
    
    start_idx = max(1, break_point - 5);
    end_idx = min(floor(N/2), break_point + 5);
    
    % Linearly decreasing envelope from 5 points below to 5 points above break-point
    H_LP(1:start_idx-1) = 1;
    if end_idx >= start_idx
        H_LP(start_idx:end_idx) = linspace(1, 0, end_idx - start_idx + 1);
    end
    
    % Mirror filter for negative frequencies to maintain valid IFFT
    H_LP(N/2+2:end) = flipud(H_LP(2:N/2));
    H_HP = 1 - H_LP;
    
    Lo = ifft(fft_data .* H_LP);
    Hi = ifft(fft_data .* H_HP);
end
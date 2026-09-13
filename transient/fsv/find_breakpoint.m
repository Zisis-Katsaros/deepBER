function bp = find_breakpoint(fft_data)
    % Ignore first 4 points (DC/low freq), sum intensities from 5th point
    mag = abs(fft_data(5:floor(end/2))); 
    total_val = sum(mag);
    % Find location where sum reaches 40% of total
    idx = find(cumsum(mag) >= 0.4 * total_val, 1);
    if isempty(idx), idx = 1; end
    bp = idx + 4; 
end
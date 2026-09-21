function conf = calc_confidence(metric_array)
    % Confidence histogram categories: Excellent, Very Good, Good, Fair, Poor, Very Poor
    N = length(metric_array);
    conf = zeros(1, 6);
    conf(1) = sum(metric_array < 0.1) / N;
    conf(2) = sum(metric_array >= 0.1 & metric_array < 0.2) / N;
    conf(3) = sum(metric_array >= 0.2 & metric_array < 0.4) / N;
    conf(4) = sum(metric_array >= 0.4 & metric_array < 0.8) / N;
    conf(5) = sum(metric_array >= 0.8 & metric_array < 1.6) / N;
    conf(6) = sum(metric_array >= 1.6) / N;
end
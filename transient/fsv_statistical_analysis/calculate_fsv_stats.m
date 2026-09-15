function fsv_stats = calculate_fsv_stats(metric)
    fsv_median = median(metric);
    q25 = quantile(metric, 0.25);
    q75 = quantile(metric, 0.75);
    iqr = q75 - q25;
    fsv_max = max(metric);
    fsv_var = var(metric);
    std_dev = std(metric);
    coeff_var = std_dev / mean(metric);

    fsv_stats = struct('Median', fsv_median, 'Q25', q25, 'Q75', q75, 'IQR', iqr, ...
        'Max', fsv_max, 'Variance', fsv_var, 'StdDev', std_dev, 'CoeffVar', coeff_var);
end

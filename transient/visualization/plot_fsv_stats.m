function plot_fsv_stats(metric_data, metric_name)
    % Create a wide figure to accommodate three horizontal subplots
    fig = figure('Name', sprintf('%s Statistical Distribution', metric_name), ...
           'Position', [150, 150, 1500, 450]);

    clean_metric = metric_data(~isnan(metric_data));
    max_val = max(clean_metric);

    % Histogram
    figure(fig);
    subplot(1, 3, 1);
    histogram(metric_data, 50, 'Normalization', 'probability', 'FaceColor', [0.2 0.5 0.7]);
    title(sprintf('%s Histogram', metric_name), 'FontSize', 12);
    xlabel(metric_name, 'FontSize', 11);
    ylabel('Probability', 'FontSize', 11);
    grid on;
    % Add reference lines to histogram
    hold on;
    ybounds_hist = ylim;
    if max_val >= 0.1, plot([0.1 0.1], ybounds_hist, 'k--', 'HandleVisibility', 'off'); end
    if max_val >= 0.2, plot([0.2 0.2], ybounds_hist, 'k--', 'HandleVisibility', 'off'); end
    if max_val >= 0.4, plot([0.4 0.4], ybounds_hist, 'k--', 'HandleVisibility', 'off'); end
    hold off;

    % Boxplot
    figure(fig);
    subplot(1, 3, 2);
    boxchart(metric_data);
    xticklabels({'Absolute Error'}); % This replaces the 'Labels' argument
    title(sprintf('%s Boxplot', metric_name), 'FontSize', 12);
    ylabel(sprintf('%s', metric_name));
    grid on;

    % Empirical CDF
    figure(fig);
    subplot(1, 3, 3); 
    x_cdf = sort(clean_metric); 
    y_cdf = (1:length(x_cdf)) / length(x_cdf); 
    stairs(x_cdf, y_cdf, 'LineWidth', 1.5, 'Color', [0.8 0.3 0.2]); 
    title(sprintf('%s Empirical CDF', metric_name), 'FontSize', 12);
    xlabel(metric_name, 'FontSize', 11);
    ylabel('Cumulative Probability (F(x))', 'FontSize', 11);
    grid on;
    % Add reference lines for standard FSV categorical limits
    hold on;
    ybounds = ylim;
    if max(x_cdf) >= 0.1, plot([0.1 0.1], ybounds, 'k--', 'HandleVisibility', 'off'); end
    if max(x_cdf) >= 0.2, plot([0.2 0.2], ybounds, 'k--', 'HandleVisibility', 'off'); end
    if max(x_cdf) >= 0.4, plot([0.4 0.4], ybounds, 'k--', 'HandleVisibility', 'off'); end
    hold off;

    drawnow;
end
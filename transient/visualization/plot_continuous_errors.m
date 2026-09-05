function plot_continuous_errors(pred, act, metric_name)
    raw_error = pred - act;
    abs_error = abs(raw_error);

    fig = figure('Name', sprintf('Error Analysis: %s', metric_name), 'Position', [100, 100, 1000, 800], 'Visible', 'on');

    % 1. Predicted vs Actual Plot
    figure(fig);
    subplot(2, 2, 1);
    scatter(act, pred, 20, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    min_val = min([act(:); pred(:)]);
    max_val = max([act(:); pred(:)]);
    plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 1.5);
    title('Predicted vs Actual');
    xlabel(sprintf('Actual %s', metric_name));
    ylabel(sprintf('Predicted %s', metric_name));
    grid on;

    % 2. Error Histogram
    figure(fig);
    subplot(2,2,2);
    histogram(raw_error, 30, 'Normalization', 'probability');
    xline(0, 'r--', 'LineWidth', 1.5);
    title('Error Histogram (Raw Error)');
    xlabel('Error (Predicted - Actual)');
    ylabel('Probability');
    grid on;

    % 3. Box Plot
    figure(fig); 
    subplot(2,2,3);
    boxchart(abs_error);
    xticklabels({'Absolute Error'}); % This replaces the 'Labels' argument
    title('Absolute Error Spread');
    ylabel(sprintf('Absolute Error in %s', metric_name));
    grid on;

    % 4. Cumulative Distribution Function (CDF)
    figure(fig);
    subplot(2,2,4);
    valid_err = abs_error(~isnan(abs_error)); % Remove NaNs
    x_cdf = sort(valid_err); % Sort the errors in ascending order (X-axis)
    y_cdf = (1:length(x_cdf)) / length(x_cdf); % Calculate cumulative probability
    stairs(x_cdf, y_cdf, 'LineWidth', 1.5);  
    title('CDF of Absolute Error');
    xlabel(sprintf('Absolute Error in %s', metric_name));
    ylabel('Cumulative Probability');
    grid on;

    drawnow;
end
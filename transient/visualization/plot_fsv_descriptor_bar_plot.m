function plot_fsv_descriptor_bar_plot(category_data, metric_name, start_geom, max_geoms)
    % Create figure
    figure('Name', sprintf('%s Descriptor Distribution', metric_name), ...
           'Position', [100, 100, 1200, 500]);
       
    % Plot stacked bars
    x_axis = start_geom:(start_geom + max_geoms - 1);
    b = bar(x_axis, category_data, 'stacked');
    
    % FSV Interpretation Scale Categories[cite: 1]
    labels = {'Excellent (<0.1)', 'Very Good (0.1-0.2)', 'Good (0.2-0.4)', ...
              'Fair (0.4-0.8)', 'Poor (0.8-1.6)', 'Very Poor (>1.6)'};
    
    % Intuitive Traffic-Light Colormap
    colors = [0.15 0.55 0.15;  % Excellent: Dark Green
              0.45 0.75 0.25;  % Very Good: Light Green
              0.90 0.80 0.10;  % Good: Yellowish
              0.95 0.50 0.10;  % Fair: Orange
              0.85 0.15 0.15;  % Poor: Red
              0.50 0.00 0.00]; % Very Poor: Dark Red
              
    for i = 1:6
        b(i).FaceColor = colors(i, :);
    end
    
    % Formatting
    legend(labels, 'Location', 'eastoutside', 'FontSize', 10);
    title(sprintf('%s - FSV Quality Distribution per Geometry (18x18 Matrix)', metric_name), 'FontSize', 12);
    xlabel('Geometry Index', 'FontSize', 11);
    ylabel(sprintf('Proportion of %s values', metric_name), 'FontSize', 11);
    ylim([0 1]);
    xlim([start_geom - 1, start_geom + max_geoms]);
    grid on;
end
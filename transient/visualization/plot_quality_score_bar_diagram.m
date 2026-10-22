function plot_quality_score_bar_diagram(quality_scores, metric)
    % PLOT_QUALITY_SCORE_BAR_DIAGRAM Calculates percentages of quality groups 
    % and plots a bar diagram based on IEEE 370 metrics.
    
    % Validate inputs
    arguments
        quality_scores (:,1) double
        metric (1,:) char {mustBeMember(metric, {'causality', 'passivity'})}
    end
    
    total_geoms = length(quality_scores);
    metric = lower(metric); % ensure case-insensitivity
    
    % Calculate counts based on the metric
    if strcmp(metric, 'causality')
        num_good = sum(quality_scores >= 80);
        num_acceptable = sum(quality_scores >= 50 & quality_scores < 80);
        num_inconclusive = sum(quality_scores >= 20 & quality_scores < 50);
        num_poor = sum(quality_scores < 20);
        metric_title = 'Causality';
    else % passivity
        num_good = sum(quality_scores >= 99.9);
        num_acceptable = sum(quality_scores >= 99 & quality_scores < 99.9);
        num_inconclusive = sum(quality_scores >= 80 & quality_scores < 99); 
        num_poor = sum(quality_scores < 80);
        metric_title = 'Passivity';
    end
    
    % Convert counts to percentages
    pct_good = (num_good / total_geoms) * 100;
    pct_acceptable = (num_acceptable / total_geoms) * 100;
    pct_inconclusive = (num_inconclusive / total_geoms) * 100;
    pct_poor = (num_poor / total_geoms) * 100;
    
    percentages = [pct_good, pct_acceptable, pct_inconclusive, pct_poor];
    categories = {'Good', 'Acceptable', 'Inconclusive', 'Poor'};
    
    % Create a categorical array and reorder to prevent alphabetical sorting on X-axis
    x_categorical = reordercats(categorical(categories), categories);
    
    % Generate the plot
    figure('Name', sprintf('%s Quality', metric_title), 'Color', 'w');
    b = bar(x_categorical, percentages);
    
    % Style the bar diagram
    title(sprintf('IEEE 370 %s Quality Scores', metric_title), 'FontSize', 14);
    ylabel('Percentage of Geometries (%)', 'FontSize', 12);
    ylim([0, 100]); % Set Y-axis from 0 to 100%
    grid on;
    
    % Add percentage labels on top of each bar
    xtips = b.XEndPoints;
    ytips = b.YEndPoints;
    labels = string(round(percentages, 1)) + "%";
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 11, 'FontWeight', 'bold');
        
    % Optional: Color-code the bars for visual clarity
    b.FaceColor = 'flat';
    b.CData(1,:) = [0.4660 0.6740 0.1880]; % Green for Good
    b.CData(2,:) = [0.9290 0.6940 0.1250]; % Yellow for Acceptable
    b.CData(3,:) = [0.8500 0.3250 0.0980]; % Orange for Inconclusive
    b.CData(4,:) = [0.6350 0.0780 0.1840]; % Red for Poor
end
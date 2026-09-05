function plot_eye_pred_vs_act(t, eye_matrix_Vout_pred, eye_matrix_Vout_actual, plot_title, eye_matrix_Vout_pred_adj)
    arguments
        t
        eye_matrix_Vout_pred
        eye_matrix_Vout_actual
        plot_title (1,1) string = "Eye Diagram Prediction Vs Actual"
        eye_matrix_Vout_pred_adj = []
    end
    
    fig = figure('Name', plot_title);
    ax = axes('Parent', fig);
    hold(ax, 'on');
    grid(ax, 'on');
    
    % Plot Actual First
    plot(ax, t, eye_matrix_Vout_actual, 'r', 'Color', [1 0 0 0.1]); 
    
    if ~isempty(eye_matrix_Vout_pred_adj)
        plot(ax, t, eye_matrix_Vout_pred, 'b', 'Color', [0 0 1 0.05]); % Lighter opacity for raw
        plot(ax, t, eye_matrix_Vout_pred_adj, 'g', 'Color', [0 1 0 0.1]); % Green for adjusted
        
        h_leg_actual   = plot(ax, NaN, NaN, 'r', 'LineWidth', 2); 
        h_leg_pred_raw = plot(ax, NaN, NaN, 'b', 'LineWidth', 2); 
        h_leg_pred_adj = plot(ax, NaN, NaN, 'g', 'LineWidth', 2); 
        legend(ax, [h_leg_actual, h_leg_pred_raw, h_leg_pred_adj], {'Actual', 'Predicted (Raw)', 'Predicted (Adjusted)'}, 'Location', 'best');
    else
        plot(ax, t, eye_matrix_Vout_pred, 'b', 'Color', [0 0 1 0.1]); 
        
        h_leg_actual = plot(ax, NaN, NaN, 'r', 'LineWidth', 2); 
        h_leg_pred   = plot(ax, NaN, NaN, 'b', 'LineWidth', 2); 
        legend(ax, [h_leg_actual, h_leg_pred], {'Actual', 'Predicted'}, 'Location', 'best');
    end
    
    title(ax, sprintf('%s', plot_title));
    xlabel(ax, 'Time (UI)');
    ylabel(ax, 'Voltage (V)');
    xlim(ax, [0 2]);
    hold(ax, 'off');
end
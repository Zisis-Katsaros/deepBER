function plot_step_response_pred_vs_act(t, V_in, V_out_main_pred, V_out_main_act, V_out_next1_pred, V_out_next1_act, V_out_fext1_pred, ...
    V_out_fext1_act, V_out_next2_pred, V_out_next2_act, V_out_fext2_pred, V_out_fext2_act, plot_title, V_out_main_pred_adj, V_out_target)
    arguments
        t
        V_in
        V_out_main_pred
        V_out_main_act
        V_out_next1_pred = []
        V_out_next1_act = []
        V_out_fext1_pred = []
        V_out_fext1_act = []
        V_out_next2_pred = []
        V_out_next2_act = []
        V_out_fext2_pred = [] 
        V_out_fext2_act = []
        plot_title (1,1) string = "Step Response Prediction Vs Actual"
        V_out_main_pred_adj = []
        V_out_target = []
    end

    figure('Name', plot_title);
    grid on;
    plot_handles = [];
    legend_labels = {};

    % Vin
    h_in = plot(t*1e9, V_in, 'k--', 'LineWidth', 1.5); hold on;
    plot_handles = [plot_handles, h_in];
    legend_labels = [legend_labels, {'V_{in} (Source)'}];

    % Vout_main
    h_main_act = plot(t*1e9, V_out_main_act, '-', 'Color', [0, 0, 0.8], 'LineWidth', 1.5);
    
    if ~isempty(V_out_main_pred_adj)
        h_main_pred = plot(t*1e9, V_out_main_pred, ':', 'Color', [0.4, 0.4, 1.0], 'LineWidth', 1.5); % Lighter dotted for raw
        h_main_pred_adj = plot(t*1e9, V_out_main_pred_adj, '--', 'Color', [0, 0.5, 0], 'LineWidth', 1.5); % Distinct Green for adjusted
        plot_handles = [plot_handles, h_main_act, h_main_pred, h_main_pred_adj];
        legend_labels = [legend_labels, {'V_{out, HFSS} (Main)', 'V_{out, Raw DNN} (Main)', 'V_{out, Adjusted} (Main)'}];
    else
        h_main_pred = plot(t*1e9, V_out_main_pred, '--', 'Color', [0, 0, 0.8], 'LineWidth', 1.2);
        plot_handles = [plot_handles, h_main_act, h_main_pred];
        legend_labels = [legend_labels, {'V_{out, HFSS} (Main)', 'V_{out, DNN} (Main)'}];
    end

    % Optional Target Line
    if ~isempty(V_out_target)
        h_target = yline(V_out_target, 'm-.', 'LineWidth', 1.5);
        plot_handles = [plot_handles, h_target];
        legend_labels = [legend_labels, {'V_{out} Target'}];
    end
    
    %  1st adjacent line
    if ~isempty(V_out_next1_act) && ~isempty(V_out_next1_pred)
        h_n1_act = plot(t*1e9, V_out_next1_act, '-', 'Color', [0, 0.8, 0.8], 'LineWidth', 1.2);
        h_n1_pred = plot(t*1e9, V_out_next1_pred, '--', 'Color', [0, 0.8, 0.8], 'LineWidth', 1.0);
        plot_handles = [plot_handles, h_n1_act, h_n1_pred];
        legend_labels = [legend_labels, {'NEXT_{HFSS} (Adj 1)', 'NEXT_{DNN} (Adj 1)'}];
    end

    % 1st far-end line
    if ~isempty(V_out_fext1_act) && ~isempty(V_out_fext1_pred)
        h_f1_act = plot(t*1e9, V_out_fext1_act, '-', 'Color', [0.8, 0, 0.4], 'LineWidth', 1.2);
        h_f1_pred = plot(t*1e9, V_out_fext1_pred, '--', 'Color', [0.8, 0, 0.4], 'LineWidth', 1.0);
        plot_handles = [plot_handles, h_f1_act, h_f1_pred];
        legend_labels = [legend_labels, {'FEXT_{HFSS} (Adj 1)', 'FEXT_{DNN} (Adj 1)'}];
    end

    % 2nd adjacent line
    if ~isempty(V_out_next2_act) && ~isempty(V_out_next2_pred)
        h_n2_act = plot(t*1e9, V_out_next2_act, '-', 'Color', [0, 0.8, 0.4], 'LineWidth', 1.2);
        h_n2_pred = plot(t*1e9, V_out_next2_pred, '--', 'Color', [0, 0.8, 0.4], 'LineWidth', 1.0);
        plot_handles = [plot_handles, h_n2_act, h_n2_pred];
        legend_labels = [legend_labels, {'NEXT_{HFSS} (Adj 2)', 'NEXT_{DNN} (Adj 2)'}];
    end

    % 2nd far-end line
    if ~isempty(V_out_fext2_act) && ~isempty(V_out_fext2_pred)
        h_f2_act = plot(t*1e9, V_out_fext2_act, '-', 'Color', [0.8, 0, 0.8], 'LineWidth', 1.2);
        h_f2_pred = plot(t*1e9, V_out_fext2_pred, '--', 'Color', [0.8, 0, 0.8], 'LineWidth', 1.0);
        plot_handles = [plot_handles, h_f2_act, h_f2_pred];
        legend_labels = [legend_labels, {'FEXT_{HFSS} (Adj 2)', 'FEXT_{DNN} (Adj 2)'}];
    end

    title(plot_title);
    xlabel('Time (ns)');
    ylabel('Voltage (V)');
    legend(plot_handles, legend_labels, 'Location', 'best');
    hold off;
end
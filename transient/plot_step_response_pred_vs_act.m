function plot_step_response_pred_vs_act(t, V_in, V_out_main_pred, V_out_main_act, V_out_next1_pred, V_out_next1_act, V_out_fext1_pred, ...
    V_out_fext1_act, V_out_next2_pred, V_out_next2_act, V_out_fext2_pred, V_out_fext2_act, plot_title)
    %{
    Plots predicted and actual curves of main and adjacent channels with 0->1 step stimulus
    Inputs:
    - t: Time vector
    - V_in: Vin values
    - V_out_main_*: Vout of main channel (predicted and actual)
    - V_out_next1_*: Crosstalk response of the adjacent channel (above tx)
    - V_out_fext1_*: Crosstalk response of the adjacent far-end channel (above rx)
    - V_out_next2_*: Crosstalk response of the adjacent channel (below tx)
    - V_out_fext2_*: Crosstalk response of the adjacent far-end channel (below rx)
    - title: Title of the plot
    Outputs:
    *none*
    %}
    arguments
        t
        V_in
        V_out_main_pred
        V_out_main_act

        % Crosstalk
        V_out_next1_pred = []
        V_out_next1_act = []
        V_out_fext1_pred = []
        V_out_fext1_act = []
        V_out_next2_pred = []
        V_out_next2_act = []
        V_out_fext2_pred = [] 
        V_out_fext2_act = []

        plot_title (1,1) string = "Step Response Prediction Vs Actual"
    end

    figure('Name', plot_title);
    grid on;
    % Initialize arrays to hold plot handles and legend labels dynamically
    plot_handles = [];
    legend_labels = {};

    % Vin
    h_in = plot(t*1e9, V_in, 'k--', 'LineWidth', 1.5); hold on;
    plot_handles(end+1) = h_in;
    legend_labels{end+1} = 'V_{in} (Source)';

    % Vout_main
    h_main_act = plot(t*1e9, V_out_main_act, '-', 'Color', [0, 0, 0.8], 'LineWidth', 1.5);
    h_main_pred = plot(t*1e9, V_out_main_pred, '--', 'Color', [0, 0, 0.8], 'LineWidth', 1.2);
    plot_handles(end+1:end+2) = [h_main_act, h_main_pred];
    legend_labels(end+1:end+2) = {'V_{out, HFSS} (Main)', 'V_{out, DNN} (Main)'};
    
    %  1st adjacent line
    if ~isempty(V_out_next1_act) && ~isempty(V_out_next1_pred)
        h_n1_act = plot(t*1e9, V_out_next1_act, '-', 'Color', [0, 0.8, 0.8], 'LineWidth', 1.2);
        h_n1_pred = plot(t*1e9, V_out_next1_pred, '--', 'Color', [0, 0.8, 0.8], 'LineWidth', 1.0);

        plot_handles(end+1:end+2) = [h_n1_act, h_n1_pred];
        legend_labels(end+1:end+2) = {'NEXT_{HFSS} (Adj 1)', 'NEXT_{DNN} (Adj 1)'};
    end

    % 1st far-end line
    if ~isempty(V_out_fext1_act) && ~isempty(V_out_fext1_pred)
        h_f1_act = plot(t*1e9, V_out_fext1_act, '-', 'Color', [0.8, 0, 0.4], 'LineWidth', 1.2);
        h_f1_pred = plot(t*1e9, V_out_fext1_pred, '--', 'Color', [0.8, 0, 0.4], 'LineWidth', 1.0);
        
        plot_handles(end+1:end+2) = [h_f1_act, h_f1_pred];
        legend_labels(end+1:end+2) = {'FEXT_{HFSS} (Adj 1)', 'FEXT_{DNN} (Adj 1)'};
    end

    % 2nd adjacent line
    if ~isempty(V_out_next2_act) && ~isempty(V_out_next2_pred)
        h_n2_act = plot(t*1e9, V_out_next2_act, '-', 'Color', [0, 0.8, 0.4], 'LineWidth', 1.2);
        h_n2_pred = plot(t*1e9, V_out_next2_pred, '--', 'Color', [0, 0.8, 0.4], 'LineWidth', 1.0);

        plot_handles(end+1:end+2) = [h_n2_act, h_n2_pred];
        legend_labels(end+1:end+2) = {'NEXT_{HFSS} (Adj 2)', 'NEXT_{DNN} (Adj 2)'};
    end

    % 2nd far-end line
    if ~isempty(V_out_fext2_act) && ~isempty(V_out_fext2_pred)
        h_f2_act = plot(t*1e9, V_out_fext2_act, '-', 'Color', [0.8, 0, 0.8], 'LineWidth', 1.2);
        h_f2_pred = plot(t*1e9, V_out_fext2_pred, '--', 'Color', [0.8, 0, 0.8], 'LineWidth', 1.0);
        
        plot_handles(end+1:end+2) = [h_f2_act, h_f2_pred];
        legend_labels(end+1:end+2) = {'FEXT_{HFSS} (Adj 2)', 'FEXT_{DNN} (Adj 2)'};
    end

    
    title(plot_title);
    xlabel('Time (ns)');
    ylabel('Voltage (V)');
    legend(plot_handles, legend_labels, 'Location', 'best');
    hold off;
end
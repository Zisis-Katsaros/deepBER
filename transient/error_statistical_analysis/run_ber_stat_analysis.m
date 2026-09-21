function run_ber_stat_analysis(all_ber_preds, all_ber_acts, show_plots, log_ber)
     % Initialize dynamic table variables
    table_columns = {};
    var_names = {};
    Row_Metrics = {};

    if log_ber
        all_ber_preds = log10(all_ber_preds);
        all_ber_acts = log10(all_ber_acts);
    end

    ber_stats = calculate_continuous_stats(all_ber_preds, all_ber_acts);
    Row_Metrics = fieldnames(ber_stats);
    table_columns{end+1} = cell2mat(struct2cell(ber_stats));
    var_names = [{'BER'}];

    % DISPLAY TABLE 
    % Construct the table dynamically based on collected columns
    stats_table = table(table_columns{:}, 'RowNames', Row_Metrics, 'VariableNames', var_names);
    disp(stats_table);

    if show_plots
        plot_continuous_errors(all_ber_preds, all_ber_acts, 'Bit Error Rate (BER)');
    end
end

function run_error_stat_analysis(all_PRBS_EH_preds, all_PRBS_EH_acts, all_PRBS_EW_preds, all_PRBS_EW_acts, all_PDA_EH_preds, all_PDA_EH_acts, all_PDA_EW_preds, ... 
    all_PDA_EW_acts, all_Verdict_preds, all_Verdict_acts, show_plots)

    % Detect which analyses were run based on data presence
    has_prbs = ~isempty(all_PRBS_EH_acts);
    has_pda  = ~isempty(all_PDA_EH_acts);

    if ~has_prbs && ~has_pda
        fprintf('No data provided for statistical analysis.\n');
        return;
    end

    % Initialize dynamic table variables
    table_columns = {};
    var_names = {};
    Row_Metrics = {};

    % PRBS STATS
    if has_prbs
        PRBS_EH_stats = calculate_continuous_stats(all_PRBS_EH_preds, all_PRBS_EH_acts);
        PRBS_EW_stats = calculate_continuous_stats(all_PRBS_EW_preds, all_PRBS_EW_acts);
        
        if isempty(Row_Metrics)
            Row_Metrics = fieldnames(PRBS_EH_stats); 
        end
        
        table_columns{end+1} = cell2mat(struct2cell(PRBS_EH_stats));
        table_columns{end+1} = cell2mat(struct2cell(PRBS_EW_stats));
        var_names = [var_names, {'PRBS_Eye_Height_V', 'PRBS_Eye_Width_s'}];
    end

    % PDA STATS
    if has_pda
        PDA_EH_stats = calculate_continuous_stats(all_PDA_EH_preds, all_PDA_EH_acts);
        PDA_EW_stats = calculate_continuous_stats(all_PDA_EW_preds, all_PDA_EW_acts);
        
        if isempty(Row_Metrics)
            Row_Metrics = fieldnames(PDA_EH_stats); 
        end
        
        table_columns{end+1} = cell2mat(struct2cell(PDA_EH_stats));
        table_columns{end+1} = cell2mat(struct2cell(PDA_EW_stats));
        var_names = [var_names, {'PDA_Eye_Height_V', 'PDA_Eye_Width_s'}];
    end

    % DISPLAY TABLE 
    % Construct the table dynamically based on collected columns
    stats_table = table(table_columns{:}, 'RowNames', Row_Metrics, 'VariableNames', var_names);
    disp(stats_table);

    % PDA VERDICT STATS
    if has_pda && ~isempty(all_Verdict_acts)
        fprintf('\n--- PDA VERDICT (PASS/FAIL) ---\n');
        calculate_verdict_stats(all_Verdict_preds, all_Verdict_acts);
    end

    % PLOTS
    if show_plots
        if has_prbs
            plot_continuous_errors(all_PRBS_EH_preds, all_PRBS_EH_acts, 'PRBS Eye Height (V)');
            plot_continuous_errors(all_PRBS_EW_preds, all_PRBS_EW_acts, 'PRBS Eye Width (s)');
        end

        if has_pda
            plot_continuous_errors(all_PDA_EH_preds, all_PDA_EH_acts, 'PDA Eye Height (V)');
            plot_continuous_errors(all_PDA_EW_preds, all_PDA_EW_acts, 'PDA Eye Width (s)');
        end
    end
end
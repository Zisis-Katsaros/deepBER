function run_ber_stat_analysis(all_ber_preds, all_ber_acts, show_plots, error_free_threshold, log_ber)
    arguments
        all_ber_preds (:,1) double
        all_ber_acts (:,1) double
        show_plots (1,1) logical
        error_free_threshold (1,1) double {mustBeNonnegative} = 0
        log_ber (1,1) logical = true
    end
    
    % Initialize dynamic table variables
    table_columns = {};
    
    % Determine which elements to keep BEFORE applying max() or log10()
    perform_filtering = (error_free_threshold > 0);
    if perform_filtering
        non_error_free_indices = (all_ber_preds > error_free_threshold);
    end

    % Apply max bounds
    preds = max(all_ber_preds, error_free_threshold);
    acts = max(all_ber_acts, error_free_threshold);

    % Apply log transformation
    if log_ber
        preds = log10(preds);
        acts = log10(acts);
    end

    % Compute overall statistics
    ber_stats = calculate_continuous_stats(preds, acts);
    Row_Metrics = fieldnames(ber_stats);
    
    % Convert struct to column array, handling any unexpected empty metrics
    c = struct2cell(ber_stats);
    c(cellfun(@isempty, c)) = {NaN}; 
    table_columns{end+1} = cellfun(@(x) double(x(1)), c);
    var_names = {'BER'};

    % Compute filtered statistics
    if perform_filtering
        filtered_preds = preds(non_error_free_indices);
        filtered_acts = acts(non_error_free_indices);

        % Safety check to ensure data exists after filtering
        if isempty(filtered_preds)
            warning('No data points exceeded the error-free threshold. Skipping filtered statistics.');
        else
            ber_stats_filtered = calculate_continuous_stats(filtered_preds, filtered_acts);
            
            c_filt = struct2cell(ber_stats_filtered);
            c_filt(cellfun(@isempty, c_filt)) = {NaN};
            table_columns{end+1} = cellfun(@(x) double(x(1)), c_filt);
             
            var_names = [var_names, {'BER_non_error_free'}]; 
        end
    end

    % DISPLAY TABLE 
    stats_table = table(table_columns{:}, 'RowNames', Row_Metrics, 'VariableNames', var_names);
    disp(stats_table);

    if show_plots
        plot_continuous_errors(preds, acts, 'Bit Error Rate (BER)');
    end
end
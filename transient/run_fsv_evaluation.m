function run_fsv_evaluation(pred_file, act_file)
    act_obj = sparameters(act_file);
    act_params = act_obj.Parameters;
    num_ports = size(act_params, 1);

    for port_i = 1:num_ports
        for port_j = port_i:num_ports
            [ADM, FDM, GDM, ADMi, FDMi, GDMi, ADMc, FDMc, GDMc] = perform_fsv(pred_file, act_file, port_i, port_j);
            
            % Display results
            fprintf('Port Pair (%d,%d):\n', port_i, port_j);
            fprintf('ADM: %.4f\n', ADM);
            fprintf('FDM: %.4f\n', FDM);
            fprintf('GDM: %.4f\n', GDM);
            fprintf('ADMc: %.4f\n', mean(ADMc));
            fprintf('FDMc: %.4f\n', mean(FDMc));
            fprintf('GDMc: %.4f\n', mean(GDMc));
            fprintf('\n');
        end
    end
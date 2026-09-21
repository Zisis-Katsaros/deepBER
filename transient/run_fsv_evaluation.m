function [geom_ADMc, geom_FDMc, geom_GDMc, ADM_matrix, FDM_matrix, GDM_matrix] = run_fsv_evaluation(pred_file, act_file, geom_idx, show_plots)
    act_obj = sparameters(act_file);
    act_params = act_obj.Parameters;
    num_ports = size(act_params, 1);

    % Initialize matrices to store the average measures for the whole S-matrix
    ADM_matrix = zeros(num_ports, num_ports);
    FDM_matrix = zeros(num_ports, num_ports);
    GDM_matrix = zeros(num_ports, num_ports);

    fprintf('[FSV Evaluation] Beginning FSV evaluation for Geometry %d\n', geom_idx);
    for port_i = 1:num_ports
        for port_j = port_i:num_ports
            [ADM, FDM, GDM, ADMi, FDMi, GDMi, ADMc, FDMc, GDMc] = perform_fsv(pred_file, act_file, port_i, port_j);
        
            % Store results
            ADM_matrix(port_i, port_j) = ADM;
            ADM_matrix(port_j, port_i) = ADM; % Symmetric
            FDM_matrix(port_i, port_j) = FDM;
            FDM_matrix(port_j, port_i) = FDM;
            GDM_matrix(port_i, port_j) = GDM;
            GDM_matrix(port_j, port_i) = GDM;

            geom_ADMc = calc_confidence(ADM_matrix(:));
            geom_FDMc = calc_confidence(FDM_matrix(:));
            geom_GDMc = calc_confidence(GDM_matrix(:));
        end
    end

    if show_plots
        visualize_fsv(ADM_matrix, FDM_matrix, GDM_matrix, geom_idx);
    end
end
function mape = safe_mape(actual, predicted, zero_thresh)
    % Calculates MAPE safely to avoid divide-by-zero spikes on closed eyes
    if isnan(actual) || actual <= zero_thresh
        if isnan(predicted) || predicted <= zero_thresh
            mape = 0; % Both accurately report complete closure
        else
            mape = 100; % Act closed, Pred open (Max Error)
        end
    else
        mape = abs((actual - predicted) / actual) * 100;
    end
end
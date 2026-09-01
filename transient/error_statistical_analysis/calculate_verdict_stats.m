function calculate_verdict_stats(pred, act)
    % Logical indexing for confusion matrix
    TP = sum(pred == 1 & act == 1);
    TN = sum(pred == 0 & act == 0);
    FP = sum(pred == 1 & act == 0); % FATAL: Predicted pass, actual fail
    FN = sum(pred == 0 & act == 1); % Costly: Predicted fail, actual pass

    total = length(act);
    accuracy = (TP + TN) / total;
    
    % Safety against divide by zero
    recall_fail = TN / max((TN + FP), 1); % How many actual failures where detected
    
    % Matthews Correlation Coefficient (MCC) - ranges from -1 to +1
    mcc_num = (TP * TN) - (FP * FN);
    mcc_den = sqrt(double((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
    if mcc_den == 0; mcc_den = 1; end
    mcc = mcc_num / mcc_den;

    fprintf('Confusion Matrix:\n');
    fprintf('                 Actual PASS    Actual FAIL\n');
    fprintf('Predicted PASS       %4d           %4d  <-- (False Positives/Fatal)\n', TP, FP);
    fprintf('Predicted FAIL       %4d           %4d\n', FN, TN);
    
    fprintf('\nMetrics:\n');
    fprintf('  Accuracy:           %.2f%%\n', accuracy * 100);
    fprintf('  Failure Catch Rate: %.2f%% (Recall of Failing class)\n', recall_fail * 100);
    fprintf('  MCC Score:          %.4f (-1 to 1, higher is better)\n', mcc);
end
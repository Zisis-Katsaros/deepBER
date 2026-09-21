function ADMi = calc_ADMi(Lo1, Lo2)
    N = length(Lo1);
    denom = sum(abs(Lo1) + abs(Lo2)) / N;
    ADMi = abs(Lo1 - Lo2) ./ (denom + eps);
end
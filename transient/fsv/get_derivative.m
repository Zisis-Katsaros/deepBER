function d = get_derivative(x, step)
    % Simple difference approach ignoring x-axis data scale
    d = zeros(size(x));
    d(step+1:end-step) = x(2*step+1:end) - x(1:end-2*step);
end
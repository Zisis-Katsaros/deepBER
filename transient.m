%% Transient analysis script

%% 1. Load S-parameter matrix (Touchstone)
filename = 'csv_files/s_params/s18p/geom_1.s18p'; 
S_data = sparameters(filename);
freq = S_data.Frequencies;

% Configure tx, rx, next and fext ports
tx_port = 5;
rx_port = 14;
next_port = 4;
fext_port = 15;

% Extract specific frequency-domain arrays
S_main = squeeze(S_data.Parameters(rx_port, tx_port, :));
S_next = squeeze(S_data.Parameters(next_port, tx_port, :));
S_fext = squeeze(S_data.Parameters(fext_port, tx_port, :));

%% 2. Convert S-parameters to time-domain impulse responses (Vector Fitting)
fprintf('Fitting rational models. This may take a moment...\n');
fit_main = rationalfit(freq, S_main, 'Tolerance', -40); % -40dB precision
fit_next = rationalfit(freq, S_next, 'Tolerance', -40);
fit_fext = rationalfit(freq, S_fext, 'Tolerance', -40);
fprintf('Models fitted successfully.\n');

%% 3. Setup time vectors
fs = 1e12;               % 1 THz sampling rate 
Ts = 1/fs;               % 1ps Sampling period
t_step = 2e-9;           % 2 ns duration for Step Response
t = (0:Ts:t_step)';


%% 4. Stimulus
rise_time = 40e-12;
delay = 100e-12; 

% Create a linear ramp from 0V to 1V
V_in_step = min(max((t - delay) / rise_time, 0), 1);

%% 5. Evaluate step response
V_out_main_step = timeresp(fit_main, V_in_step, Ts) / 2;
V_out_next_step = timeresp(fit_next, V_in_step, Ts) / 2;
V_out_fext_step = timeresp(fit_fext, V_in_step, Ts) / 2;

%% 6. PRBS for eye diagram
bit_rate = 10e9;         % 10 Gbps data rate
bit_period = 1/bit_rate;
num_bits = 1000;         % Simulate 1000 bits

% Generate random 1s and 0s
bits = randi([0 1], num_bits, 1);

% Create the PRBS time vector
t_prbs = (0:Ts:(num_bits*bit_period))';

V_in_prbs = zeros(size(t_prbs));
for i = 1:num_bits
    if bits(i) == 1
        % Superimpose a shifted step up
        V_in_prbs = V_in_prbs + min(max((t_prbs - (i-1)*bit_period) / rise_time, 0), 1);
        % Superimpose a shifted step down at the end of the bit
        V_in_prbs = V_in_prbs - min(max((t_prbs - i*bit_period) / rise_time, 0), 1);
    end
end

%% 7. Evaluate PRBS response
fprintf('Calculating PRBS Transient...\n');
V_out_main_prbs = timeresp(fit_main, V_in_prbs, Ts) / 2;

%% 8. Plotting results
% --- Plot 1: Step Response & Crosstalk ---
figure('Name', 'Step Response and Crosstalk');
plot(t*1e9, V_in_step, 'k--', 'LineWidth', 1.5); hold on;
plot(t*1e9, V_out_main_step, 'b', 'LineWidth', 1.5);
plot(t*1e9, V_out_next_step, 'r', 'LineWidth', 1.2);
plot(t*1e9, V_out_fext_step, 'g', 'LineWidth', 1.2);
grid on;
title('Transient Step Response (40ps Rise Time)');
xlabel('Time (ns)');
ylabel('Voltage (V)');
legend('V_{in} (Source)', 'V_{out} (Main Channel)', 'NEXT (Adjacent Tx)', 'FEXT (Adjacent Rx)', 'Location', 'best');

% --- Plot 2: Eye Diagram ---
% We reshape the long PRBS vector to fold it over itself based on the bit period
samples_per_bit = round(bit_period * fs);
% Discard the first few bits to allow the transient to settle
settle_bits = 10;
valid_samples = (num_bits - settle_bits) * samples_per_bit;

% Reshape into a matrix where each column is 2 bit periods wide (standard for eye diagrams)
eye_matrix_Vout = reshape(V_out_main_prbs(settle_bits*samples_per_bit + 1 : settle_bits*samples_per_bit + valid_samples), ...
                          samples_per_bit * 2, []);

time_eye = linspace(0, 2, samples_per_bit * 2);

figure('Name', 'Receiver Eye Diagram');
plot(time_eye, eye_matrix_Vout, 'b', 'Color', [0 0 1 0.1]); % 10% opacity for density effect
grid on;
title(sprintf('Eye Diagram @ %g Gbps', bit_rate/1e9));
xlabel('Time (UI)');
ylabel('Voltage (V)');
xlim([0 2]);



import torch
import torch.nn as nn
import torch.fft
import math

class GaussianSmoothingLayer(nn.Module):
    def __init__(self, channels, kernel_size=11):
        """
        # GaussianSmoothingLayer

        ## Learnable Smoothing Layer using an adaptive Gaussian filter.
        Applies a channel-wise 1D convolution where weights are defined by a learnable standard deviation.
        """
        super(GaussianSmoothingLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Learnable standard deviation per channel
        self.sigma = nn.Parameter(torch.ones(channels, 1, 1))
        grid = torch.arange(kernel_size).float() - (kernel_size - 1) / 2 # standard grid for kernel computation
        self.register_buffer('grid', grid.view(1, 1, -1))

    def forward(self, x):
        # Calculate Gaussian kernel dynamically based on the learnable sigma
        varience = self.sigma ** 2
        gaussian_kernel = torch.exp(-0.5 * (self.grid ** 2) / varience)
        gaussian_kernel = gaussian_kernel / torch.sqrt(2 * math.pi * varience)

        # Channel-wise filtering
        return nn.functional.conv1d(x, gaussian_kernel, padding=self.padding, groups=self.channels)
    

class CausalityEnforcementLayer(nn.Module):
    def __init__(self, N, M, K):
        """
        # CausalityEnforcementLayer
        ## Reconstructs the imaginary part from the extrapolated real part using Kramers-Kronig relations, thus ensuring causality

        ## Args:
        N: Number of frequency points int he training data
        M: Extrapolation factor
        K: Interpolation/Truncation factor
        """

        super(CausalityEnforcementLayer, self).__init__()
        self.N = N
        self.M = M
        self.K = K

    def forward(self, y_sl):
        # y_sl is the smoothed extrapolated real part Size(Nd, Dy, N*M + 1)
        Ny, Dy, seq_len = y_sl.shape

        # Make double sided
        y_rev = torch.flip(y_sl[:, :, 1:], dims=[2])  # Reverse the sequence dimension
        y_double = torch.cat([y_sl, y_rev], dim=2)

        # FFT along the frequency axis
        Y_tilde = torch.fft.fft(y_double, dim=2)

        # Create discrete analytic signal in v-domain
        Z_v = torch.zeros_like(Y_tilde)

        Z_v[:, :, 0] = Y_tilde[:, :, 0]  # DC component remains unchanged
        Z_v[:, :, 1:seq_len] = 2 * Y_tilde[:, :, 1:seq_len]  # Positive frequencies doubled
        # Negative frequencies remain zero

        # IFFT to transform back and extract real and imaginary parts
        z_analytic = torch.fft.ifft(Z_v, dim=2)

        # Apply factor K for interpolation
        y_cel = self.K * z_analytic.real
        z_cel = self.K * z_analytic.imag

        # Truncate to N*K + 1, discarding out-of-band predictions
        target_len = int(self.N * self.K + 1)
        S_real = y_cel[:, :, :target_len]
        S_imag = z_cel[:, :, :target_len]

        return S_real, S_imag
    

class PassivityEnforcementLayer(nn.Module):
    def __init__(self, num_ports):
        """
        # PassivityEnforcementLayer

        ## Enforces maximum singular value <=1 using a minimum-phase filter

        ## Args:
        - num_ports: Number of ports
        """
        super(PassivityEnforcementLayer, self).__init__()
        self.P = num_ports

    def reconstruct_symmetric_mat(self, S_real, S_imag):
        Nd, Dy, F = S_real.shape
        mat_real = torch.zeros((Nd, self.P, self.P, F), device=S_real.device)
        mat_imag = torch.zeros((Nd, self.P, self.P, F), device=S_real.device)

        idx = 0
        for i in range(self.P):
            for j in range(i, self.P):
                mat_real[:, i, j, :] = S_real[:, idx, :]
                mat_real[:, j, i, :] = S_real[:, idx, :] # due to symmetry
                mat_imag[:, i, j, :] = S_imag[:, idx, :]
                mat_imag[:, j, i, :] = S_imag[:, idx, :] # due to symmetry
                idx += 1
        return mat_real, mat_imag
    
    def forward(self, S_real, S_imag):
        # Reshape into matrix form
        mat_real, mat_imag = self.reconstruct_symmetric_mat(S_real, S_imag)

        # Create isomorphic matrix for complex operations using real tensors
        # SP = [[Re, Im], 
        #       [-Im, Re]]
        row1 = torch.cat([mat_real, mat_imag], dim=2)
        row2 = torch.cat([-mat_imag, mat_real], dim=2)
        Sp = torch.cat([row1, row2], dim=1)  # Shape: (Nd, 2P, 2P, F)

        # Rearrange to compute batched matrix operations 
        Sp = Sp.permute(0, 3, 1, 2)  # Shape: (Nd, F, 2P, 2P)
        Sp_T = Sp.transpose(-1, -2)  # Transpose for S^H

        # Calculate upper bound max singular value
        C_f = torch.sum(Sp ** 2, dim=(2,3)) / 2 # /2 due to isomorphic representation

        # D calculated via Hadamard product
        SH_S = torch.matmul(Sp_T, Sp)
        D_f = torch.sum(SH_S * SH_S, dim=(2,3)) / 2 

        # Calculate sigma_1 upper bound
        term1 = C_f / self.P
        term2 = ((self.P - 1) / self.P) * (D_f - (C_f ** 2) / self.P)
        term2 = torch.clamp(term2, min=0.0)  # Prevent NaNs from numerical instability
        sigma_bound = torch.sqrt(term1 + torch.sqrt(term2))

        # Enforce passivity using minimum-phase filter: 1/sigma_bound if > 1, else 1
        mag_filter = torch.where(sigma_bound > 1.0, 1.0 / sigma_bound, torch.ones_like(sigma_bound))

        # FFT-based Hilbert Transform to compute find minimum phase
        log_mag = torch.log(mag_filter + 1e-12)  # Avoid log(0)
        log_mag_double = torch.cat([log_mag, torch.flip(log_mag[:, 1:], dims=[1])], dim=1)  # Make double-sided
        fft_log = torch.fft.fft(log_mag_double, dim=1)

        Z_v = torch.zeros_like(fft_log)
        seq_len = log_mag.shape[1]
        Z_v[:, 0] = fft_log[:, 0]  # DC component remains unchanged
        Z_v[:, 1:seq_len] = 2 * fft_log[:, 1:seq_len]  # Positive frequencies doubled
        # Negative frequencies remain zero

        phase_filter = -torch.fft.ifft(Z_v, dim=1).imag[:, :seq_len]  

        # Apply complex passivity enforcement filter
        filter_real = mag_filter * torch.cos(phase_filter)
        filter_imag = mag_filter * torch.sin(phase_filter)

        # Expand filter to match the shape of S
        filter_real_exp = filter_real.unsqueeze(1)
        filter_imag_exp = filter_imag.unsqueeze(1)

        # S_pel = S_cel * Sigma
        S_pel_real = S_real * filter_real_exp - S_imag * filter_imag_exp
        S_pel_imag = S_real * filter_imag_exp + S_imag * filter_real_exp

        return S_pel_real, S_pel_imag

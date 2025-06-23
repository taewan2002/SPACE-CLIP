# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss.
    
    This loss is commonly used for monocular depth estimation tasks. It measures the
    structural similarity between the predicted and target depth maps, invariant
    to scale and shift. This implementation is inspired by the loss functions used
    in papers like MiDaS and AdaBins.

    Args:
        variance_focus (float): A hyperparameter to control the focus between the
            variance of the log-difference and the mean of the log-difference.
            A higher value focuses more on variance. Default: 0.85 (as in MiDaS).
    """
    def __init__(self, variance_focus: float = 0.85):
        super().__init__()
        self.name = 'SILogLoss'
        self.variance_focus = variance_focus

    def forward(self, 
                input_depth: torch.Tensor, 
                target_depth: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                interpolate: bool = True, 
                eps: float = 1e-7
               ) -> torch.Tensor:
        """
        Calculates the SILog loss.

        Args:
            input_depth (torch.Tensor): Predicted depth map of shape (B, 1, H, W) or (B, H, W).
            target_depth (torch.Tensor): Ground truth depth map of shape (B, 1, H, W) or (B, H, W).
            mask (torch.Tensor, optional): A boolean tensor where True indicates valid pixels.
                                           Shape should be broadcastable to input_depth. Defaults to None.
            interpolate (bool): If True, resizes input_depth to match target_depth's spatial dimensions.
                                Defaults to True.
            eps (float): A small epsilon to avoid numerical issues like log(0). Defaults to 1e-7.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        # Ensure inputs are float32 for stability
        input_depth = input_depth.to(torch.float32)
        target_depth = target_depth.to(torch.float32)

        # Ensure 4D tensor format (B, 1, H, W)
        if input_depth.ndim == 3:
            input_depth = input_depth.unsqueeze(1)
        if target_depth.ndim == 3:
            target_depth = target_depth.unsqueeze(1)
        
        # Interpolate if spatial dimensions don't match
        if interpolate and input_depth.shape[-2:] != target_depth.shape[-2:]:
            input_depth = F.interpolate(input_depth, size=target_depth.shape[-2:],
                                        mode='bilinear', align_corners=True)

        # Prepare mask
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # Ensure mask is boolean and matches the interpolated input shape
            if mask.shape != input_depth.shape:
                mask = F.interpolate(mask.float(), size=input_depth.shape[-2:], mode='nearest').to(torch.bool)
        
        # Clamp values to avoid log(0)
        input_depth_valid = torch.clamp(input_depth, min=eps)
        target_depth_valid = torch.clamp(target_depth, min=eps)

        # Apply mask to select valid pixels
        if mask is not None:
            # If no pixels are valid in the mask, return 0 loss
            if not mask.any():
                return torch.tensor(0.0, device=input_depth.device, requires_grad=True)
            
            input_pixels = input_depth_valid[mask]
            target_pixels = target_depth_valid[mask]
        else:
            input_pixels = input_depth_valid.reshape(-1)
            target_pixels = target_depth_valid.reshape(-1)

        # If masking results in no pixels, return 0 loss
        if input_pixels.numel() == 0:
            return torch.tensor(0.0, device=input_depth.device, requires_grad=True)

        # Calculate the log-space difference
        log_diff = torch.log(input_pixels) - torch.log(target_pixels)

        # Calculate variance and mean squared for the log-difference
        num_pixels = log_diff.numel()
        if num_pixels < 2: # Variance is not well-defined for a single element
            loss_var = torch.tensor(0.0, device=log_diff.device)
        else:
            loss_var = torch.var(log_diff)
        
        loss_mean_sq = torch.pow(torch.mean(log_diff), 2)

        # Combine variance and mean-squared terms
        # Dg = (1/N) * sum(g_i^2) - (lambda/N^2) * (sum(g_i))^2
        # which is equivalent to Var(g) + (1-lambda) * Mean(g)^2
        # Here, lambda is self.variance_focus
        alpha_for_mean_sq = 1.0 - self.variance_focus
        d_g = loss_var + alpha_for_mean_sq * loss_mean_sq

        # The final loss is often scaled (e.g., by 10) and rooted
        return 10 * torch.sqrt(torch.clamp(d_g, min=eps))

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    
    This loss function measures the structural similarity between two images,
    which is often more aligned with human visual perception than traditional
    pixel-wise losses like L1 or L2. It considers luminance, contrast, and
    structure. This implementation is designed to be a differentiable loss
    function for training neural networks.

    Args:
        window_size (int): The size of the Gaussian window to use for SSIM calculation.
                           Must be an odd number. Defaults to 11.
        sigma (float): The standard deviation of the Gaussian filter. Defaults to 1.5.
        data_range (float or int): The dynamic range of the input images (i.e., the
                                   difference between the maximum and minimum possible
                                   values). Defaults to 1.0 for normalized images.
        channels (int): The number of channels in the input images. Defaults to 1
                        for depth maps.
    """
    def __init__(self, 
                 window_size: int = 11, 
                 sigma: float = 1.5, 
                 data_range: float = 1.0, 
                 channels: int = 1):
        super().__init__()
        self.name = 'SSIMLoss'
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.channels = channels
        
        # Create a 1D Gaussian kernel
        self.gaussian_kernel = self._create_gaussian_kernel(self.window_size, self.sigma)
        
        # Create a 2D Gaussian window from the 1D kernel
        self.window = self._create_2d_window(self.window_size, self.channels)

        # SSIM constants
        self.K1 = 0.01
        self.K2 = 0.03
        self.C1 = (self.K1 * self.data_range) ** 2
        self.C2 = (self.K2 * self.data_range) ** 2

    def _create_gaussian_kernel(self, window_size: int, sigma: float) -> torch.Tensor:
        """Creates a 1D Gaussian kernel."""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum() # Normalize
        return g

    def _create_2d_window(self, window_size: int, channels: int) -> nn.Parameter:
        """Creates a 2D Gaussian window suitable for convolution."""
        kernel_1d = self._create_gaussian_kernel(window_size, self.sigma).unsqueeze(1)
        kernel_2d = kernel_1d @ kernel_1d.t()
        
        # Reshape to (channels, 1, window_size, window_size) for conv2d
        window = kernel_2d.expand(channels, 1, window_size, window_size)
        
        # Register as a parameter, but don't train it
        return nn.Parameter(window, requires_grad=False)

    def _ssim(self, 
              img1: torch.Tensor, 
              img2: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The core SSIM calculation function."""
        
        # Move window to the same device as images
        window = self.window.to(img1.device)

        # Calculate means
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2
        
        # Calculate SSIM components
        ssim_numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        ssim_denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        ssim_map = ssim_numerator / ssim_denominator
        
        # Contrast sensitivity component (often used for MS-SSIM, but useful here too)
        cs_numerator = 2 * sigma12 + self.C2
        cs_denominator = sigma1_sq + sigma2_sq + self.C2
        cs_map = cs_numerator / cs_denominator
        
        return ssim_map, cs_map


    def forward(self, 
                input_depth: torch.Tensor, 
                target_depth: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                interpolate: bool = True
               ) -> torch.Tensor:
        """
        Calculates the SSIM loss. The loss is 1 - SSIM.

        Args:
            input_depth (torch.Tensor): Predicted depth map of shape (B, 1, H, W).
            target_depth (torch.Tensor): Ground truth depth map of shape (B, 1, H, W).
            mask (torch.Tensor, optional): A boolean tensor for valid pixels. Defaults to None.
            interpolate (bool): If True, resizes input_depth to match target_depth's dimensions.
                                Defaults to True.

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        input_depth = input_depth.to(torch.float32)
        target_depth = target_depth.to(torch.float32)

        if input_depth.ndim == 3: input_depth = input_depth.unsqueeze(1)
        if target_depth.ndim == 3: target_depth = target_depth.unsqueeze(1)

        if interpolate and input_depth.shape[-2:] != target_depth.shape[-2:]:
            input_depth = F.interpolate(input_depth, size=target_depth.shape[-2:],
                                        mode='bilinear', align_corners=True)
            
        # SSIM is sensitive to the dynamic range of the data.
        # Normalize depth maps to [0, 1] range for consistent SSIM calculation.
        # This is a common practice for depth maps which can have arbitrary scales.
        min_depth = torch.min(target_depth)
        max_depth = torch.max(target_depth)
        
        input_norm = (input_depth - min_depth) / (max_depth - min_depth)
        target_norm = (target_depth - min_depth) / (max_depth - min_depth)
        
        # Clamp to [0, 1] range
        input_norm = torch.clamp(input_norm, 0, 1)
        target_norm = torch.clamp(target_norm, 0, 1)
        
        # Calculate SSIM
        ssim_val, _ = self._ssim(input_norm, target_norm)

        # The loss is 1 - mean(SSIM)
        loss = 1.0 - ssim_val

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 3: mask = mask.unsqueeze(1)
            if mask.shape != loss.shape:
                mask = F.interpolate(mask.float(), size=loss.shape[-2:], mode='nearest').to(torch.bool)
            
            # Apply mask and calculate mean only on valid pixels
            if mask.any():
                loss = loss[mask].mean()
            else:
                return torch.tensor(0.0, device=input_depth.device, requires_grad=True)
        else:
            loss = loss.mean()
            
        return loss
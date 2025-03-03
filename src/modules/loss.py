import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim


class MAELoss(nn.Module):
    """
    Mean Absolute Error (MAE) loss.
    """

    def __init__(self) -> None:
        """
        Initializes the MAELoss.
        """
        super(MAELoss, self).__init__()
        self.logger = logging.getLogger(__name__)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MAELoss.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: MAE loss value.
        """
        try:
            return torch.mean(torch.abs(input - target))
        except Exception as e:
            self.logger.error(f"Error during MAE loss calculation: {e}")
            raise


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) loss.
    """

    def __init__(self, data_range: float = 1.0, size_average: bool = True, channel: int = 3) -> None:
        """
        Initializes the MSSSIMLoss.

        Args:
            data_range (float): The maximum value of the input data (e.g., 1.0 for normalized images).
            size_average (bool): Whether to average the loss over the batch.
            channel (int): Number of channels in the input images.
        """
        super(MSSSIMLoss, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.data_range = data_range
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MSSSIMLoss.

        Args:
            img1 (torch.Tensor): Input image tensor.
            img2 (torch.Tensor): Target image tensor.

        Returns:
            torch.Tensor: MS-SSIM loss value.
        """
        try:
            img1 = img1.cpu().detach().numpy().transpose(0, 2, 3, 1)
            img2 = img2.cpu().detach().numpy().transpose(0, 2, 3, 1)

            msssim_val = []
            for i in range(img1.shape[0]):
                msssim_val.append(ssim(img1[i], img2[i], data_range=self.data_range, multichannel=True, win_size=11))

            msssim_val = torch.tensor(msssim_val)

            if self.size_average:
                return 1 - torch.mean(msssim_val)
            else:
                return 1 - msssim_val.mean(0)

        except Exception as e:
            self.logger.error(f"Error during MS-SSIM loss calculation: {e}")
            raise


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create an instance of the MAELoss
        mae_loss = MAELoss()

        # Create an instance of the MSSSIMLoss
        msssim_loss = MSSSIMLoss(data_range=1.0, size_average=True, channel=3)

        # Generate random input and target tensors
        batch_size = 4
        channels = 3
        height = 256
        width = 256
        input_tensor = torch.randn(batch_size, channels, height, width)
        target_tensor = torch.randn(batch_size, channels, height, width)

        # Calculate the MAE loss
        mae_value = mae_loss(input_tensor, target_tensor)

        # Calculate the MS-SSIM loss
        msssim_value = msssim_loss(input_tensor, target_tensor)

        # Print the loss values
        logger.info(f"MAE Loss: {mae_value.item()}")
        logger.info(f"MS-SSIM Loss: {msssim_value.item()}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
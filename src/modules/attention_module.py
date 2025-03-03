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
from typing import Optional


class ChannelAttention(nn.Module):
    """
    Channel attention module.
    """

    def __init__(self, num_features: int, reduction_ratio: int = 16) -> None:
        """
        Initializes the ChannelAttention module.

        Args:
            num_features (int): Number of input feature maps.
            reduction_ratio (int): Reduction ratio for the channel attention module.
        """
        super(ChannelAttention, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_features, num_features // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(num_features // reduction_ratio, num_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ChannelAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features, height, width).
        """
        try:
            avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
            return x * self.sigmoid(out)
        except Exception as e:
            self.logger.error(f"Error during channel attention forward pass: {e}")
            raise


class SpatialAttention(nn.Module):
    """
    Spatial attention module.
    """

    def __init__(self) -> None:
        """
        Initializes the SpatialAttention module.
        """
        super(SpatialAttention, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpatialAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features, height, width).
        """
        try:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(out)
            return x * self.sigmoid(out)
        except Exception as e:
            self.logger.error(f"Error during spatial attention forward pass: {e}")
            raise


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create an instance of the ChannelAttention module
        num_features = 64
        channel_attention = ChannelAttention(num_features=num_features, reduction_ratio=16)

        # Create an instance of the SpatialAttention module
        spatial_attention = SpatialAttention()

        # Generate a random input tensor
        batch_size = 1
        height = 256
        width = 256
        input_tensor = torch.randn(batch_size, num_features, height, width)

        # Perform a forward pass through the ChannelAttention module
        channel_output = channel_attention(input_tensor)

        # Perform a forward pass through the SpatialAttention module
        spatial_output = spatial_attention(input_tensor)

        # Print the output shapes
        logger.info(f"Input shape: {input_tensor.shape}")
        logger.info(f"Channel Attention Output shape: {channel_output.shape}")
        logger.info(f"Spatial Attention Output shape: {spatial_output.shape}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
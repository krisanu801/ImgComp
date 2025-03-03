import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import logging
from typing import Optional

# Local imports
try:
    from src.modules.attention_module import ChannelAttention, SpatialAttention  # Assuming attention_module.py contains ChannelAttention and SpatialAttention
except ImportError as e:
    print(f"ImportError: {e}.  Please ensure your project structure is correct and the necessary files exist.")
    sys.exit(1)


class AttentionCNN(nn.Module):
    """
    Attention-based CNN model for image compression post-processing.
    """

    def __init__(self, num_channels: int = 3, num_features: int = 64, reduction_ratio: int = 16, num_attention_blocks: int = 2) -> None:
        """
        Initializes the AttentionCNN model.

        Args:
            num_channels (int): Number of input channels (e.g., 3 for RGB images).
            num_features (int): Number of feature maps in the convolutional layers.
            reduction_ratio (int): Reduction ratio for the channel attention module.
            num_attention_blocks (int): Number of attention blocks to use.
        """
        super(AttentionCNN, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(num_features, reduction_ratio) for _ in range(num_attention_blocks)
        ])

        self.conv2 = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initializes the weights of the convolutional layers using Kaiming He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_channels, height, width).
        """
        try:
            x = self.conv1(x)
            x = self.relu1(x)

            for block in self.attention_blocks:
                x = block(x)

            x = self.conv2(x)
            return x
        except Exception as e:
            self.logger.error(f"Error during forward pass: {e}")
            raise


class AttentionBlock(nn.Module):
    """
    Attention block consisting of channel and spatial attention modules.
    """

    def __init__(self, num_features: int, reduction_ratio: int = 16) -> None:
        """
        Initializes the AttentionBlock.

        Args:
            num_features (int): Number of input feature maps.
            reduction_ratio (int): Reduction ratio for the channel attention module.
        """
        super(AttentionBlock, self).__init__()
        self.channel_attention = ChannelAttention(num_features, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_features, height, width).
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create an instance of the AttentionCNN model
        model = AttentionCNN(num_channels=3, num_features=64, reduction_ratio=16, num_attention_blocks=2)

        # Generate a random input tensor
        batch_size = 1
        num_channels = 3
        height = 256
        width = 256
        input_tensor = torch.randn(batch_size, num_channels, height, width)

        # Perform a forward pass
        output_tensor = model(input_tensor)

        # Print the output shape
        logger.info(f"Input shape: {input_tensor.shape}")
        logger.info(f"Output shape: {output_tensor.shape}")

        # Print the model architecture
        logger.info(f"Model architecture:\n{model}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
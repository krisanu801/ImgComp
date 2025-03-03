import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import torch
import logging

# Local imports
try:
    from src.modules.attention_module import ChannelAttention, SpatialAttention
    from src.modules.loss import MAELoss, MSSSIMLoss
except ImportError as e:
    print(f"ImportError: {e}.  Please ensure your project structure is correct and the necessary files exist.")
    sys.exit(1)


class TestAttentionModules(unittest.TestCase):
    """
    Unit tests for the attention modules and loss functions.
    """

    def setUp(self):
        """
        Setup method to initialize the modules and logger.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.num_features = 64
        self.reduction_ratio = 16
        self.batch_size = 2
        self.height = 32
        self.width = 32
        self.channel_attention = ChannelAttention(num_features=self.num_features, reduction_ratio=self.reduction_ratio)
        self.spatial_attention = SpatialAttention()
        self.mae_loss = MAELoss()
        self.msssim_loss = MSSSIMLoss()

    def test_channel_attention_forward(self):
        """
        Test the forward pass of the ChannelAttention module.
        """
        try:
            input_tensor = torch.randn(self.batch_size, self.num_features, self.height, self.width)
            output_tensor = self.channel_attention(input_tensor)
            self.assertEqual(output_tensor.shape, input_tensor.shape)
            self.logger.info("ChannelAttention forward pass test passed.")
        except Exception as e:
            self.logger.error(f"ChannelAttention forward pass test failed: {e}")
            self.fail(f"ChannelAttention forward pass test failed: {e}")

    def test_spatial_attention_forward(self):
        """
        Test the forward pass of the SpatialAttention module.
        """
        try:
            input_tensor = torch.randn(self.batch_size, self.num_features, self.height, self.width)
            output_tensor = self.spatial_attention(input_tensor)
            self.assertEqual(output_tensor.shape, input_tensor.shape)
            self.logger.info("SpatialAttention forward pass test passed.")
        except Exception as e:
            self.logger.error(f"SpatialAttention forward pass test failed: {e}")
            self.fail(f"SpatialAttention forward pass test failed: {e}")

    def test_mae_loss_forward(self):
        """
        Test the forward pass of the MAELoss function.
        """
        try:
            input_tensor = torch.randn(self.batch_size, self.num_features, self.height, self.width)
            target_tensor = torch.randn(self.batch_size, self.num_features, self.height, self.width)
            loss_value = self.mae_loss(input_tensor, target_tensor)
            self.assertIsInstance(loss_value, torch.Tensor)
            self.logger.info("MAELoss forward pass test passed.")
        except Exception as e:
            self.logger.error(f"MAELoss forward pass test failed: {e}")
            self.fail(f"MAELoss forward pass test failed: {e}")

    def test_msssim_loss_forward(self):
        """
        Test the forward pass of the MSSSIMLoss function.
        """
        try:
            input_tensor = torch.randn(self.batch_size, 3, self.height, self.width)
            target_tensor = torch.randn(self.batch_size, 3, self.height, self.width)
            loss_value = self.msssim_loss(input_tensor, target_tensor)
            self.assertIsInstance(loss_value, torch.Tensor)
            self.logger.info("MSSSIMLoss forward pass test passed.")
        except Exception as e:
            self.logger.error(f"MSSSIMLoss forward pass test failed: {e}")
            self.fail(f"MSSSIMLoss forward pass test failed: {e}")


if __name__ == '__main__':
    # Example Usage:
    # To run the tests, execute this file directly:
    # python test/modules/test_modules.py
    unittest.main()
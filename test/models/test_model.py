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
    from src.models.model import AttentionCNN  # Assuming model.py contains AttentionCNN
except ImportError as e:
    print(f"ImportError: {e}.  Please ensure your project structure is correct and the necessary files exist.")
    sys.exit(1)


class TestAttentionCNN(unittest.TestCase):
    """
    Unit tests for the AttentionCNN model.
    """

    def setUp(self):
        """
        Setup method to initialize the model and logger.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model = AttentionCNN(num_channels=3, num_features=64, reduction_ratio=16, num_attention_blocks=2)

    def test_model_creation(self):
        """
        Test that the model can be created successfully.
        """
        try:
            model = AttentionCNN(num_channels=3, num_features=64, reduction_ratio=16, num_attention_blocks=2)
            self.assertIsInstance(model, AttentionCNN)
            self.logger.info("Model creation test passed.")
        except Exception as e:
            self.logger.error(f"Model creation test failed: {e}")
            self.fail(f"Model creation test failed: {e}")

    def test_forward_pass(self):
        """
        Test the forward pass of the model with a dummy input.
        """
        try:
            batch_size = 1
            num_channels = 3
            height = 256
            width = 256
            input_tensor = torch.randn(batch_size, num_channels, height, width)

            output_tensor = self.model(input_tensor)

            self.assertEqual(output_tensor.shape, (batch_size, num_channels, height, width))
            self.logger.info("Forward pass test passed.")
        except Exception as e:
            self.logger.error(f"Forward pass test failed: {e}")
            self.fail(f"Forward pass test failed: {e}")

    def test_weight_initialization(self):
        """
        Test that the weights are initialized correctly.
        """
        try:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.assertFalse(torch.all(module.weight == 0), f"Weights of {name} are not initialized.")
            self.logger.info("Weight initialization test passed.")
        except Exception as e:
            self.logger.error(f"Weight initialization test failed: {e}")
            self.fail(f"Weight initialization test failed: {e}")

    def test_device_placement(self):
        """
        Test that the model can be moved to a device (CPU or GPU).
        """
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            self.model.to(device)
            input_tensor = torch.randn(1, 3, 256, 256).to(device)
            output_tensor = self.model(input_tensor)

            self.assertTrue(output_tensor.device == device)
            self.logger.info("Device placement test passed.")
        except Exception as e:
            self.logger.error(f"Device placement test failed: {e}")
            self.fail(f"Device placement test failed: {e}")


if __name__ == '__main__':
    # Example Usage:
    # To run the tests, execute this file directly:
    # python test/models/test_model.py
    unittest.main()
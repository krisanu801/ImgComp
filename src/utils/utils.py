import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import logging
import yaml
from typing import Tuple, Any
import torch.nn as nn
import torch.optim as optim


def setup_logging(log_file: str) -> logging.Logger:
    """
    Sets up logging to a file and console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(epoch: int, model: nn.Module, optimizer: optim.Optimizer, checkpoint_path: str) -> None:
    """
    Saves a checkpoint of the model and optimizer state.

    Args:
        epoch (int): The current epoch number.
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        checkpoint_path (str): Path to save the checkpoint.
    """
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        logging.getLogger(__name__).info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving checkpoint: {e}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[nn.Module, optim.Optimizer, int]:
    """
    Loads a checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer.
        device (torch.device): The device to load the checkpoint onto.

    Returns:
        Tuple[nn.Module, optim.Optimizer, int]: A tuple containing the loaded model, optimizer, and epoch number.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        logging.getLogger(__name__).info(f"Checkpoint loaded from {checkpoint_path}")
        return model, optimizer, epoch
    except FileNotFoundError:
        logging.getLogger(__name__).warning(f"Checkpoint file not found: {checkpoint_path}")
        return model, optimizer, 0  # Return 0 for the starting epoch if no checkpoint is found
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading checkpoint: {e}")
        raise


if __name__ == '__main__':
    # Example usage
    # 1. Setup logging
    logger = setup_logging('test.log')
    logger.info("Logging setup complete.")

    # 2. Save and load a dummy checkpoint
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyModel()
    dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=0.001)
    checkpoint_path = 'dummy_checkpoint.pth'

    try:
        save_checkpoint(10, dummy_model, dummy_optimizer, checkpoint_path)
        logger.info("Dummy checkpoint saved.")

        loaded_model = DummyModel()
        loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)
        loaded_model, loaded_optimizer, loaded_epoch = load_checkpoint(checkpoint_path, loaded_model, loaded_optimizer, torch.device('cpu'))

        logger.info(f"Dummy checkpoint loaded. Epoch: {loaded_epoch}")

        # Clean up the dummy checkpoint file
        os.remove(checkpoint_path)
        logger.info("Dummy checkpoint file removed.")

    except Exception as e:
        logger.error(f"An error occurred during the example: {e}")
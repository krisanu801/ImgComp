import sys
import os

# Dynamically adjust sys.path to allow imports from the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
from torchvision import transforms

# Local imports
try:
    from src.models.model import AttentionCNN  # Assuming model.py contains AttentionCNN
    from src.data.dataset import ImageDataset  # Assuming dataset.py contains ImageDataset
    from src.utils.utils import setup_logging, save_checkpoint, load_checkpoint  # Assuming utils.py contains these functions
except ImportError as e:
    print(f"ImportError: {e}.  Please ensure your project structure is correct and the necessary files exist.")
    sys.exit(1)


def train(config: Dict[str, Any], model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int, logger: logging.Logger, device: torch.device) -> None:
    """
    Trains the model for one epoch.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (optim.Optimizer): The optimizer used for training.
        criterion (nn.Module): The loss function.
        epoch (int): The current epoch number.
        logger (logging.Logger): Logger for logging training progress.
        device (torch.device): The device to train on (CPU or GPU).
    """
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % config['logging_interval'] == 0:
            logger.info(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

    epoch_loss = running_loss / len(train_loader)
    logger.info(f"Epoch: {epoch}, Training Loss: {epoch_loss}")


def validate(config: Dict[str, Any], model: nn.Module, val_loader: DataLoader, criterion: nn.Module, epoch: int, logger: logging.Logger, device: torch.device) -> float:
    """
    Validates the model.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): The loss function.
        epoch (int): The current epoch number.
        logger (logging.Logger): Logger for logging validation progress.
        device (torch.device): The device to validate on (CPU or GPU).

    Returns:
        float: The validation loss.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    val_loss = running_loss / len(val_loader)
    logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}")
    return val_loss



def main(config: Dict[str, Any]) -> None:
    """
    Main function to train and evaluate the image compression model.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    # Setup logging
    logger = setup_logging(config['log_file'])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')
    logger.info(f"Using device: {device}")

    # Data loading
    try:
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = ImageDataset(config['train_data_dir'], transform=transform)  # Add transforms as needed
        val_dataset = ImageDataset(config['val_data_dir'], transform=transform)  # Add transforms as needed
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        logger.info("Data loaders initialized.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    # Model initialization
    try:
        model = AttentionCNN(num_channels=config['num_channels']).to(device)
        logger.info("Model initialized.")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Load checkpoint if specified
    if config['load_checkpoint']:
        try:
            model, optimizer, start_epoch = load_checkpoint(config['checkpoint_path'], model, optimizer, device)
            logger.info(f"Checkpoint loaded from {config['checkpoint_path']}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            start_epoch = 0
    else:
        start_epoch = 0

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config['num_epochs']):
        train(config, model, train_loader, optimizer, criterion, epoch, logger, device)
        val_loss = validate(config, model, val_loader, criterion, epoch, logger, device)

        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, model, optimizer, config['checkpoint_path'])
            logger.info(f"Checkpoint saved to {config['checkpoint_path']}")

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image compression model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

    main(config)

    # Example Usage (after training):
    # 1. Load a trained model:
    #    model = AttentionCNN(num_channels=config['num_channels']).to(device)
    #    model, _, _ = load_checkpoint('path/to/checkpoint.pth', model, None, device)
    #    model.eval() # Set to evaluation mode
    #
    # 2. Load an image and preprocess it:
    #    from PIL import Image
    #    image = Image.open('path/to/image.jpg').convert('RGB')
    #    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]) # Example transforms
    #    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    #
    # 3. Pass the image through the model:
    #    with torch.no_grad():
    #        compressed_image = model(input_tensor)
    #
    # 4. Post-process and save the compressed image:
    #    compressed_image = compressed_image.squeeze(0).cpu()
    #    compressed_image = transforms.ToPILImage()(compressed_image)
    #    compressed_image.save('path/to/compressed_image.jpg')
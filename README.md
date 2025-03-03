# ImgComp: Low-Bitrate Image Compression with Attention-Based CNNs

## Project Description

This project develops an innovative low-bitrate image compression model using PyTorch. It integrates convolutional neural networks (CNNs) with attention mechanisms for enhanced post-processing. The core of the project is an attention-based CNN architecture that leverages both channel and spatial attention to improve the quality of compressed images. The post-processing module is trained using Mean Absolute Error (MAE) and Multi-Scale Structural Similarity Index (MS-SSIM) loss functions to achieve visually appealing reconstructions. The model efficiently balances compression rates and perceptual quality while optimizing for real-world image compression scenarios.

## Project Structure

```
ImgComp/
├── src/
│   ├── models/
│   │   └── model.py          # Defines the attention-based CNN model architecture.
│   ├── modules/
│   │   ├── attention_module.py # Implements the channel and spatial attention modules.
│   │   └── loss.py           # Defines the MAE and MS-SSIM loss functions.
│   ├── data/
│   │   ├── dataset.py        # Handles data loading and preprocessing.
│   │   └── transforms.py     # Defines image transformations.
│   ├── utils/
│   │   └── utils.py          # Utility functions for training, evaluation, and logging.
│   └── main.py             # Main application file for training and evaluation.
├── configs/
│   ├── config.yaml         # Configuration file for training parameters, model settings, and data paths.
│   └── logging.conf        # Logging configuration file.
├── test/
│   ├── models/
│   │   └── test_model.py     # Unit tests for the model architecture.
│   └── modules/
│       └── test_modules.py   # Unit tests for the attention modules and loss functions.
├── data/                   # Directory for storing datasets (train, val)
├── requirements.txt        # Lists the project dependencies.
├── README.md               # Project documentation.
├── setup.py                # Setup file for packaging and distribution.
└── .gitignore              # Specifies intentionally untracked files that Git should ignore.
```

## Setup Instructions

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```
2.  **Activate the virtual environment:**

    *   **Linux/macOS:**

        ```bash
        source venv/bin/activate
        ```
    *   **Windows:**

        ```bash
        venv\Scripts\activate
        ```
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

The project dependencies are listed in `requirements.txt`.  Key dependencies include:

*   `torch`: PyTorch deep learning framework.
*   `torchvision`: PyTorch library for computer vision tasks.
*   `numpy`: Numerical computing library.
*   `Pillow`: Image processing library.
*   `PyYAML`: YAML parsing library.
*   `tqdm`: Progress bar library.
*   `scikit-image`: Image processing algorithms.
*   `pytest`: Testing framework.

## Usage

1.  **Configure the training parameters:**

    Modify the `configs/config.yaml` file to adjust training parameters, model settings, and data paths.
2.  **Run the training script:**

    ```bash
    python src/main.py --config configs/config.yaml
    ```

## Logging

The project uses a logging configuration defined in `configs/logging.conf`. Training progress and other relevant information are logged to the console and to a log file specified in the configuration.

## Testing

Unit tests are provided for the model architecture and attention modules. To run the tests:

```bash
python -m unittest discover -s test
```

## Checkpoints

The training script saves checkpoints of the model and optimizer state during training. The checkpoint path is specified in `configs/config.yaml`.

## Contributing

Contributions to this project are welcome. Please submit pull requests with detailed descriptions of the changes.

## License

[Specify the license for the project]

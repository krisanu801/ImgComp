import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sl2",  # Replace with your own username
    version="0.0.1",
    author="Example Author",  # Replace with your name
    author_email="author@example.com", # Replace with your email
    description="Low-Bitrate Image Compression with Attention-Based CNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/sl2", # Replace with your repo URL
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "torch==2.1.0",
        "torchvision==0.16.0",
        "numpy==1.26.2",
        "Pillow==10.1.0",
        "PyYAML==6.0.1",
        "tqdm==4.66.1",
        "scikit-image==0.22.0",
        "pytest==7.4.3"
    ],
    # Example Usage:
    # To build a distribution package:
    # python setup.py sdist bdist_wheel
    #
    # To install the package:
    # pip install .
    #
    # To upload to PyPI (after building):
    # twine upload dist/*
)
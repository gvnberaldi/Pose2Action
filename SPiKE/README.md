# Human Pose Estimation from Point Cloud

This repository define SPiKE, a deep learning model designed to perform human pose estimation from point cloud data. The model leverages advanced neural network techniques for interpreting 3D point clouds and provides a robust solution for tasks like human body pose detection and analysis.

## Features
- Human pose estimation from point clouds.
- Compatible with **CUDA** for GPU acceleration.
- Based on the **PointNet++** architecture for point cloud data processing.

## Requirements
- Python 3.12 with pip
- CUDA-compatible GPU
- Libraries listed in `requirements.txt`.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/gvnberaldi/SPiKE.git
    cd SPiKE
    ```

2. Set up a Python environment (recommended):
    ```bash
    python3.12 -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   
4. Ensure CUDA is installed and properly configured for GPU usage.

5. Install **PointNet++**. Follow the instructions provided in the [PointNet++ repository](https://github.com/charlesq34/pointnet2) to install it properly.
   ```bash
   cd modules
   python setup.py install
   ```
   
## Usage

### Training the Model
To train the model on the **BAD dataset**, run the following command:

```bash
python train_bad.py --config=config_file
```
Replace config_file with the path to your configuration file that specifies training parameters such as dataset path, learning rate, and number of epochs. You can modify the `generate_config.py` script to generate a custom configuration file.

### Inference on the Bad Dataset
To make predictions using the trained model on the bad dataset, run:

```bash
python predict_bad.py --config=configuration_file
```
Replace configuration_file with the path to your configuration file that specifies the model path, input data, and other relevant parameters for inference. You can modify the `generate_config.py` script to generate a custom configuration file.
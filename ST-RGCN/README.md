# Spatio-Temporal Relational Graph Convolution Network for Human Action Recognition from 3D Skeleton

## Overview
This repository contains the implementation of a Spatio-Temporal Relational Graph Convolution Network (ST-RGCN) for human action recognition from 3D skeleton data. The ST-GCN model is a powerful deep learning architecture for processing spatio-temporal data, particularly suited for tasks such as action recognition in videos or motion capture data.

## Features
- Implementation of ST-GCN architecture for human action recognition.
- Preprocessing scripts for converting 3D skeleton data into a suitable format.
- Training and evaluation scripts for the action recognition task.
- Pretrained models for quick deployment and experimentation.
- Example usage scripts and notebooks for getting started.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- [Optional] CUDA-enabled GPU for faster training

## Installation
1. Clone this repository:
   
   ```shell
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```
2. Install dependencies:
   
   ```shell
   pip install -r requirements.txt
   ```

## Usage
### Training
To train the ST-RGCN model on your dataset, run:

```shell
   python train.py --data data/ --batch_size 64 --epochs 50
```

### Evaluation
Evaluate the trained model on a test set:

```shell
   python evaluate.py --data data/ --model pretrained_model.pth
```
## Pretrained Models
We provide pretrained models for quick deployment and experimentation. You can find them in the pretrained_models directory.

## Contributing
Contributions are welcome! Feel free to open issues or pull requests for any improvements or bug fixes.

## Contact
For any inquiries, please contact [Your Name] at your.email@example.com.

## 🚀 Getting Started

### 1️⃣ **Setup**
- Ensure you have Conda installed.
- Create an environment and install dependencies:
```bash
conda env create -f environment.yaml
conda activate Multilayer_FFNN 
```

## Plot samples
```bash
python plot_samples.py
```

## Run Training Script

To run the training script `train.py`, use the following command:


```bash
python train.py --wandb_entity name --wandb_project projectname
```
## Run wandb sweep
```bash
wandb sweep sweep.yaml
wandb agent <sweep-ID>
```

## Directory Structure
```bash
ASSIGNMENT_1
│── wandb                  # Directory for Weights & Biases logs
│── .gitignore              # Git ignore file
│── config.py               # Configuration settings
│── dataset.py              # Dataset loading and processing
│── environment.yaml        # Conda environment setup
│── layer.py                # Neural network layer definitions
│── loss.py                 # Loss function implementations
│── model.py                # Model architecture
│── optimizers.py           # Optimizer implementations
│── plot_samples.py         # Visualization of sample results
│── README.md               # Project documentation
│── sweep.yaml              # W&B hyperparameter sweep configuration
│── train.py                # Training script
```

## Link to wandb Report

[WandB Report](https://wandb.ai/jay_gupta-indian-institute-of-technology-madras/Multilayer_FeedForward_Network/reports/Multilayer-Feedforward-Network--VmlldzoxMTcwMDA0Nw?accessToken=pcia1myw5f09hbwec4h1892cvenh03ipsufz4c886yritsb4161dcjk16oxuhsv5)

## Link to Github
[Repository](https://github.com/jaygupta907/da6401_assignment1)

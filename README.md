# Introduction to Deep Learning - Assignment 1

## Setup Environment

To create a virtual environment and install the required packages, run the following commands:

```bash
conda env create -f environment.yaml
conda activate Multilayer_FFNN  
```

## Run Training Script

To run the training script `train.py`, use the following command:


```bash
python train.py --wandb_entity name --wandb_project projectname
```
## Run wandb sweep
```bash
wandb sweep sweep.yaml
```

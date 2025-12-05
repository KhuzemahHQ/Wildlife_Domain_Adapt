# Wildlife_Domain_Adapt

This is our project for the CSC2529 Computational Inmaging Course at the University of Toronto.

# File Overview
- `model.py`: Contain model architecture used for the project
- `dataset.py`: Convert data into a pytorch dataset suitable for the project 
- `train.py`: Script to run the training process
- `evaluate.py`: Script to run the evaluation process
- `config.yaml`: Contains hyperparameters and set up input and output directories

# Data
Missouri Camera Traps [[LILA link](http://lila.science/datasets/missouricameratraps)]

Zhang, Z., He, Z., Cao, G., & Cao, W. (2016). Animal detection from highly cluttered natural scenes using spatiotemporal object region proposals and patch verification. IEEE Transactions on Multimedia, 18(10), 2079-2092.

Please refer to instructions on LILA to download the data

# Set Up
`pip install -r requirements.txt`

# Train
Set up `config.yaml` and run `python train.py`

# Evaluation
Set up `config.yaml` and run `python evaluate.py`
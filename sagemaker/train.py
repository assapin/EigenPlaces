import os

import numpy as np
import torch
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch

# Set up the SageMaker local session
sagemaker_local = LocalSession()
sagemaker_local.config = {'local': {'local_code': True}}

# Specify your training data and validation data locations
train_data = 'file://path_to_your_train_data'
validation_data = 'file://path_to_your_validation_data'
test_data = 'file://path_to_your_test_data'  # if applicable

# Define the estimator using the custom Docker image
estimator = Estimator(image_uri='597530458568.dkr.ecr.us-east-1.amazonaws.com/fingerpic/eigen-places-sm:latest',
                      role='SageMakerRole',  # this isn't used with local mode but is required
                      instance_count=1,
                      instance_type='local',  # use 'local_gpu' for GPU instances
                      hyperparameters={
                          'run_on_sm': True,
                          # ... add other hyperparameters you want to set
                      })

# Train the model locally
estimator.fit({'train': train_data, 'validation': validation_data, 'test': test_data})

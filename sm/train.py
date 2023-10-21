from sagemaker.local import LocalSession
from sagemaker.estimator import Estimator

# Set up the SageMaker local session
sagemaker_local = LocalSession()
sagemaker_local.config = {'local': {'local_code': True}}

base_path='/Users/assaf/projects/fingerpic/data/small'
# Specify your training data and validation data locations
train_data = f'{base_path}/train'
validation_data = f'{base_path}/val'
test_data = f'{base_path}/test'

# Define the estimator using the custom Docker image
estimator = Estimator(image_uri='597530458568.dkr.ecr.us-east-1.amazonaws.com/fingerpic/eigen-places-sm:latest',
                      role='SageMakerRole',  # this isn't used with local mode but is required
                      instance_count=1,
                      instance_type='local',  # use 'local_gpu' for GPU instances
                      hyperparameters={
                          'run_on_sm': True,

                      })

# Train the model locally
estimator.fit({'train': train_data, 'validation': validation_data, 'test': test_data})

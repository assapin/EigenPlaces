FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-cpu-py310-ubuntu20.04-sagemaker

# Install mesa-libGL (or any other dependencies)
RUN apt-get update && apt-get install -y libgl1-mesa-glx -y

# Set a working directory
WORKDIR /opt/ml/code

# Copy your code and modules into the Docker image
COPY  ./ .

# If you have a requirements.txt, install your Python dependencies
RUN pip install -r requirements.txt

# Set up the entry point so SageMaker knows how to start your training script
ENTRYPOINT ["python", "train.py"]


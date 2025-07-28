FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set up environment
ENV CONDA_ENV_NAME=neurobooster
ENV PATH=/opt/conda/envs/$CONDA_ENV_NAME/bin:/opt/conda/bin:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /app
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x Miniconda3-latest-Linux-x86_64.sh \
    && ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Copy environment.yml and create conda environment
COPY environment.yml /app/environment.yml
RUN /opt/conda/bin/conda env create -f /app/environment.yml

# Copy bash script and source code into the container
COPY run_example_docker.sh /app/run_example_docker.sh
COPY source /app/source

# Activate the conda environment
RUN echo "source activate $CONDA_ENV_NAME" > ~/.bashrc
ENV PATH=/opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH

# Set entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "neurobooster", "bash", "/app/run_exp_docker.sh"]

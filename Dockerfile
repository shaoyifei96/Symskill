# Use the official Ubuntu 22.04 image as the base
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /root

# Update package lists and install system-level dependencies in a single layer
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 && \
    # Clean up apt-get cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Clone the required git repositories
RUN git clone https://github.com/Carperis/robosuite.git && \
    git clone https://github.com/shaoyifei96/robocasa.git && \
    git clone https://github.com/felixzheng02/ds_policy.git

# Install the Python packages
RUN pip3 install numpy pytest && \
    pip3 install -e robosuite && \
    pip3 install -e robocasa && \
    pip3 install \
    numpy==1.23.3 \
    pytest>=8.3.4 \
    mypy==1.8.0 \
    gym==0.26.2 \
    matplotlib==3.6.2 \
    imageio==2.37.0 \
    imageio-ffmpeg \
    pandas==1.5.1 \
    torch==2.5.1 \
    scipy==1.13.1 \
    tabulate==0.9.0 \
    dill==0.3.5.1 \
    pyperplan \
    pathos \
    pillow==11.1.0 \
    requests \
    slack_bolt \
    pybullet>=3.2.0 \
    scikit-learn==1.1.2 \
    graphlib-backport \
    openai==1.19.0 \
    pyyaml==6.0.2 \
    pylint==2.14.5 \
    types-PyYAML \
    lisdf \
    seaborn==0.12.1 \
    "smepy @ git+https://github.com/sebdumancic/structure_mapping.git" \
    "pg3 @ git+https://github.com/tomsilver/pg3.git" \
    "gym_sokoban @ git+https://github.com/Learning-and-Intelligent-Systems/gym-sokoban.git" \
    ImageHash \
    google-generativeai \
    tenacity \
    httpx==0.27.0 \
    ruptures \
    pytest-cov==2.12.1 \
    pytest-pylint==0.18.0 \
    yapf==0.32.0 \
    docformatter==1.4 \
    isort==5.10.1
    
# upgrade pip and setuptools
RUN pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools && \
    pip3 install -e ds_policy

# Set the working directory to the robocasa repository
WORKDIR /root/robocasa

# Run the setup scripts for robocasa
RUN sed -i 's/input("The script will download.*")/"y"/' /root/robocasa/robocasa/scripts/download_kitchen_assets.py && \
    python3 /root/robocasa/robocasa/scripts/download_kitchen_assets.py

WORKDIR /root

# get files from another repo
RUN git clone https://github.com/shaoyifei96/predicators_robocasa_files.git && \
    cp predicators_robocasa_files/* .

# copy mosek license
RUN mkdir -p /root/.mosek && \
    cp /root/predicators_robocasa_files/mosek.lic /root/.mosek/mosek.lic

# Set the default command to execute when the container starts
CMD ["bash"]



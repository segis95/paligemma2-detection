#!/bin/bash

python3 -m venv ultralytics_env

source ultralytics_env/bin/activate

pip install --upgrade pip

pip install numpy pandas matplotlib ipython jupyter

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu125

pip install ultralytics

pip install tritonclient[all]

pip install opencv-python

pip install scikit-learn scipy

pip install torch transformers accelerate peft datasets evaluate

pip install pycocotools

pip install tqdm

pip install flash-attn --no-build-isolation

pip install -U "ray[default, data]"

pip install hydra-core --upgrade

pip install black

pip install mypy

mypy --install-types --non-interactive

pip install pylint
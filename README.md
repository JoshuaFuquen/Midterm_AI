cat > README.md <<'EOF'
# Midterm: MNIST Digit Classification with a CNN

## Problem
Classify handwritten digits (0–9) using a deep learning model.

## Dataset
MNIST from Keras (28x28 grayscale images).

## Model
Small CNN with two convolution blocks and a dense classifier.

## How to run
```bash
source .venv/bin/activate
pip install -r requirements.txt
python train_mnist_cnn.py
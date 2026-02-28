# Midterm: MNIST Digit Classification with a CNN (Deep Learning)

## Group Members
- Joshua Fuquen
- Chris Tham
- Annie Wang

## Files in github
Results(folder): This is where the results are storaged.  
.DS_Store: Libraries.  
.gitignore: Gitignore of .venv folder.  
Midterm_Report_FinalVersion.pdf: Report of the proyect in pdf.  
README.md: Readme to know more about this git repository.  
requirements.txt: Code for the program.  
train_mnist_cnn.py: Program for deep learning proyect.  

## Project Overview
This project trains a **deep learning** model to recognize handwritten digits.  
Given a 28x28 grayscale image, the model predicts a label from **0 to 9**.

We use a **Convolutional Neural Network (CNN)**, a standard deep learning architecture for image data.  
CNNs learn features automatically (strokes, edges, and shapes) using trainable convolution filters, instead of relying on hand-crafted rules.

## Dataset
- **MNIST** (via TensorFlow/Keras)
- Images are normalized to the range **0–1**
- Input shape for the CNN is **28x28x1**

## How to Run (for Grading)

These steps are included so the instructor can reproduce the run and see terminal output.

1) Open a terminal and go to the repository folder
```bash
cd "Your specific folder"
```

2) Activate the virtual environment
```bash
source .venv/bin/activate
```

3) Install dependencies
```bash
pip install -r requirements.txt
```

4) Run the training script (prints epochs + final evaluation in the terminal)
```bash
python train_mnist_cnn.py
```

Optional: run twice to confirm repeatability
```bash
python train_mnist_cnn.py
python train_mnist_cnn.py
```

## Outputs
- Terminal: epoch logs + final test accuracy/loss
- File: `results/metrics.txt` (saved metrics after training)

## Results
After running, the script prints the final metrics and writes them to `results/metrics.txt`.

Example (1st attempt / trial):
- Test accuracy: 0.9909
- Test loss: 0.0312

## Why We Use These Steps
- **Virtual environment** keeps dependencies isolated and avoids version conflicts
- **requirements.txt** makes installs reproducible across machines
- Running the script demonstrates a complete deep learning workflow: preprocessing → training → validation → test evaluation → reporting results

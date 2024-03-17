# MNIST Digit Recognition Model

This repository contains a convolutional neural network (CNN) model implemented in Python using TensorFlow for recognizing handwritten digits from the MNIST dataset.

## Overview

The model built in this project is designed to accurately classify handwritten digits (0-9) from the MNIST dataset.

## Dataset

The MNIST dataset is a benchmark dataset widely used in the field of machine learning and computer vision. It consists of 28x28 pixel grayscale images of handwritten digits, divided into a training set of 60,000 images and a test set of 10,000 images.

## Model Architecture

The model architecture used in this project is as follows:

- Input layer: 28x28 pixels with a single channel
- Convolutional layers with ReLU activation and Batch Normalization
- MaxPooling layers for downsampling
- Fully connected layers with ReLU activation and Batch Normalization
- Output layer with softmax activation for classification

## Training and Evaluation

The model is trained on the training set of the MNIST dataset and evaluated on the test set. During training, the model minimizes the sparse categorical crossentropy loss function using the Adam optimizer. The accuracy of the model is measured on the test set.

## Results

After training the model for 5 epochs with a batch size of 64, the achieved accuracy on the test set is:

- Without Batch Normalization: 99.029%
- With Batch Normalization: 99.279%

## Usage

To use the model:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the Python script `mnist_digit_recognition.py` to train and evaluate the model.
4. Experiment with different architectures, hyperparameters, and optimization techniques to improve performance.

## Files and Directory Structure

- `mnist_digit_recognition.py`: Python script containing the code for building, training, and evaluating the model.
- `requirements.txt`: Text file listing the required Python dependencies.
- `README.md`: This file, providing an overview of the project and instructions for usage.
- `mnist_digit_recognition_model`: Directory containing the saved trained model.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The MNIST dataset is sourced from the National Institute of Standards and Technology (NIST).
- TensorFlow for providing the framework for building and training deep learning models.

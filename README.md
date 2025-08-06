# Deep Belief Networks Visual Concept Learning

## Project Overview

This project investigates how hierarchical neural network models, specifically Deep Belief Networks (DBNs) and Feedforward Neural Networks (FFNNs), learn to represent and classify visual information. The models are trained and evaluated on the Fashion-MNIST dataset.

## Dataset

The Fashion-MNIST dataset is used for this project. It consists of 60,000 training images and 10,000 test images of 10 different clothing categories. The images are grayscale and have a size of 28x28 pixels.

## Models

Two types of neural network models are implemented and compared:

* **Deep Belief Network (DBN):** A generative hierarchical model composed of stacked Restricted Boltzmann Machines (RBMs).
* **Feed-Forward Neural Network (FFNN):** A standard feedforward network with a comparable architecture to the DBN.

## Methodology

1.  **Data Loading and Preprocessing:** The Fashion-MNIST dataset is downloaded, and the pixel values are scaled to a [0, 1] range.
2.  **DBN Training:** A DBN with three hidden layers ([500, 500, 1000] units) is trained on the flattened Fashion-MNIST images.
3.  **FFNN Training:** An FFNN with the same hidden layer sizes is trained for comparison.
4.  **Linear Read-outs:** To evaluate the representations learned by the DBN, linear classifiers are trained on the activations of each hidden layer.
5.  **Receptive Fields Visualization:** The receptive fields of the DBN's hidden layers are visualized to understand the features learned by the model.

## Results

The test accuracies of the models are as follows:

* **FFNN:** 84.05%
* **DBN (Linear Read-outs):**
    * H1 Accuracy: 84.42%
    * H2 Accuracy: 84.68%
    * H3 Accuracy: 84.60%

The receptive fields visualization shows that the DBN learns to extract progressively more complex features in its hidden layers, starting from simple edge detectors and moving towards more "prototype-like" representations of object categories.

## Getting Started

### Prerequisites

You will need to have Python 3 and the following libraries installed:

* matplotlib
* seaborn
* numpy
* torch
* torchvision
* scipy
* scikit-learn

### Installation

1.  Clone the repository:
    ```
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    ```
2.  The DBN library is downloaded from a GitHub repository within the notebook.

### Usage

Open the `dbn_ffnn_fashion_mnist.ipynb` notebook in a Jupyter environment and run the cells sequentially to reproduce the results.
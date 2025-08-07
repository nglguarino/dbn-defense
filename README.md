# Adversarial Robustness and Visual Learning in Deep Belief Networks

This project investigates the **adversarial robustness** and **hierarchical feature learning** capabilities of **Deep Belief Networks (DBNs)**. Using the Fashion-MNIST dataset, we compare the performance and internal representations of a DBN against a standard **Feedforward Neural Network (FFNN)** to understand how these models learn to represent and classify complex visual data.

---

## Project Overview

* **Comparative Analysis:** Direct performance comparison between a generative DBN and a discriminative FFNN.
* **Adversarial Robustness Focus:** The core of the project is analyzing the DBN's resilience to noisy and manipulated inputs.
* **Receptive Field Visualization:** Gain insight into the DBN's learning process through visualizations of its hierarchical features.

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
5.  **Adversarial Attack:** The Fast Gradient Sign Method (FGSM) is used to generate adversarial examples from the test set, creating a new dataset to evaluate model robustness. Both models are then tested on this adversarial dataset.
6.  **Receptive Fields Visualization:** The receptive fields of the DBN's hidden layers are visualized to understand the features learned by the model.


## Results

The test accuracies of the models on both the original and adversarial datasets are as follows:

* **FFNN:**
    * Standard Accuracy: 84.05%
    * Adversarial Accuracy: 16.71%
* **DBN (Linear Read-outs):**
    * H1 Accuracy: 84.42%
    * H2 Accuracy: 84.68%
    * H3 Accuracy: 84.60%
    * Adversarial Accuracy (using H3 features): 26.69%

Under the FGSM attack, the FFNN's performance collapses, showing a significant lack of robustness. The DBN, while also impacted, maintains a higher accuracy, suggesting that its hierarchically learned features provide greater resilience against adversarial perturbations.

The receptive fields visualization shows that the DBN learns to extract progressively more complex features in its hidden layers, starting from simple edge detectors and moving towards more "prototype-like" representations of object categories. This robust feature hierarchy likely contributes to its improved performance on adversarial examples.



## Installation

1.  Clone the repository:
    ```
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    ```
2.  The DBN library is downloaded from a GitHub repository within the notebook.

## Usage

Open the `dbn_ffnn_fashion_mnist.ipynb` notebook in a Jupyter environment and run the cells sequentially to reproduce the results.
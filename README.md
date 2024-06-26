# CNN_final-Project
Explanation:
This Python code sets up a comparison of the LeNet-5 architecture on the MNIST dataset in a Google Colab environment. Here's a breakdown of what each part does:

Imports: Necessary libraries are imported, including PyTorch for deep learning operations, NumPy for numerical operations, and Matplotlib for plotting.

Device: Checks if GPU is available and sets the device accordingly for accelerated computations.

Hyperparameters: Defines parameters such as input dimensions, number of classes, learning rate, epochs, and batch size.

Datasets and Data Loaders: Loads the MNIST dataset, splits it into training and validation sets, and creates data loaders for efficient batch processing.

LeNet-5 Architecture: Defines the LeNet-5 CNN architecture using nn.Module. It consists of convolutional layers (Conv2d), max pooling layers (max_pool2d), and fully connected layers (Linear).

Model Initialization: Initializes the LeNet-5 model, loss function (CrossEntropyLoss), and optimizer (Adam).

Training Loop: Iterates through epochs, performs forward and backward passes through the network using training data, computes losses, and updates model parameters.

Validation Loop: Evaluates the model on the validation set to monitor overfitting.

Test Evaluation: Evaluates the trained model on the test set to measure final performance using accuracy, precision, recall, and F1-score.

Plotting: Visualizes training and validation losses over epochs using Matplotlib.

This setup can be extended to include other CNN architectures (e.g., AlexNet, VGGNet, ResNet) and datasets (e.g., FMNIST, CIFAR-10) by defining their architectures similarly and adjusting dataset loading and preprocessing accordingly.

#Multifunctional NLP and Image Generation Tool using Hugging Face Models

1. Environment Setup
In Google Colab, you have access to GPU acceleration, which is beneficial for running deep learning models efficiently. Ensure that you enable GPU support:

Go to "Runtime" -> "Change runtime type" and select "GPU" as the hardware accelerator.
2. Installing Dependencies
You'll need to install the transformers library from Hugging Face, which provides easy access to a wide range of pre-trained models for NLP and beyond:

3. Importing Libraries
Import the required libraries to interface with the models:

torch: PyTorch library, useful if you need to handle tensors or perform additional computations.
pipeline from transformers: Provides a simple API for using pretrained models and pipelines, such as text generation and summarization.
Implementing Task Functions
Text Summarization
Text summarization is useful for condensing long pieces of text into shorter summaries:

Usage: Pass a piece of text to text_summarization() and it returns a summarized version.
Next Word Prediction
Next word prediction generates text based on a starting prompt, completing it with probable next words:
Usage: Provide a starting sentence to next_word_prediction() and it generates the continuation.


#Sequence-to-Sequence Modeling with Attention Mechanism

Explanation:
Dataset and DataLoader:

SyntheticDataset: Generates a synthetic dataset where each sequence is a random array of floats, and the target sequence is its reverse.
collate_fn: A function to pad sequences in each batch to the maximum sequence length using pad_sequence from PyTorch.
Seq2SeqWithAttention Model:

Encoder: Uses an RNN to process the input sequence (x).
Decoder: Uses an RNNCell with attention mechanism to generate the output sequence.
Attention Mechanism: Computes attention scores and context vector to weigh encoder outputs based on the decoder's current state.
Output Layer: Maps decoder hidden states to the output dimension.
Training Loop:

Iterates through epochs and batches, computes predictions (output) from the model, calculates loss using Mean Squared Error (MSELoss), and updates the model parameters using Adam optimizer (optim.Adam).
Example Usage:

Demonstrates how to use the trained model (model) to predict output sequences given input sequences (test_input).
This setup allows you to train a sequence-to-sequence model with attention on a synthetic dataset and can be extended to real-world applications such as machine translation or text summarization by replacing the synthetic dataset with appropriate data.


END








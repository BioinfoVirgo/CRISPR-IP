# Prediction of CRISPR-Cas9 off-target based on effective use of sequence information
This repository includes a coding scheme and a neural network named CRISPR-IP to predict off-target activities with insertions, deletions, and mismatches in CRISPR/Cas9 gene editing. 

## Prerequisite
Program running environment based on docker image: tensorflow/tensorflow:2.3.2-gpu-jupyter.

Following Python packages should be installed:

pandas == 1.1.5

scikit_learn == 0.24.2

## Usage
1.Encode the gRNA-DNA pairs using the encoding scheme mentioned in our paper.

2.Train the CRISPR-IP model with a training dataset.

3.Use the trained model for prediction and evaluation.

## Example
**example-train-test-split.(ipynb/py)** Randomly divide the training set and the test set. Such as example_saved/example-train-data.csv and example_saved/example-test-data.csv.

> The first column of the output file is the on-target sequence.

> The second column of the output file is the off-target sequence.

> The third column of the output file is the label.

**example-train-CRISPR-IP.(ipynb/py)** Train and save the model based on the training set. Such as: example_saved/example+crispr_ip.h5.

**example-evaluation-CRISPR-IP.(ipynb/py)** Use the trained model to make predictions and save the prediction results.

> The first column of the output file is the on-target sequence.

> The second column of the output file is the off-target sequence.

> The third column of the output file is the actual label.

> The fourth column of the output file is the predicted label.

> The fifth column of the output file is the predicted score.
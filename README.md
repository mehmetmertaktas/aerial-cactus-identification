# Aerial Cactus Identification

This is a learning project where I implement a convolutional neural network using PyTorch. I have used a small sized dataset from Kaggle to be able to run the model on my local machine quickly.
If you would like to just see how the model performs and submit the results to [the competition](https://www.kaggle.com/c/aerial-cactus-identification/), you can just use the saved model by running

`python run_saved_model.py`.

To see the whole training process and visualizations of results, you can run

`python main.py`.

Below is a list of the scripts and their functions:

* `parameters.py`: Model parameters and global variables used in other scripts.
* `dataset.py`: Includes `ACIDataset` class for putting the dataset in PyTorch-readable format.
* `load_data.py`: Places the data in `Dataloader`s&mdash;PyTorch data iterators.
* ``

## Dataset

The dataset used in this project can be found on [Kaggle](https://www.kaggle.com/c/aerial-cactus-identification/data). This dataset contains a large number of 32x32 RGB images of aerial photos of a cactus. There are 17500 images in the training test and 4000 in the test set. Below is a small selection from the dataset.



<p align="center">
  <img width="600" height="120" src="./dataset.png">
</p>

## Model

A simple CNN is used. The model is summarized in the figure below and can also be looked at `model.py`.

* `model.py`: Includes `Net` class.

<p align="center">
  <img width="600" height="300" src="./nn.png">
</p>

## Results

Since the labels of the real test set are unknown, the AUC score can only be seen after submitting the results to Kaggle. To visualize the model's performance, the training set is split into 3: 

* Training (%70)
* Validation (%20)
* Test (%10).

The AUC score on this test set taken from the training set provided by Kaggle is 0.998, see the figure below. After training the model with all of the training set instead of %70 percent of it, the score evaluated on the real test set on Kaggle is 0.995.

<p align="center">
  <img width="480" height="360" src="./auc_graph.png">
</p>


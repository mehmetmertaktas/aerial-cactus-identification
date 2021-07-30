# Aerial Cactus Identification

## Dataset

The dataset can be found on [Kaggle](https://www.kaggle.com/c/aerial-cactus-identification/data). This dataset contains a large number of 32x32 RGB images of aerial photos of a cactus. There are 17500 images in the training test and 4000 in the test set. 

* `dataset.py`: Includes `ACIDataset` class.
* `load_data.py`: Places the data in `Dataloader`s.

## Model

A simple CNN is used. The model is summarized in the figure below.

* `model.py`: Includes `Net` class.

<p align="center">
  <img width="600" height="300" src="./nn.png">
</p>

## Results

Since the labels of the real test set are unknown, the AUC score can only be seen after submitting the results to Kaggle. To visualize the model's performance, the training set is split into 3: 

* Training (%70)
* Validation (%20)
* Test (%10).

The AUC score on this test set taken from the training set provided by Kaggle is 0.998. 

<p align="center">
  <img width="640" height="480" src="./auc_graph.png">
</p>

After training the model with all of the training set instead of %70 percent of it, the score evaluated on the real test set on Kaggle is 0.995.



# Fire-Detection-from-FLAME-Dataset
Deep Learning model implementation for Fire detection both classification and segmentation from the FLAME dataset.

- The dataset is Freely Available [click here](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)


## Introduction
There are two applications that can be defined based on the collected FLAME dataset along with Deep Learning solutions for these problems. The first problem is the fire versus no-fire classification using a deep neural network (DNN) approach. The second problem deals with fire segmentation, which can be used for fire detection by masking the identified fire regions on video frames classified as fire-containing in the first problem.

- Theoritical Description of the model used [click here](Theoritical_Description.md)



## Training the Model

Run the following python code for training the models:

- Classification Training (classification_training.py) [click here](classification_training.py)
- Segmentation Training (segmentation_training.py) [click here](segmentation_training.py)


## Evalutation, Prediction and Training Plots:

After training, run the following code to plot the dataset, evaluate the model, training metrices, model predictions.

- Fire-vs-NoFire Classification (fire_classification.py) [click here](fire_classification.py)
- Fire Segmentation (fire_segmentation.py) [click here](fire_segmentation.py)

#### Supporting Code Description
- Fetching the dataset from the directory (dataset_fetching.py) [click here](dataset_fetching.py)
- Plotting the dataset visualization, training metrices, model evalutation, model predication (data_plotting.py) [click here](data_plotting.py)



## Directory Structure of the Dataset

The downloaded dataset should be put into the folder structure as shown below:

<img width="933" alt="Screenshot 2022-07-11 at 22 56 22" src="https://user-images.githubusercontent.com/47363228/178322597-768483fd-0633-4fe6-b6b7-e22b738c6f2d.png">

## Dataset Visualization

Classification Dataset with Labels:

![fire_classification_data_visualization](https://user-images.githubusercontent.com/47363228/178327387-a1c39093-d52e-4977-94d8-5b4e13181bde.png)

Segmentation Dataset with Mask (ground truth):

![fire_segmentation_data_visualization](https://user-images.githubusercontent.com/47363228/178327708-02edbb70-c990-4b46-9d2c-a8a64a74b239.png)


## Training of the model (Metrices vs Epochs)
- Classification Training:

![fire_classification_training](https://user-images.githubusercontent.com/47363228/178330319-ce947801-4c9c-406c-8ea7-887ceff2a8ca.png)

- Segmentation Training:

![fire_segmentation_training](https://user-images.githubusercontent.com/47363228/178330366-b809a087-1751-42af-9973-534726be2b52.png)

## Model Evaluation and Prediction
- Classification Tasks Confusion Matrix:

![confusion_matrix](https://user-images.githubusercontent.com/47363228/178330549-34371c6a-a510-4cf1-9c25-4911171e5fce.png)

- Segmentation Task Predicted Masks:


![fire_segmentation_prediction1](https://user-images.githubusercontent.com/47363228/178330628-089549c6-eb77-4751-a950-e4eb872a3c4e.png)
![fire_segmentation_prediction](https://user-images.githubusercontent.com/47363228/178330654-4aa1edc8-f785-4db7-8e90-3628a82adab6.png)

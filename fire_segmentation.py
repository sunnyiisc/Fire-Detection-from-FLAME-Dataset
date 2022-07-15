"""
Created on 15 Jun, 2022 at 11:50
    Title: fire_segmentation.py - ...
    Description:
        -   Sementic segmentation of the fires from the image
        -   Model evaluation and prediction
        -   Trained model should be loaded from the .h5 file
@author: Supantha Sen, nrsc, ISRO
"""

# Importing Modules
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf

# Importing Custom Modules
from dataset_fetching import fetch_data_segmentation
from data_plotting import data_visualize_segmentation, training_plot

...
##Main Program
path = './Fire_Segmentation'
val_generator, train_generator = fetch_data_segmentation(path)

#Visualising the Dataset
data_visualize_segmentation(train_generator)

#Load Saved model
model = tf.keras.models.load_model('./fire_segmentation_output/trained_model.h5')

# Plotting the Training Metrices
training_plot('./fire_segmentation_output/trained_model_history.npy')

#Evaluating the model
# score = model.evaluate(val_generator, steps=50)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


# Model Prediction
for i in range(random.randint(1,20)):
    img, msk = next(val_generator)
print(i)

y_pred = model.predict(img, verbose=1)
msk_pred = (y_pred > 0.5).astype(np.uint8)

# Visualizing the Predicted Mask
print('Image Shape:', img.shape)
print('Mask Shape:', msk.shape)

num_img=img.shape[0]//8
for i in range(num_img):
    plt.subplot(3, num_img, i+1)
    plt.imshow(img[i].astype(np.uint8))
    plt.title('Fire Image')
    plt.axis('off')

    plt.subplot(3, num_img, i+1+num_img)
    plt.imshow(msk_pred[i], cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(3, num_img, i+1+(2*num_img))
    plt.imshow(msk[i], cmap='gray')
    plt.title('Actual Mask')
    plt.axis('off')

plt.show()

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
import tensorflow as tf

# Importing Custom Modules
from dataset_fetching import fetch_data_segmentation

...
def data_visualize(data_gen):
    img, msk = next(data_gen)

    print('Image Shape:', img[0].shape)
    print('Mask Shape:', msk[0].shape)

    for i in range(img.shape[0] // 4):
        # plt.subplot(img.shape[0], 2, (2*i)+1)
        plt.subplot(2, img.shape[0] // 4, i + 1)
        plt.imshow(img[i].astype(np.uint8))
        plt.title('Fire Image')
        plt.axis('off')

        # plt.subplot(img.shape[0], 2, (2*i)+2)
        plt.subplot(2, img.shape[0] // 4, i + 1 + (img.shape[0] // 4))
        plt.imshow(msk[i].astype(np.uint8), cmap='gray')
        plt.title('Mask Image')
        plt.axis('off')
    plt.show()


def training_plot(npy_path):
    hist = np.load(npy_path, allow_pickle='TRUE').item()

    print(hist.keys())
    # Plot the model accuracy on training data
    plt.subplot(2, 1, 1)
    plt.plot(hist['accuracy'], '-o')
    plt.plot(hist['val_accuracy'], '-x')
    plt.legend(['Train Accuracy', 'Validation Accuracy'])
    plt.title('Training/Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.grid()
    # Plot the model loss on training data
    plt.subplot(2, 1, 2)
    plt.plot(hist['loss'], '-o')
    plt.plot(hist['val_loss'], '-x')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.title('Training/Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.show()


##Main Program
path = '/home/nrsc/Documents/AI-ML_training_2022-04/Project_SupanthaSen/Fire_Segmentation'
val_generator, train_generator = fetch_data_segmentation(path)

#Visualising the Dataset
data_visualize(train_generator)

#Load Saved model
model = tf.keras.models.load_model('./fire_segmentation_output/trained_model.h5')

# Plotting the Training Metrices
training_plot('./fire_segmentation_output/trained_model_history.npy')

#Evaluating the model
score = model.evaluate(val_generator, steps=50)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Model Prediction
img, msk = next(val_generator)

y_pred = model.predict(img, verbose=1)
msk_pred = (y_pred > 0.5).astype(np.uint8)

# Visualizing the Predicted Mask
print('Image Shape:', img.shape)
print('Mask Shape:', msk.shape)

num_img=img.shape[0]//4
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
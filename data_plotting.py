"""
Created on 20 Jun, 2022 at 15:38
    Title: data_plotting.py - Functions to plot at different steps
    Description:
        -   Plotting the dataset for the classification and segmentation tasks.
        -   Plotting the training metrices such as accuracy and loss
@author: Supantha Sen, nrsc, ISRO
"""

# Importing Modules
import numpy as np
from matplotlib import pyplot as plt

...

# Importing Custom Modules
...

...
def data_visualize_classification(data_gen):
    # Visualising the training set
    img, lbl = next(data_gen)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img[i].astype("uint8"))
        # plt.title(lbl[i])
        plt.title('Fire' if lbl[i] == 0 else 'No Fire')
        plt.axis('off')
    plt.show()


def data_visualize_segmentation(data_gen):
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
    plt.ylabel('Accuracy')
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
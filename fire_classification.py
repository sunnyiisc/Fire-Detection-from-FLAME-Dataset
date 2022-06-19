"""
Created on 15 Jun, 2022 at 10:40
    Title: fire_classification.py - ...
    Description:
        -   Classification between fire and no fire images
@author: Supantha Sen, nrsc, ISRO
"""

# Importing Modules
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Importing Custom Modules
from dataset_fetching import fetch_data_classification

...
def data_visualize(data_gen):
    # Visualising the training set
    img, lbl = next(data_gen)
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img[i].astype("uint8"))
        # plt.title(lbl[i])
        plt.title('Fire' if lbl[i] == 0 else 'No Fire')
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


## Main program
# Fetching the dataset from the directory
path = '/home/nrsc/Documents/AI-ML_training_2022-04/Project_SupanthaSen/Fire_vs_NoFire'
val_generator, train_generator, test_generator = fetch_data_classification(path)

#Visualising the Dataset
data_visualize(train_generator)

#Loading the model
model = tf.keras.models.load_model('./fire_classification_output/trained_model.h5')

# Plotting the Training Metrices
training_plot('./fire_classification_output/trained_model_history.npy')

#Evaluating the model
score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Model Prediction
Y_pred = model.predict(test_generator, verbose=1)
y_pred = np.where(Y_pred>0.5, 1, 0).reshape(1,-1)[0].astype('float32')

#True Value from test dataset
y_true = test_generator.classes

print(y_pred.shape)
print(y_true.shape)

#Prediction accuracy
num_correct = np.sum(y_true == y_pred)
print('Prediction Accuracy =', num_correct/len(y_true))

#Confution Matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices)
disp.plot(cmap = 'Greens')
plt.show()

#Print Classification Report
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices))
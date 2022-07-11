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
from data_plotting import data_visualize_classification, training_plot

...
## Main program
# Fetching the dataset from the directory
path = '/home/nrsc/Documents/AI-ML_training_2022-04/Project_SupanthaSen/Fire_vs_NoFire'
val_generator, train_generator, test_generator = fetch_data_classification(path)

#Visualising the Dataset
#1data_visualize_classification(train_generator)

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
##y_true = test_generator.classes       #For Flow from directory

##True Value from test dataset      #For Image dataset from directory
y_true=[]
for images, labels in test_generator:
    for i in range(len(labels)):
        y_true.append(labels[i])
y_true = np.array(y_true).reshape(1,-1)[0].astype('float32')

print(y_pred.shape)
print(y_true.shape)

#Prediction accuracy
num_correct = np.sum(y_true == y_pred)
print('Prediction Accuracy =', num_correct/len(y_true))

#Confution Matrix
print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_names)
disp.plot(cmap = 'Greens')
plt.show()

#Print Classification Report
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=test_generator.class_names))
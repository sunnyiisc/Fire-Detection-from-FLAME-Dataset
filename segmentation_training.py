"""
Created on 16 Jun, 2022 at 16:42
    Title: segmentation_training.py - Model training for Fire Segmentation
    Description:
        -   Training the model for fire segmentation task
@author: Supantha Sen, nrsc, ISRO
"""

# Importing Modules
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

# Importing Custom Modules
from model_architectures import unet_model
from dataset_fetching import fetch_data_segmentation

...

def train_model(val_generator, train_generator, batchsize):
    model = unet_model()
    model.summary()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Visualising model architecture
    tf.keras.utils.plot_model(model, to_file='./fire_segmentation_output/unet_model.png', show_shapes=True)
    # display(Image.open('unet_model.png'))

    # Saving Model Checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./fire_segmentation_output/saved_weights_epoch{epoch:02d}.h5',
                                                    monitor='accuracy',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_wrights_only=True,
                                                    mode='auto',
                                                    save_freq='epoch')

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                 min_delta=0.005,
                                                 patience=3,
                                                 verbose=1,
                                                 mode='auto',
                                                 baseline=None,
                                                 restore_best_weights=True)

    # Fitting the model
    hist = model.fit(train_generator,
                     epochs=30,
                     batch_size=batchsize,
                     validation_data=val_generator,
                     verbose=1,
                     steps_per_epoch=(1603//batchsize),
                     validation_steps=(400//batchsize),
                     callbacks=[earlystop, checkpoint],
                     use_multiprocessing=True)



    # Saving the trained model
    np.save('./fire_segmentation_output/trained_model_history.npy', hist.history)
    model.save('./fire_segmentation_output/trained_model.h5')

    return model


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

# Training the model
batchsize = 128
model = train_model(val_generator, train_generator, batchsize)

# Plotting the Training Metrices
training_plot('./fire_segmentation_output/trained_model_history.npy')
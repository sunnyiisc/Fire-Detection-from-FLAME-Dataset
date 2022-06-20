"""
Created on 17 Jun, 2022 at 13:11
    Title: dataset_fetching.py - ...
    Description:
        -   Importing the data from the directory using data generator
@author: Supantha Sen, nrsc, ISRO
"""

# Importing Modules
import tensorflow as tf

# Importing Custom Modules
...

...
def fetch_data_classification(path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    print('Validation Data:')

    val_generator = datagen.flow_from_directory(path + '/Training',
                                                target_size=(224, 224),
                                                color_mode='rgb',
                                                class_mode='binary',
                                                seed=100,
                                                shuffle=True,
                                                subset='validation')

    print('Training Data:')

    train_generator = datagen.flow_from_directory(path + '/Training',
                                                  target_size=(224, 224),
                                                  color_mode='rgb',
                                                  class_mode='binary',
                                                  seed=100,
                                                  shuffle=True,
                                                  subset='training')

    print('Test Data:')

    test_generator = test_datagen.flow_from_directory(path + '/Test',
                                                 target_size=(224, 224),
                                                 color_mode='rgb',
                                                 class_mode='binary',
                                                 seed=100,
                                                 shuffle=False)

    return (val_generator, train_generator, test_generator)


def fetch_data_segmentation(path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

    print('Validation Datset:')
    image_val_generator = datagen.flow_from_directory(path + '/Images_Dataset',
                                                      target_size=(512, 512),
                                                      color_mode='rgb',
                                                      class_mode=None,
                                                      seed=100,
                                                      shuffle=False,
                                                      subset='validation')
    mask_val_generator = datagen.flow_from_directory(path + '/Masks_Dataset',
                                                     target_size=(512, 512),
                                                     color_mode='grayscale',
                                                     class_mode=None,
                                                     seed=100,
                                                     shuffle=False,
                                                     subset='validation')

    val_generator = zip(image_val_generator, mask_val_generator)

    print('Training Datset:')
    image_train_generator = datagen.flow_from_directory(path + '/Images_Dataset',
                                                        target_size=(512, 512),
                                                        color_mode='rgb',
                                                        class_mode=None,
                                                        seed=100,
                                                        subset='training')
    mask_train_generator = datagen.flow_from_directory(path + '/Masks_Dataset',
                                                       target_size=(512, 512),
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       seed=100,
                                                       subset='training')

    train_generator = zip(image_train_generator, mask_train_generator)

    return (val_generator, train_generator)
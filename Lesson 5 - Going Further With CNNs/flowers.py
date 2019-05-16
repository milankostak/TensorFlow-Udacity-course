# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb

import os  # to read files and directory structure
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import *

tf.logging.set_verbosity(tf.logging.ERROR)

##############################
# 1. prepare data
##############################

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# divide images to training and validation sets
for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} images".format(cl, len(images)))
    train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')

##############################
# 2. augment data
##############################

BATCH_SIZE = 100
IMG_SHAPE = 150


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


train_image_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                              class_mode='binary')

##############################
# 3. train model
##############################

model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation=tf.nn.relu),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation=tf.nn.relu),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation=tf.nn.relu),
    MaxPooling2D(2, 2),

    Dropout(0.2),
    Flatten(),
    Dense(512, activation=tf.nn.relu),
    Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
#
# history = model.fit_generator(
#     train_data_gen,
#     steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
#     epochs=5,
#     validation_data=val_data_gen,
#     validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
# )

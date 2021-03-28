# exercise
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c03_exercise_flowers_with_data_augmentation.ipynb
# solution
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb

import os
import matplotlib.pyplot as plt
import numpy as np
import glob

import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import *


##############################
# 1. prepare data
##############################

URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(
    origin=URL,
    fname="flower_photos.tgz",
    extract=True
)

base_dir = os.path.join(os.path.dirname(zip_file), "flower_photos")

classes = ["roses", "daisy", "dandelion", "sunflowers", "tulips"]

# divide images to training and validation sets (run only once)
# import shutil
# for cl in classes:
#     img_path = os.path.join(base_dir, cl)
#     images = glob.glob(img_path + '/*.jpg')
#     print("{}: {} images".format(cl, len(images)))
#     train, val = images[:round(len(images) * 0.8)], images[round(len(images) * 0.8):]
#
#     for t in train:
#         if not os.path.exists(os.path.join(base_dir, 'train', cl)):
#             os.makedirs(os.path.join(base_dir, 'train', cl))
#         shutil.move(t, os.path.join(base_dir, 'train', cl))
#
#     for v in val:
#         if not os.path.exists(os.path.join(base_dir, 'val', cl)):
#             os.makedirs(os.path.join(base_dir, 'val', cl))
#         shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, "train")
total_train = 0
for cl in classes:
    sub_dir = os.path.join(train_dir, cl)
    train_images = glob.glob(sub_dir + "/*.jpg")
    total_train += len(train_images)

validation_dir = os.path.join(base_dir, "val")
total_validation = 0
for cl in classes:
    sub_dir = os.path.join(validation_dir, cl)
    validation_images = glob.glob(sub_dir + "/*.jpg")
    total_validation += len(validation_images)

##############################
# 2. augment data
##############################

def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


BATCH_SIZE = 100
IMG_SHAPE = 150

train_image_generator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='binary'
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plot_images(augmented_images)

validation_image_generator = ImageDataGenerator(rescale=1. / 255)
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=validation_dir,
    shuffle=False,
    target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
    class_mode='binary'
)

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

    Flatten(),
    Dropout(0.2),
    Dense(512, activation=tf.nn.relu),

    Dropout(0.2),
    Dense(2, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 5
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_validation / float(BATCH_SIZE)))
)

##############################
# 4. visualize results
##############################

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
# plt.savefig("./foo.png")
plt.show()

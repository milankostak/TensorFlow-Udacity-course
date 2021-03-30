# without augmentation
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c01_dogs_vs_cats_without_augmentation.ipynb
# with augmentation
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c02_dogs_vs_cats_with_augmentation.ipynb

# conda install pillow

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

##############################
# 1. prepare data
##############################

URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_dir = tf.keras.utils.get_file(
    fname="cats_and_dogs_filterted.zip",
    origin=URL,
    extract=True
)

# prepare folders and data count
base_dir = os.path.join(os.path.dirname(zip_dir), "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_cats_dir = os.path.join(train_dir, "cats")  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, "dogs")  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, "cats")  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, "dogs")  # directory with our validation dog pictures

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("Total training cat images:", num_cats_tr)
print("Total training dog images:", num_dogs_tr)

print("Total validation cat images:", num_cats_val)
print("Total validation dog images:", num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


##############################
# 2. augment data
##############################


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# Number of training examples to process before updating our models variables
BATCH_SIZE = 100
# Our training data consists of images with width of 150 pixels and height of 150 pixels
IMG_SHAPE = 150

# 1. Read images from the disk
# 2. Decode contents of these images and convert it into proper grid format as per their RGB content
# 3. Convert them into floating point tensors
# 4. Rescale the tensors from values between 0 and 255 to values between 0 and 1,
#    as neural networks prefer to deal with small input values.

# Generator for our training data
# train_image_generator = ImageDataGenerator(rescale=1. / 255)
# train_data_gen = train_image_generator.flow_from_directory(
#     batch_size=BATCH_SIZE,
#     directory=train_dir,
#     shuffle=True,
#     target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
#     class_mode='binary'
# )


# sample_training_images, _ = next(train_data_gen)
# plot_images(sample_training_images[:5])  # Plot images 0-4

# randomly applying horizontal flip augmentation
# image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

# the rotation augmentation will randomly rotate the image up to a specified number of degrees
# image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)

# apply Zoom augmentation to our dataset, zooming images up to 50% randomly.
# image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)

# train_data_gen = image_gen.flow_from_directory(
#     batch_size=BATCH_SIZE,
#     directory=train_dir,
#     shuffle=True,
#     target_size=(IMG_SHAPE, IMG_SHAPE)
# )

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

# Generator for our validation data
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

    Conv2D(128, (3, 3), activation=tf.nn.relu),
    MaxPooling2D(2, 2),

    Dropout(0.5),
    Flatten(),
    Dense(512, activation=tf.nn.relu),
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
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

##############################
# 4. visualize results
##############################

# visualize convolutional kernels
print(model.layers[0].get_weights()[0])
weights = model.layers[0].get_weights()[0][:, :, 0, :]

for i in range(0, 16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(weights[:, :, i], interpolation="nearest", cmap="gray")
plt.show()
print(weights.shape)

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


##############################
# 5. experimenting
##############################


def preprocess_image(img):
    img = Image.open(img)
    # img.show()

    new_size = (IMG_SHAPE, IMG_SHAPE)
    img = img.resize(new_size)
    img.show()

    img_array = np.array(img)
    # print(img_array.shape)
    img_array = img_array.astype(np.float)
    img_array /= 255
    img_array = img_array.reshape((1, IMG_SHAPE, IMG_SHAPE, 3))  # 3 -> color RGB image
    # print(img_array.shape)
    return img_array


def predict_image(img_array):
    result = model.predict(img_array)
    print(result)
    return result


dog_url = "https://upload.wikimedia.org/wikipedia/commons/9/99/Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg"
dog = tf.keras.utils.get_file("dog.jpg", origin=dog_url)
predict_image(preprocess_image(dog))

cat_url = "https://upload.wikimedia.org/wikipedia/commons/9/9b/Domestic_cat_cropped.jpg"
cat = tf.keras.utils.get_file("cat3.jpg", origin=cat_url)
predict_image(preprocess_image(cat))

elephant_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/African_Bush_Elephant.jpg/1200px-African_Bush_Elephant.jpg"
elephant = tf.keras.utils.get_file("elephant.jpg", origin=elephant_url)
predict_image(preprocess_image(elephant))

# later, try other images!


# visualize image processed by the first layer of convolutional kernels
# redefine the model to output after the first layer
model2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
# model2.summary()

dog_array = preprocess_image(dog)
feature_maps = model2.predict(dog_array)

plt.figure(figsize=(15, 15))
square = 4
for i in range(square):
    for j in range(square):
        index = i * square + j
        plt.subplot(square, square, index + 1)
        plt.imshow(feature_maps[0, :, :, index], cmap="gray")

plt.show()

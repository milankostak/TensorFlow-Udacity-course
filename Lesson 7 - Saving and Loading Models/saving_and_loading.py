# exercise
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l07c01_saving_and_loading_models.ipynb

import time
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

tfds.disable_progress_bar()

#
# Part 1: Load the Cats vs. Dogs Dataset
#

splits = tfds.Split.ALL.subsplit(weighted=(80, 20))

splits, info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True, split=splits)

(train_examples, validation_examples) = splits


def format_image(image, label):
    # `hub` image modules expect their data normalized to the [0,1] range.
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


num_examples = info.splits["train"].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224

train_batches = train_examples.cache().shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

#
# Part 2: Transfer Learning with TensorFlow Hub
#

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(2, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS = 3
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

#
# Check Predictions
#

class_names = np.array(info.features["label"].names)
print("Class names: ", class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
print("Predicted class names: ", predicted_class_names)

print("Labels:           ", label_batch)
print("Predicted labels: ", predicted_ids)

plt.figure(figsize=(10, 9))
for n in range(30):
    plt.subplot(6, 5, n + 1)
    plt.imshow(image_batch[n])
    color = "blue" if predicted_ids[n] == label_batch[n] else "red"
    plt.title(predicted_class_names[n].title(), color=color)
    plt.axis("off")
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.show()

#
# Part 3: Save as Keras .h5 model
#

t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)

#
# Part 4: Load the Keras .h5 Model
#

reloaded = tf.keras.models.load_model(
    export_path_keras,
    # `custom_objects` tells keras how to load a `hub.KerasLayer`
    custom_objects={"KerasLayer": hub.KerasLayer}
)

reloaded.summary()

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

print(abs(result_batch - reloaded_result_batch).max())

#
# Keep Training
#

EPOCHS = 3
history = reloaded.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

#
# Part 5: Export as SavedModel
#

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)

#
# Part 6: Load SavedModel
#

reloaded_sm = tf.saved_model.load(export_path_sm)
reload_sm_result_batch = reloaded_sm(image_batch, training=False).numpy()
print(abs(result_batch - reload_sm_result_batch).max())

#
# Part 7: Loading the SavedModel as a Keras Model
#

t = time.time()

export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)

reload_sm_keras = tf.keras.models.load_model(
    export_path_sm,
    custom_objects={"KerasLayer": hub.KerasLayer}
)

reload_sm_keras.summary()

result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)

print(abs(result_batch - reload_sm_keras_result_batch).max())

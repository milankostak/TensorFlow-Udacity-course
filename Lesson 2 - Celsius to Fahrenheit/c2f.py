# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# generate train data
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# the right output for the train data
# (usually you will not calculate those in runtime, but they will be provided in the dataset)
fahrenheit = [t * 1.8 + 32 for t in celsius]
fahrenheit = np.array(fahrenheit, dtype=float)
print(fahrenheit)
# -40, 14, 32, 46, 59, 72, 100

for i, c in enumerate(celsius):
    print(f"{c} °C = {fahrenheit[i]} °F")

# the most basic NN - 1 layer with 1 neuron
# later experiment with more layers and more neurons
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
#     tf.keras.layers.Dense(units=2, input_shape=[1]),
#     tf.keras.layers.Dense(units=1, input_shape=[2])
# ])

# "loss" is the loss function
# optimizer will be explained some other time
# later experiment with MeanSquaredError loss - it usually helps to achieve better results for this problem
model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(0.1)
)

# you can also use string values which will be converted to the corresponding objects with default values
# model.compile(
#     loss="mean_absolute_error",
#     optimizer="adam"
# )

model.summary()

history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Training finished")

plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history["loss"])
plt.show()

value = 22
print("Predicted value:", model.predict([value]))
print("Correct value:", value * 1.8 + 32)  # 71.6

print("These are the l0 variables:", l0.get_weights())
# print("These are the l0 variables:", model.layers[0].get_weights())

celsius_test = np.array([-45, -20, 5, 25, 40, 60])  # only values that were not in the train set
fahrenheit_true = np.array([t * 1.8 + 32 for t in celsius_test])
fahrenheit_predicted = np.array([])
error = []
# print(fahrenheit_true)

for i, c in enumerate(celsius_test):
    predicted = model.predict([np.array([c], dtype=float)])
    error.append(abs(predicted - fahrenheit_true[i]))
    fahrenheit_predicted = np.append(fahrenheit_predicted, predicted)

print("Correct values:", fahrenheit_true)
print("Predicted values:", fahrenheit_predicted)
print()
print("Absolute error values:", error)
print("MAE:", np.average(error))

mae = tf.keras.losses.MeanAbsoluteError()
mae_val = mae(fahrenheit_true, fahrenheit_predicted).numpy()
print("MAE:", mae_val)

mse = tf.keras.losses.MeanSquaredError()
mse_val = mse(fahrenheit_true, fahrenheit_predicted).numpy()
print("MSE:", mse_val)

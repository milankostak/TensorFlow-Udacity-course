# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf.logging.set_verbosity(tf.logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# for i, c in enumerate(celsius_q):
#     print("{} °C = {} °F".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
#     tf.keras.layers.Dense(units=2, input_shape=[1]),
#     tf.keras.layers.Dense(units=1, input_shape=[2])
# ])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

print(model.predict(np.array([22])))  # 71.6

print("These are the l0 variables: {}".format(l0.get_weights()))

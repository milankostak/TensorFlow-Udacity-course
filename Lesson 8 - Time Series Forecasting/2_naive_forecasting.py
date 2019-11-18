# exercise
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c02_naive_forecasting.ipynb

import numpy as np
import matplotlib.pyplot as plt


def plot_series(time_p, series, format="-", start=0, end=None, label=None):
    plt.plot(time_p[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time_p, slope_p=0.0):
    return slope_p * time_p


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time_p, period, amplitude_p=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time_p + phase) % period) / period
    return amplitude_p * seasonal_pattern(season_time)


def white_noise(time_p, noise_level_p=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time_p)) * noise_level_p


#
# Trend and Seasonality
#

time = np.arange(4 * 365 + 1)

slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude_p=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

#
# Naive Forecast
#

naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, naive_forecast, label="Forecast")

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150, label="Series")
plot_series(time_valid, naive_forecast, start=1, end=151, label="Forecast")

errors = naive_forecast - x_valid
abs_errors = np.abs(errors)
mae = abs_errors.mean()
print(mae)

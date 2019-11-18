# exercise
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c01_common_patterns.ipynb

import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series_p, format_p="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series_p[start:end], format_p, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


#
# Trend and Seasonality
#

def trend(time, slope_p=0.0):
    return slope_p * time


times = np.arange(4 * 365 + 1)
baseline = 10
series = baseline + trend(times, 0.1)

plt.figure(figsize=(10, 6))
plot_series(times, series)
plt.show()

print(times)

print(series)


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude_p=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude_p * seasonal_pattern(season_time)


amplitude = 40
series = seasonality(times, period=365, amplitude_p=amplitude)

plt.figure(figsize=(10, 6))
plot_series(times, series)
plt.show()

slope = 0.05
series = baseline + trend(times, slope) + seasonality(times, period=365, amplitude_p=amplitude)

plt.figure(figsize=(10, 6))
plot_series(times, series)
plt.show()


#
# NOISE
#

def white_noise(time, noise_level_p=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level_p


noise_level = 5
noise = white_noise(times, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(times, noise)
plt.show()

series += noise

plt.figure(figsize=(10, 6))
plot_series(times, series)
plt.show()

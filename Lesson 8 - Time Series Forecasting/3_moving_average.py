# exercise
# https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l08c03_moving_average.ipynb

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_series(time_p, series_p, format_p="-", start=0, end=None, label=None):
    plt.plot(time_p[start:end], series_p[start:end], format_p, label=label)
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

#
# Naive Forecast
#

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150, label="Series")
plot_series(time_valid, naive_forecast, start=1, end=151, label="Forecast")

mae = tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy()
print("Naive Forecast")
print(mae)


#
# Moving Average
#

def moving_average_forecast(series_p, window_size):
    """Forecasts the mean of the last few values.
       If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time_p in range(len(series_p) - window_size):
        forecast.append(series_p[time_p:time_p + window_size].mean())
    return np.array(forecast)


# def moving_average_forecast(series, window_size):
#     """Forecasts the mean of the last few values.
#        If window_size=1, then this is equivalent to naive forecast
#        This implementation is *much* faster than the previous one"""
#     mov = np.cumsum(series)
#     mov[window_size:] = mov[window_size:] - mov[:-window_size]
#     return mov[window_size - 1:-1] / window_size


moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, moving_avg, label="Moving average (30 days)")

mae = tf.keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy()
print("Moving average (30 days)")
print(mae)

diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series, label="Series(t) – Series(t–365)")
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) – Series(t–365)")
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:], label="Series(t) – Series(t–365)")
plot_series(time_valid, diff_moving_avg, label="Moving Average of Diff")
plt.show()

diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_avg_plus_past, label="Forecasts")
plt.show()

mae = tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy()
print("Moving average with trend and seasonality")
print(mae)

diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-359], 11) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, label="Series")
plot_series(time_valid, diff_moving_avg_plus_smooth_past, label="Forecasts")
plt.show()

mae = tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()
print("Moving average with trend and seasonality 2")
print(mae)

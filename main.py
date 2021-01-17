from compute_func import set_equal_velocity, import_data, regres, all_vel_indicies, find_vel_borders, wavelets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate

from graph_func import filter_acceleration, calculate_position, calculate_velocity, zero_velocity, create_graphs

sns.set_theme()
data = import_data("acce2.csv")
print(data.head())
filter_acceleration(data, div_freq=300)

# x_wave=wavelets(data = data["X"], wavelet = 'haar', uselevels = 6, mode = 'zero')
# y_wave=wavelets(data = data["Y"], wavelet = 'haar', uselevels = 6, mode = 'zero')
# z_wave=wavelets(data = data["Z"], wavelet = 'haar', uselevels = 6, mode = 'zero')
# data["X"] = x_wave[:len(data["X"])]
# data["Y"] = y_wave[:len(data["Y"])]
# data["Z"] = z_wave[:len(data["Z"])]

# plt.figure()
# plt.title("wavelets")
# sns.lineplot(y=data["X"], x=range(len(data["X"])))
# sns.lineplot(y=x_wave,  x=range(len(x_wave)))

calculate_velocity(data)
print(len(data))

data["X_velocity"]= signal.detrend(data["X_velocity"])
data["Y_velocity"] = signal.detrend(data["Y_velocity"])
data["Z_velocity"] = signal.detrend(data["Z_velocity"])

# data["X_velocity"] = regres(data["X_velocity"], data["TIME"], 100, len(data)-1)
# data["Y_velocity"] = regres(data["Y_velocity"], data["TIME"], 100, len(data)-1)
# data["Z_velocity"] = regres(data["Z_velocity"], data["TIME"], 100, len(data)-1)

# zero_velocity(data)
# x, y, z = all_vel_indicies(data)

# x0, xf = find_vel_borders(x)
# y0, yf = find_vel_borders(y)
# z0, zf = find_vel_borders(z)

# set_equal_velocity(data, x0, xf, "X_velocity")
# set_equal_velocity(data, y0, yf, "Y_velocity")
# set_equal_velocity(data, z0, zf, "Z_velocity")


calculate_position(data)

create_graphs(["raw", "raw_vs_filter", "position", "velocity"], data)
# create_graphs([ "position","velocity"], data)

plt.figure()
plt.plot(data["X_position"],data["Y_position"])

plt.show()

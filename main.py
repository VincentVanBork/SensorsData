from compute_func import set_equal_velocity, import_data, regres, all_vel_indicies, find_vel_borders, wavelets, PolyArea
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate
from scipy.spatial.transform import Rotation as R

from graph_func import filter_acceleration, calculate_position, calculate_velocity, zero_velocity, create_graphs
from gyro_sensor import gyro1

sns.set_theme()
data = import_data("acce20.csv")
# print(data.head())
# filter_acceleration(data, div_freq=20)

#wave becomes more "square" but this doesnt change outcome of velocity.

# x_wave=wavelets(data = data["X"], wavelet = 'haar', uselevels = 3, mode = 'zero')
# y_wave=wavelets(data = data["Y"], wavelet = 'haar', uselevels = 3, mode = 'zero')
# z_wave=wavelets(data = data["Z"], wavelet = 'haar', uselevels = 3, mode = 'zero')
# data["X"] = x_wave[:len(data["X"])]
# data["Y"] = y_wave[:len(data["Y"])]
# data["Z"] = z_wave[:len(data["Z"])]

# plt.figure()
# plt.title("wavelets")
# sns.lineplot(y=data["X"], x=range(len(data["X"])))
# sns.lineplot(y=x_wave,  x=range(len(x_wave)))

#rotate vector by some angle described by quaternion
acce_vectors = data[["X","Y","Z"]].to_numpy()
rotation_vectors = gyro1[["X","Y","Z","ANGLE"]].to_numpy()
rotation = R.from_quat(rotation_vectors)
rotated_vectors = rotation.apply(acce_vectors)
rotated_acce = pd.DataFrame(rotated_vectors, columns=["X", "Y", "Z"])

plt.figure()
sns.lineplot(y=data["Y"], x=data["TIME"], label="Y_NORMAL")
sns.lineplot(y=rotated_acce["Y"], x=data["TIME"], label="Y_ROTATED")

data["X"] = rotated_acce["X"]
data["Y"] = rotated_acce["Y"]
data["Z"] = rotated_acce["Z"]

# filter_acceleration(data, div_freq=30)


calculate_velocity(data)


#we dont use regres cause we have to pick range here 100 , until end but can be 200 400 or sth
# data["X_velocity"] = regres(data["X_velocity"], data["TIME"], 100, len(data)-1)
# data["Y_velocity"] = regres(data["Y_velocity"], data["TIME"], 100, len(data)-1)
# data["Z_velocity"] = regres(data["Z_velocity"], data["TIME"], 100, len(data)-1)

# needs repeating motion to function properly and hard to calculate ranges when values differ much - are negative or positive closely
# zero_velocity(data)
# x, y, z = all_vel_indicies(data)

# x0, xf = find_vel_borders(x)
# y0, yf = find_vel_borders(y)
# z0, zf = find_vel_borders(z)

# set_equal_velocity(data, x0, xf, "X_velocity")
# set_equal_velocity(data, y0, yf, "Y_velocity")
# set_equal_velocity(data, z0, zf, "Z_velocity")

data = data.dropna()

# data["X_velocity"]= signal.detrend(data["X_velocity"])
# data["Y_velocity"] = signal.detrend(data["Y_velocity"])
# data["Z_velocity"] = signal.detrend(data["Z_velocity"])


calculate_position(data)
print("ARE OF THE MODEL", np.pi * 37 * 37)
print(PolyArea(data["X_position"]*120, data["Y_position"]*120))

create_graphs(["raw", "raw_vs_filter", "position", "velocity"], data)
# create_graphs([ "position","velocity"], data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data["X_position"],data["Y_position"], data["Z_position"])

fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(data["X_position"],data["Y_position"])

plt.show()



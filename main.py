from compute_func import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d
from graph_func import *

sns.set_theme()
data = import_data("acce1.csv")


filter_acceleration(data, div_freq=1400)

calculate_velocity(data)

# zero_velocity(data)
# x,y,z = all_vel_indicies(data)

# x0, xf = find_vel_borders(x)
# y0, yf = find_vel_borders(y)
# z0, zf = find_vel_borders(z)

# set_equal_velocity(data, x0, xf, "X_velocity")
# set_equal_velocity(data, y0, yf, "Y_velocity")
# set_equal_velocity(data, z0, zf, "Z_velocity")

calculate_position(data)

fx = interp1d(data["TIME"], data["X_velocity"], kind='cubic', fill_value="extrapolate")
fy = interp1d(data["TIME"], data["Y_velocity"], kind='cubic')
fz = interp1d(data["TIME"], data["Z_velocity"], kind='cubic')

plt.figure()
plt.title("interpolated X,Y,Z")
sns.lineplot(y=fx(data["TIME"]), x=data["TIME"], label="X")
# sns.lineplot(y=fy(data["TIME"]), x=data["TIME"], label="Y")
# sns.lineplot(y=fz(data["TIME"]), x=data["TIME"], label="Z")


create_graphs([""], data)

plt.show()
from compute_func import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate

from graph_func import *

sns.set_theme()
data = import_data("acce1.csv")


filter_acceleration(data, div_freq=450)

calculate_velocity(data)

zero_velocity(data)
x,y,z = all_vel_indicies(data)

x0, xf = find_vel_borders(x)
y0, yf = find_vel_borders(y)
z0, zf = find_vel_borders(z)

set_equal_velocity(data, x0, xf, "X_velocity")
set_equal_velocity(data, y0, yf, "Y_velocity")
set_equal_velocity(data, z0, zf, "Z_velocity")

calculate_position(data)

create_graphs(["raw_vs_filter", "position", "velocity"], data)

plt.show()
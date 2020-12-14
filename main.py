from compute_func import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate


sns.set_theme()
data = import_data("acce1.csv")

plt.figure()
plt.title("RAW DATA X,Y,Z")
sns.lineplot(y=data["X"], x=data["TIME"], label="X")
sns.lineplot(y=data["Y"], x=data["TIME"], label="Y")
sns.lineplot(y=data["Z"], x=data["TIME"], label="Z")

filter_acceleration(data)

calculate_velocity(data)

plt.figure()
plt.title("FILTERED ACCELERATION X,Y,Z")
sns.lineplot(y=data["X_filter"], x=data["TIME"], label="X")
sns.lineplot(y=data["Y_filter"], x=data["TIME"], label="Y")
sns.lineplot(y=data["Z_filter"], x=data["TIME"], label="Z")


calculate_position(data)

plt.figure()
plt.title("VELOCITY X,Y,Z")
sns.lineplot(y=data["X_velocity"], x=data["TIME"], label="X")
sns.lineplot(y=data["Y_velocity"], x=data["TIME"], label="Y")
sns.lineplot(y=data["Z_velocity"], x=data["TIME"], label="Z")

plt.figure()
plt.title("position X,Y,Z")
sns.lineplot(y=data["X_position"], x=data["TIME"], label="X")
sns.lineplot(y=data["Y_position"], x=data["TIME"], label="Y")
sns.lineplot(y=data["Z_position"], x=data["TIME"], label="Z")


plt.show()
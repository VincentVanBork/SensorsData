from os import name
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

sns.set_theme()
#x,y,z vectors
colnames=[ 'X', 'Y', 'Z',"ANGLE","USELESS", "ACCURACY", "TIMESTAMP"]

gyro1 = pd.read_csv("./data/gyro1.csv", names=colnames)
gyro1["TIME"] = (gyro1["TIMESTAMP"] - gyro1["TIMESTAMP"].iloc[0])/1000000000
plt.figure()
sns.lineplot(data=gyro1, x=gyro1["TIME"], y="X")
sns.lineplot(data=gyro1, x=gyro1["TIME"], y="Y")
sns.lineplot(data=gyro1, x=gyro1["TIME"], y="Z")
gyro1["SAMPLING_TIME"] = gyro1["TIME"].diff()

sampling_freq = 1 / gyro1["SAMPLING_TIME"].mean()

order=5

cutoff_freq = sampling_freq / 1400

number_of_samples = len(gyro1)

normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
#prepare filter
numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq,btype="highpass")

filtered_signal_X = signal.lfilter(numerator_coeffs, denominator_coeffs, gyro1["X"])
filtered_signal_Y = signal.lfilter(numerator_coeffs, denominator_coeffs, gyro1["Y"])
filtered_signal_Z = signal.lfilter(numerator_coeffs, denominator_coeffs, gyro1["Z"])

plt.figure()
sns.lineplot(data=gyro1, x=gyro1["TIME"], y=filtered_signal_X)
sns.lineplot(data=gyro1, x=gyro1["TIME"], y=filtered_signal_Y)
sns.lineplot(data=gyro1, x=gyro1["TIME"], y=filtered_signal_Z)

plt.show()

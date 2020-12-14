from typing import final
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate

sns.set_theme()

colnames=[ 'X', 'Y', 'Z', "ACCURACY", "TIMESTAMP"]
acce1 = pd.read_csv("./data/acce1.csv", names=colnames)
acce1["TIME"] = (acce1["TIMESTAMP"] - acce1["TIMESTAMP"].iloc[0])/1000000000
# print(acce1.head())
plt.figure()
sns.lineplot(
    data=acce1,
    x="TIME", y="X")
sns.lineplot(
    data=acce1,
    x="TIME", y="Y")
sns.lineplot(
    data=acce1,
    x="TIME", y="Z")

acce1["SAMPLING_TIME"] = acce1["TIME"].diff()
# print(acce1.head())

sampling_freq = 1 / acce1["SAMPLING_TIME"].mean()

order=5

cutoff_freq = sampling_freq / 1400

number_of_samples = len(acce1)

time = np.linspace(0, acce1["SAMPLING_TIME"].mean(), number_of_samples, endpoint=False)

normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
#prepare filter
numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

filtered_signal_X = signal.lfilter(numerator_coeffs, denominator_coeffs, acce1["X"])
filtered_signal_Y = signal.lfilter(numerator_coeffs, denominator_coeffs, acce1["Y"])
filtered_signal_Z = signal.lfilter(numerator_coeffs, denominator_coeffs, acce1["Z"])

acce1["X_filter"] = filtered_signal_X
acce1["Y_filter"] = filtered_signal_Y
acce1["Z_filter"] = filtered_signal_Z

acce1["X_filter_abs"] = acce1["X_filter"].abs()
acce1["Y_filter_abs"] = acce1["Y_filter"].abs()
acce1["Z_filter_abs"] = acce1["Z_filter"].abs()
high_numerator_coeffs, high_denominator_coeffs = signal.butter(order, normalized_cutoff_freq,btype="highpass")
acce1["X_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["X_filter_abs"])
acce1["Y_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["Y_filter_abs"])
acce1["Z_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["Z_filter_abs"])


acce1["X_velocity"] = integrate.cumtrapz(acce1["X_filter"], x=acce1["TIME"], initial=0)
acce1["Y_velocity"] = integrate.cumtrapz(acce1["Y_filter"], x=acce1["TIME"], initial=0)
acce1["Z_velocity"] = integrate.cumtrapz(acce1["Z_filter"], x=acce1["TIME"], initial=0)

acce1.loc[acce1["X_filter_abs"] < 0.0015, 'X_velocity'] = 0
acce1.loc[acce1["Y_filter_abs"] < 0.0015, 'Y_velocity'] = 0
acce1.loc[acce1["Z_filter_abs"] < 0.0015, 'Z_velocity'] = 0

print(len(acce1.loc[acce1["X_velocity"] == 0]))

# lista_poczatek_zer = [0, 4141, 5352, 6433, 7307, 8137, 8886]\
# Y_velocity
lpn_Y = [1867, 3548, 4974, 5975, 7072, 7791, 8594, 9727]
lkn_Y = [2095, 4140, 5351, 6432, 7306, 8136, 8885, 9957]

for index, begin in enumerate(lpn_Y):
    if index !=0:
        acce1["Y_velocity"][begin-1:lkn_Y[index]] = acce1["Y_velocity"][begin-1:lkn_Y[index]] + (abs(acce1["Y_velocity"][lpn_Y[index]]) - abs(acce1["Y_velocity"][lkn_Y[index-1]]) )

lp_Z = []

for index, value in acce1["Z_velocity"].iteritems():
    if value != 0:
        lp_Z.append(index)

# print(lp_Z)
lpn_Z = []
lkn_Z = []
for index, value in enumerate(lp_Z):
    if value != lp_Z[-1]:
        if value+1 != lp_Z[index+1]:
            lkn_Z.append(value)
        if value-1 != lp_Z[index-1]:
            lpn_Z.append(value)
    else:
        lkn_Z.append(value)

print(lpn_Z)
print(lkn_Z)




# plt.figure()
# sns.lineplot(y=acce1["X_filter_abs"], x=acce1["TIME"])
# sns.lineplot(y=acce1["Y_filter_abs"], x=acce1["TIME"])
# sns.lineplot(y=acce1["Z_filter_abs"], x=acce1["TIME"])
# print(acce1.head())

plt.figure()
sns.lineplot(y=filtered_signal_X, x=acce1["TIME"])
sns.lineplot(y=filtered_signal_Y, x=acce1["TIME"])
sns.lineplot(y=filtered_signal_Z, x=acce1["TIME"])


plt.figure()
sns.lineplot(y=acce1["X_velocity"], x=acce1["TIME"])
sns.lineplot(y=acce1["Y_velocity"], x=acce1["TIME"])
sns.lineplot(y=acce1["Z_velocity"], x=acce1["TIME"])

plt.figure()
sns.lineplot(y=integrate.cumtrapz(acce1["X_velocity"],x=acce1["TIME"], initial=0), x=acce1["TIME"])
sns.lineplot(y=integrate.cumtrapz(acce1["Y_velocity"],x=acce1["TIME"], initial=0), x=acce1["TIME"])
sns.lineplot(y=integrate.cumtrapz(acce1["Y_velocity"],x=acce1["TIME"],initial=0), x=acce1["TIME"])
plt.show()

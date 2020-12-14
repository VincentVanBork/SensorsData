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
    x="TIME", y="X", label="X")
sns.lineplot(
    data=acce1,
    x="TIME", y="Y" ,label="Y")
sns.lineplot(
    data=acce1,
    x="TIME", y="Z",label="Z")

acce1["SAMPLING_TIME"] = acce1["TIME"].diff()
# print(acce1.head())

sampling_freq = 1 / acce1["SAMPLING_TIME"].mean()

order=5

cutoff_freq = sampling_freq / 1400

# number_of_samples = len(acce1)

# time = np.linspace(0, acce1["SAMPLING_TIME"].mean(), number_of_samples, endpoint=False)

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

# acce1["X_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["X_filter_abs"])
# acce1["Y_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["Y_filter_abs"])
# acce1["Z_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["Z_filter_abs"])


acce1["X_velocity"] = integrate.cumtrapz(acce1["X_filter"], x=acce1["TIME"], initial=0)
acce1["Y_velocity"] = integrate.cumtrapz(acce1["Y_filter"], x=acce1["TIME"], initial=0)
acce1["Z_velocity"] = integrate.cumtrapz(acce1["Z_filter"], x=acce1["TIME"], initial=0)

acce1.loc[acce1["X_filter_abs"] < 0.0015, 'X_velocity'] = 0
acce1.loc[acce1["Y_filter_abs"] < 0.0015, 'Y_velocity'] = 0
acce1.loc[acce1["Z_filter_abs"] < 0.0015, 'Z_velocity'] = 0

lp_X = []
for index, value in acce1["X_velocity"].iteritems():
    if value != 0:
        lp_X.append(index)

# print(lp_X)
lpn_X = []
lkn_X = []
for index, value in enumerate(lp_X):
    if value != lp_X[-1]:
        if value+1 != lp_X[index+1]:
            lkn_X.append(value)
        if value-1 != lp_X[index-1]:
            lpn_X.append(value)
    else:
        lkn_X.append(value)
print("LPN_X: ")
print(lpn_X)
print("LKN_X: ")
print(lkn_X)
for index, begin in enumerate(lpn_X):
    if index !=0:
        acce1["X_velocity"][lkn_X[index-1]:lpn_X[index]]  = acce1["X_velocity"][lkn_X[index-1]]

# print(len(acce1.loc[acce1["X_velocity"] == 0]))

# Y_velocity
original_lpn_Y = [1867, 3548, 4974, 5975, 7072, 7791, 8594, 9727]
original_lkn_Y = [2095, 4140, 5351, 6432, 7306, 8136, 8885, 9957]
print(original_lkn_Y)
print(original_lpn_Y)



lp_Y = []
for index, value in acce1["Y_velocity"].iteritems():
    if value != 0:
        lp_Y.append(index)
# print(lp_Y)
lpn_Y = []
lkn_Y = []
for index, value in enumerate(lp_Y):
    if value != lp_Y[-1]:
        if value+1 != lp_Y[index+1]:
            lkn_Y.append(value)
        if value-1 != lp_Y[index-1]:
            lpn_Y.append(value)
    else:
        lkn_Y.append(value)

print("LPN_Y: ")
print(lpn_Y)
print("LKN_Y: ")
print(lkn_Y)


for index, begin in enumerate(lpn_Y):
    if index !=0:
        acce1["Y_velocity"][begin-1:lkn_Y[index]] = acce1["Y_velocity"][begin-1:lkn_Y[index]] + (abs(acce1["Y_velocity"][lpn_Y[index]]) - abs(acce1["Y_velocity"][lkn_Y[index-1]]) )
        acce1["Y_velocity"][lkn_Y[index-1]:lpn_Y[index]]  = acce1["Y_velocity"][lkn_Y[index-1]]


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
print("LPN_Z: ")
print(lpn_Z)
print("LKN_Z: ")
print(lkn_Z)

acce1["Z_velocity"][lpn_Z[1]-1:lkn_Z[1]] = acce1["Z_velocity"][lpn_Z[1]-1:lkn_Z[1]] - (acce1["Z_velocity"][lpn_Z[1]] - acce1["Z_velocity"][lkn_Z[0]])
acce1["Z_velocity"][lpn_Z[2]-1:lkn_Z[2]] = acce1["Z_velocity"][lpn_Z[2]-1:lkn_Z[2]] - (acce1["Z_velocity"][lpn_Z[1]] - acce1["Z_velocity"][lkn_Z[0]])
for index, value in acce1["Z_velocity"][lpn_Z[2]-1:lkn_Z[2]].iteritems():
    if value > 0:
        acce1["Z_velocity"][index] = acce1["Z_velocity"][index] - ( acce1["Z_velocity"][lkn_Z[1]] - acce1["Z_velocity"][lpn_Z[2]])
    else:
        acce1["Z_velocity"][index] = acce1["Z_velocity"][index] + ( acce1["Z_velocity"][lkn_Z[1]] - acce1["Z_velocity"][lpn_Z[2]])

acce1["Z_velocity"][lpn_Z[3]-1:lkn_Z[3]] = acce1["Z_velocity"][lpn_Z[3]-1:lkn_Z[3]] + (abs(acce1["Z_velocity"][lpn_Z[3]]) - abs(acce1["Z_velocity"][lkn_Z[2]]))

acce1["Z_velocity"][lpn_Z[4]-1:lkn_Z[4]] = acce1["Z_velocity"][lpn_Z[4]-1:lkn_Z[4]] + (abs(acce1["Z_velocity"][lpn_Z[4]]) - abs(acce1["Z_velocity"][lkn_Z[3]]))

acce1["Z_velocity"][lpn_Z[4]-1:lkn_Z[4]] = acce1["Z_velocity"][lpn_Z[4]-1:lkn_Z[4]] + (abs(acce1["Z_velocity"][lpn_Z[4]]) - abs(acce1["Z_velocity"][lkn_Z[3]]))

acce1["Z_velocity"][lpn_Z[5]-1:lkn_Z[5]] = acce1["Z_velocity"][lpn_Z[5]-1:lkn_Z[5]] + (abs(acce1["Z_velocity"][lpn_Z[5]]) - abs(acce1["Z_velocity"][lkn_Z[4]]))

acce1["Z_velocity"][lpn_Z[6]-1:lkn_Z[6]] = acce1["Z_velocity"][lpn_Z[6]-1:lkn_Z[6]] + (abs(acce1["Z_velocity"][lpn_Z[6]]) - abs(acce1["Z_velocity"][lkn_Z[5]]))
acce1["Z_velocity"][lpn_Z[7]-1:lkn_Z[7]] = acce1["Z_velocity"][lpn_Z[7]-1:lkn_Z[7]] + (abs(acce1["Z_velocity"][lpn_Z[7]]) - abs(acce1["Z_velocity"][lkn_Z[6]]))

for index, begin in enumerate(lpn_Z):
    if index !=0:
        acce1["Z_velocity"][lkn_Z[index-1]:lpn_Z[index]]  = acce1["Z_velocity"][lkn_Z[index-1]]

plt.figure()
plt.title("FILTER ACCELERATION ABSOLUTE VALUES")
sns.lineplot(y=acce1["X_filter_abs"], x=acce1["TIME"],label="X")
sns.lineplot(y=acce1["Y_filter_abs"], x=acce1["TIME"],label="Y")
sns.lineplot(y=acce1["Z_filter_abs"], x=acce1["TIME"],label="Z")

# print(acce1.head())

plt.figure()
plt.title("FILTERED ACCELERATION X,Y,Z")
sns.lineplot(y=filtered_signal_X, x=acce1["TIME"] , label="X")
sns.lineplot(y=filtered_signal_Y, x=acce1["TIME"], label="Y")
sns.lineplot(y=filtered_signal_Z, x=acce1["TIME"], label="Z")


plt.figure()
plt.title("VELOCITY X,Y,Z")
sns.lineplot(y=acce1["X_velocity"], x=acce1["TIME"], label="X")
sns.lineplot(y=acce1["Y_velocity"], x=acce1["TIME"], label="Y")
sns.lineplot(y=acce1["Z_velocity"], x=acce1["TIME"], label="Z")

# acce1["X_velocity"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["X_velocity"])
# acce1["Y_velocity"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["Y_velocity"])
# acce1["Z_velocity"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, acce1["Z_velocity"])


acce1["X_POSITION"] = integrate.cumtrapz(acce1["X_velocity"],x=acce1["TIME"], initial=0)
acce1["Y_POSITION"]= integrate.cumtrapz(acce1["Y_velocity"],x=acce1["TIME"], initial=0)
acce1["Z_POSITION"]= integrate.cumtrapz(acce1["Z_velocity"],x=acce1["TIME"],initial=0)

plt.figure()
plt.title("POSITION X,Y,Z")
sns.lineplot(y=acce1["X_POSITION"], x=acce1["TIME"], label="X")
sns.lineplot(y=acce1["Y_POSITION"], x=acce1["TIME"], label="Y")
sns.lineplot(y=acce1["Z_POSITION"], x=acce1["TIME"],label="Z")

plt.show()

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

# last_velocity = 0
# index_last_velocity = 0
# velocity_count = 0
# final_velocity = 0
# index_final_velocity=0
# zeroth_count = 0

# for index, vel in acce1["Y_velocity"].items():
    
#     if vel != 0:
#         f = open("zeroths.txt", "a")
#         f.write("not zero count" + " " + str(index)+"\n")
#         f.close()

#         if last_velocity != 0:
#             if acce1["Y_velocity"][1+index_last_velocity] == 0:
#                 final_velocity = vel
#                 index_final_velocity = index
#                 difference = abs(final_velocity) - abs(last_velocity)
#                 if final_velocity < 0:
#                     final_velocity += difference
#                 else:
#                     final_velocity -= difference

#                 acce1["Y_velocity"][index] = final_velocity
#                 index_last_velocity = 0
#                 last_velocity = 0
#                 velocity_count = 0
#                 final_velocity = 0
#                 index_final_velocity=0
#                 zeroth_count = 0
#             else:
#                 last_velocity = vel
#                 index_last_velocity = index
#         else:
#             last_velocity = vel
#             index_last_velocity = index

#     else:
#         f = open("zeroths.txt", "a")
#         f.write("zero count" + str(zeroth_count) + " " + str(index)+"\n")
#         f.close()
#         zeroth_count +=1

# lista_poczatek_zer = [0, 4141, 5352, 6433, 7307, 8137, 8886]\
# Y_velocity
lpn = [1867, 3548, 4974, 5975, 7072, 7791, 8594, 9727]
lkn = [2095, 4140, 5351, 6432, 7306, 8136, 8885, 9957]
for index, begin in enumerate(lpn):
    acce1[begin:lkn[index]] = acce1[begin:lkn[index]] + (abs(acce1[lpn[index+1]]) - abs(acce1[lkn[index]]) )


# acce1["Y_velocity"][3548:4141] = acce1["Y_velocity"][3548:4141] + (abs(acce1["Y_velocity"][3548]) - abs(acce1["Y_velocity"][2095]))
# acce1["Y_velocity"][4974:5352] = acce1["Y_velocity"][4974:5352] + (abs(acce1["Y_velocity"][4974]) - abs(acce1["Y_velocity"][3548]))
# acce1["Y_velocity"][5975:6433] = acce1["Y_velocity"][5975:6433] + (abs(acce1["Y_velocity"][5975]) - abs(acce1["Y_velocity"][5351]))
# acce1["Y_velocity"][7072:7307] = acce1["Y_velocity"][7072:7307] + (abs(acce1["Y_velocity"][7072]) - abs(acce1["Y_velocity"][6432]))
# acce1["Y_velocity"][7791:8137] = acce1["Y_velocity"][7791:8137] + (abs(acce1["Y_velocity"][7791]) - abs(acce1["Y_velocity"][7306]))
# acce1["Y_velocity"][8594:8886] = acce1["Y_velocity"][8594:8886] + (abs(acce1["Y_velocity"][8594]) - abs(acce1["Y_velocity"][8136]))
# acce1["Y_velocity"][9727:9958] = acce1["Y_velocity"][9727:9958]  + (abs(acce1["Y_velocity"][9727]) - abs(acce1["Y_velocity"][8885]))



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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import integrate


def import_data(name):
    colnames=[ 'X', 'Y', 'Z', "ACCURACY", "TIMESTAMP"]
    acce = pd.read_csv(f"./data/{name}", names=colnames)
    acce["TIME"] = (acce["TIMESTAMP"] - acce["TIMESTAMP"].iloc[0])/1000000000

    return acce


def prepare_test(df, btype="lowpass", order=5, div_freq=1400):
    df["SAMPLING_TIME"] = df["TIME"].diff()
    sampling_freq = 1 / df["SAMPLING_TIME"].mean()
    order=5
    cutoff_freq = sampling_freq / div_freq
    # number_of_samples = len(df)
    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    return signal.butter(order, normalized_cutoff_freq, btype=btype)


def filter_acceleration(df, btype="lowpass", order=5, div_freq=1400):
    #kwargs arg maybe
    numerator_coeffs, denominator_coeffs = prepare_test(df, btype=btype, order=order, div_freq=div_freq)
    df["X_filter"] = signal.lfilter(numerator_coeffs, denominator_coeffs, df["X"])
    df["Y_filter"] = signal.lfilter(numerator_coeffs, denominator_coeffs, df["Y"])
    df["Z_filter"] = signal.lfilter(numerator_coeffs, denominator_coeffs, df["Z"])
    df["X_filter_abs"] = df["X_filter"].abs()
    df["Y_filter_abs"] = df["Y_filter"].abs()
    df["Z_filter_abs"] = df["Z_filter"].abs()
    high_numerator_coeffs, high_denominator_coeffs = prepare_test(df, btype="highpass", order=order, div_freq=div_freq)
    df["X_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, df["X_filter_abs"])
    df["Y_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, df["Y_filter_abs"])
    df["Z_filter_abs"] = signal.lfilter(high_numerator_coeffs, high_denominator_coeffs, df["Z_filter_abs"])


def calculate_velocity(df):
    df["X_velocity"] = integrate.cumtrapz(df["X_filter"], x=df["TIME"], initial=0)
    df["Y_velocity"] = integrate.cumtrapz(df["Y_filter"], x=df["TIME"], initial=0)
    df["Z_velocity"] = integrate.cumtrapz(df["Z_filter"], x=df["TIME"], initial=0)


def calculate_position(df):
    df["X_position"] = integrate.cumtrapz(df["X_velocity"],x=df["TIME"], initial=0)
    df["Y_position"]= integrate.cumtrapz(df["Y_velocity"],x=df["TIME"], initial=0)
    df["Z_position"]= integrate.cumtrapz(df["Z_velocity"],x=df["TIME"],initial=0)


def zero_velocity(df, threshold=0.0015):
    df.loc[df["X_filter_abs"] < threshold, 'X_velocity'] = 0
    df.loc[df["Y_filter_abs"] < threshold, 'Y_velocity'] = 0
    df.loc[df["Z_filter_abs"] < threshold, 'Z_velocity'] = 0


def all_vel_indicies(df):
    lp_X=[]
    lp_Y=[]
    lp_Z=[]

    lp_X = df.loc[df["X_velocity"] != 0, "X_velocity"]
    lp_Y = df.loc[df["Y_velocity"] != 0, "Y_velocity"]  
    lp_Z = df.loc[df["Z_velocity"] != 0, "Z_velocity"]

    # for index, value in df["X_velocity"].iteritems():
    #     if value != 0:
    #         lp_X.append(index)
    
    # for index, value in df["Y_velocity"].iteritems():
    #     if value != 0:
    #         lp_Y.append(index)

    # for index, value in df["Z_velocity"].iteritems():
    #     if value != 0:
    #         lp_Z.append(index)
    return lp_X.index.tolist(), lp_Y.index.tolist(), lp_Z.index.tolist()


def find_vel_borders(velocity_indicies):
    lpn = []
    lkn = []
    for index, value in enumerate(velocity_indicies):
        if value != velocity_indicies[-1]:
            if value+1 != velocity_indicies[index+1]:
                lkn.append(value)
            if value-1 != velocity_indicies[index-1]:
                lpn.append(value)
        else:
            # if len(lpn) != len(lkn):
            lkn.append(value)
        
    print("LPN: ")
    print(lpn)
    print("LKN: ")
    print(lkn)
    return lpn,lkn


def set_equal_velocity(df, begin_range_vel, end_range_vel, velocity_name):
    #this could be better in terms of code quality I know but I refuse to make it better
    for index, begin in enumerate(begin_range_vel):
        drift_difference = abs(abs(df[velocity_name][begin_range_vel[index]]) - abs(df[velocity_name][end_range_vel[index-1]]))

        if index !=0:
            df[velocity_name][begin-1:end_range_vel[index]]= df[velocity_name][begin-1:end_range_vel[index]].apply(lambda x: x + drift_difference if x < 0 else x - drift_difference)
            df[velocity_name][end_range_vel[index-1]:begin_range_vel[index]]  = df[velocity_name][end_range_vel[index-1]]
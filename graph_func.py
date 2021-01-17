from compute_func import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from scipy import integrate
from scipy.interpolate import interp1d

def create_graphs(names, data):
    if "acc_filter" in names:
        plt.figure()
        plt.title("FILTERED ACCELERATION X,Y,Z")
        sns.lineplot(y=data["X_filter"], x=data["TIME"], label="X")
        sns.lineplot(y=data["Y_filter"], x=data["TIME"], label="Y")
        sns.lineplot(y=data["Z_filter"], x=data["TIME"], label="Z")
    if "raw" in names:        
        plt.figure()
        plt.title("RAW DATA X,Y,Z")
        sns.lineplot(y=data["X"], x=data["TIME"], label="X")
        sns.lineplot(y=data["Y"], x=data["TIME"], label="Y")
        sns.lineplot(y=data["Z"], x=data["TIME"], label="Z")

    if "raw_vs_filter" in names:
        plt.figure()
        plt.title("RAW DATA vs filtered")
        sns.lineplot(y=data["X"]/data["X"].abs().max(), x=data["TIME"], label="X")
        if "X_filter" in data.columns:
            sns.lineplot(y=data["X_filter"]/data["X_filter"].abs().max(), x=data["TIME"], label="X_FILTERED")

        plt.figure()
        plt.title("RAW DATA vs filtered")
        sns.lineplot(y=data["Y"]/data["Y"].abs().max(), x=data["TIME"], label="Y")
        if "Y_filter" in data.columns:
            sns.lineplot(y=data["Y_filter"]/data["Y_filter"].abs().max(), x=data["TIME"], label="Y_FILTERED")

        plt.figure()
        plt.title("RAW DATA vs filtered")
        sns.lineplot(y=data["Z"]/data["Z"].abs().max(), x=data["TIME"], label="Z")
        if "Z_filter" in data.columns:
            sns.lineplot(y=data["Z_filter"]/data["Z_filter"].abs().max(), x=data["TIME"], label="Z_FILTERED")

    if "position" in names:
        # print(data.head())

        plt.figure()
        plt.title("position X,Y,Z")
        sns.lineplot(y=data["X_position"], x=data["TIME"], label="X")
        sns.lineplot(y=data["Y_position"], x=data["TIME"], label="Y")
        sns.lineplot(y=data["Z_position"], x=data["TIME"], label="Z")

    if "velocity" in names:
        plt.figure()
        plt.title("velocity X,Y,Z")
        sns.lineplot(y=data["X_velocity"], x=data["TIME"], label="X")
        sns.lineplot(y=data["Y_velocity"], x=data["TIME"], label="Y")
        sns.lineplot(y=data["Z_velocity"], x=data["TIME"], label="Z")
        # print(data.head())

    if "wavelets" in names:
        plt.figure()
        plt.title("wavelets")

    

        sns.lineplot(y=data["X"], x=data["TIME"])



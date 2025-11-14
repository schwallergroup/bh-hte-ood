import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import sys
import os
import re
import glob
import copy
import pickle
import collections
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def get_reactivity_class(true_yield, b1, _LABELS):
    if true_yield >= b1:
        return _LABELS[1]
    else:
        return _LABELS[0]

def cap_values(df, column_name, cap_value):
    df[column_name] = df[column_name].clip(upper=cap_value)
    return df
   
def filter_drfp(drfp_enriched):
    return drfp_enriched[2048:]
  
def get_folder_names(dir_name, num):
    folder_names = {}
    epoch_list = [i for i in range(1,num)]
    for folder in os.listdir(dir_name):
        try:
            if int(folder.split("-")[-1]) in epoch_list:
                folder_names[int(folder.split("-")[-1])] = folder
        except:
            pass
    return folder_names

def check_file_exists(directory, file_format):
    # list all files in the directory
    for filename in os.listdir(directory):
        # check the extension of each file
        if filename.endswith(file_format):
            # if a file with the specified format is found, return True
            return True
    # if no file with the specified format is found, return False
    return False

def custom_round(value, threshold, labels = [0, 1]):
    if value < threshold:
        return labels[0]
    else:
        return labels[1]
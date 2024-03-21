import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, date

# DATA_DIR = "/hdd/traffic_data_2019/april_2019/"
# in_file = "GA0419.VOL"
day_count = {
    "JAN": [i for i in range(190101, 190132)], 
    "FEB": [i for i in range(190201, 190229)],
    "MAR": [i for i in range(190301, 190332)], 
    "APR": [i for i in range(190401, 190431)], 
    "MAY": [i for i in range(190501, 190532)], 
    "JUN": [i for i in range(190601, 190631)], 
    "JUL": [i for i in range(190701, 190732)], 
    "AUG": [i for i in range(190801, 190832)], 
    "SEP": [i for i in range(190901, 190931)], 
    "OCT": [i for i in range(191001, 191032)], 
    "NOV": [i for i in range(191101, 191131)], 
    "DEC": [i for i in range(191201, 191232)],
}

def linear_interplotion(data_dir, out_dir, in_file, month="APR"):
    """
    prev: previous valid time-series data
    next: next valid time-series data
    length: length of missing data points
    """
    with open(data_dir+in_file, "r") as fd:
        lines = fd.read().splitlines()

    full_data = []
    full_data.append(lines[0])

    for i in range(1, len(lines)-1):
        prev_date = int(lines[i-1][13:19])
        curr_date = int(lines[i][13:19])
        if curr_date == prev_date:
            continue
        elif (curr_date-prev_date) > 1 and curr_date != day_count[month][0] and curr_date != day_count[month][-1]:
            prev_volume = [int(lines[i-1][20:20+24*5][t*5:t*5+5]) for t in range(24)]
            next_volume = [int(lines[i][20:20+24*5][t*5:t*5+5]) for t in range(24)]
            dist = [next_volume[k] - prev_volume[k] for k in range(len(prev_volume))]
            inter = [int(val/(curr_date-prev_date)) for val in dist]
            for j in range(curr_date-prev_date-1):
                curr = [(1+j)*inter[h]+prev_volume[h] for h in range(len(prev_volume))]
                text_vol = [str(num).zfill(5) for num in curr]
                final_text = lines[i][:17] + str(int(lines[i-1][17:19])+1+j).zfill(2) + str((int(lines[i-1][19])%7+1)) + ''.join(text_vol) + '0'
                full_data.append(final_text)
        full_data.append(lines[i])
    full_data.append(lines[-1])

    full_data_2nd = []
    full_data_2nd.append(full_data[0])
    
    for i in range(1, len(full_data)):
        prev_date = int(full_data[i-1][13:19])
        curr_date = int(full_data[i][13:19])
        if curr_date == day_count[month][0] and prev_date != day_count[month][-1]:  # missing last day data
            volume = [int(full_data[i][20:20+24*5][t*5:t*5+5]) for t in range(24)]
            for j in range(day_count[month][-1]-prev_date):
                curr = [random.randint(-int(.1*volume[h]), int(.1*volume[h]))+volume[h] for h in range(len(volume))]
                text_vol = [str(num).zfill(5) for num in curr]
                final_text = full_data[i-1][:17] + str(int(full_data[i-1][17:19])+1+j).zfill(2) + str((int(full_data[i-1][19])%7+1)) + ''.join(text_vol) + '0'
                full_data_2nd.append(final_text)

        elif curr_date == day_count[month][1] and prev_date != day_count[month][0]:  # missing first day data
            volume = [int(full_data[i][20:20+24*5][t*5:t*5+5]) for t in range(24)]
            for j in range(curr_date-day_count[month][0]):
                curr = [random.randint(-int(.1*volume[h]), int(.1*volume[h]))+volume[h] for h in range(len(volume))]
                text_vol = [str(num).zfill(5) for num in curr]
                final_text = full_data[i][:17] + str(day_count[month][0]+j)[-2:] + str((int(full_data[i-1][19])%7+1)) + ''.join(text_vol) + '0'
                full_data_2nd.append(final_text)
        full_data_2nd.append(full_data[i])

    with open(out_dir+in_file, 'w') as f:
        for ln in full_data_2nd:
            f.write(f"{ln}\n")

# in_file = "GA0419.VOL"
# linear_interplotion(in_file)
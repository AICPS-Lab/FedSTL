import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date


DATA_DIR = "/hdd/traffic_data_2019/"
in_file = "april_2019/corrected_GA0419.VOL"
month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
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
all_stations = "/hdd/traffic_data_2019/station_list.txt"
with open(all_stations) as f:
    lines = f.read().splitlines()

lines = [l.strip().split() for l in lines]
state_to_station = {}
for l in lines:
    state_to_station[l[0]] = l[1:]


def load_data(data_dir, in_file="april_2019/corrected_GA0419.VOL"):
    """
    Load traffic volume data for all stations and all dates in the month
    Returns json format dictionary
    """
    bin_file = data_dir + in_file
    with open(bin_file, "r") as fd:
        lines = fd.read().splitlines()

    daily_temp = [line[20:20+24*5] for line in lines]
    station_id = [line[5:11] for line in lines]

    daily_vol = {}
    for i in range(len(station_id)):
        daily_vol[station_id[i]] = []

    for i in range(len(daily_temp)):
        today = [int(daily_temp[i][t*5:t*5+5]) for t in range(24)]
        assert len(today) == 24
        
        daily_dict = {}
        daily_dict["direction"] = int(lines[i][11])
        daily_dict["lane"] = int(lines[i][12])
        daily_dict["function_classification"] = lines[i][4]
        daily_dict["date"] = int(lines[i][13:19])
        daily_dict["day"] = date(int(lines[i][13:15]), int(lines[i][15:17]), int(lines[i][17:19])).weekday()
        daily_dict["traffic_volume"] = today
        
        daily_vol[station_id[i]].append(daily_dict)
    
    return daily_vol


def load_monthly_data(data_dir="/hdd/traffic_data_2019/", in_file="processed/april_2019/AK0419.VOL", state="AK"):
    """
    Load traffic volume data for all stations and all dates in the month
    Returns json format dictionary
    """
    bin_file = data_dir + in_file
    with open(bin_file, "r") as fd:
        lines = fd.read().splitlines()

    daily_temp = []
    for line in lines:
        if line[5:11] in state_to_station[state]:
            if len(line) < 141:
                daily_temp.append(line[19:19+24*5])
            else:
                daily_temp.append(line[20:20+24*5])
    
    common_stations = [line[5:13] for line in lines if line[5:11] in state_to_station[state]]
    valid_lines = [line for line in lines if line[5:11] in state_to_station[state]]

    daily_vol = {}
    for i in range(len(common_stations)):
        daily_vol[common_stations[i]] = {}

    for i in range(len(daily_temp)):
        try:
            today = [int(daily_temp[i][t*5:t*5+5]) for t in range(24)]
        except ValueError:
            today = []
            for t in range(24):
                try:
                    today.append(int(daily_temp[i][t*5:t*5+5]))
                except ValueError:
                    today.append(
                        (int(daily_temp[i][(t-1)*5:(t-1)*5+5])+int(daily_temp[i][(t+1)*5:(t+1)*5+5])) // 2
                    )
        assert len(today) == 24
        # add day into the data
        get_day = date(int(lines[i][13:15]), int(lines[i][15:17]), int(lines[i][17:19])).weekday()  # get the day of the week as a number, where Monday is 0 and Sunday is 6
        list_day = [get_day] * 24
        res = [[today[i], list_day[i]] for i in range(24)]

        daily_vol[common_stations[i]][int(valid_lines[i][13:19])] = res
        # daily_dict["day"] = date(int(lines[i][13:15]), int(lines[i][15:17]), int(lines[i][17:19])).weekday()
        # daily_dict["traffic_volume"] = today
    
    return daily_vol


def aggregate_year():
    month = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    single_lane = []
    for m in month:
        in_file = DATA_DIR + "GA_" + m + "_2017.json"
        f = open(in_file)
        curr_month = json.load(f)
        station = curr_month["B02AAA"]
        curr_date = 0
        while curr_date < len(day_count[m]):
            try:
                for rec in station:
                    if rec["lane"] == 2 and rec["direction"] == 3 and rec["date"] == day_count[m][curr_date]:
                        single_lane.append(rec["traffic_volume"])
                        curr_date += 1
            except:
                print(m, curr_date)
    print("Total records:", len(single_lane))
    out_file = DATA_DIR + "B02AAA_dir3_lane2_2017.json"
    f = Path(out_file)
    f.touch(exist_ok=True)
    with open(f, "w") as outfile:
        json.dump(single_lane, outfile, indent=4)


def check_missing(curr_station, mon):
    curr_date = 0
    
    missing_date = []
    while curr_date < len(day_count[month[mon]]):
        try:
            curr_station[str(day_count[month[mon]][curr_date])]
        except KeyError:
            missing_date.append(day_count[month[mon]][curr_date])
        curr_date += 1
    print(missing_date)


def find_station_month(curr_station, mon, state_id):
    # curr_station: data for one month
    # mon: month
    single_lane = []
    
    curr_date = 0
    while curr_date < len(day_count[month[mon]]):
        try:
            single_rec = curr_station[str(day_count[month[mon]][curr_date])]
            for item in single_rec:
                item.append(state_id)
            # single_lane.extend(curr_station[str(day_count[month[mon]][curr_date])])
            single_lane.extend(single_rec)
        except KeyError:
            missing = True
            try_ind = 0
            while missing:
                try:
                    single_lane.extend(curr_station[str(day_count[month[mon]][try_ind])])
                    missing = False
                except:
                    try_ind += 1
    
        curr_date += 1
    return single_lane
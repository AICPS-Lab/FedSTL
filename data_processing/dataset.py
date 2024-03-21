import numpy as np
import pandas as pd
from os import walk
import json
from utils.highway_traffic_monitoring import load_monthly_data, find_station_month
from utils.linear_interpolation import linear_interplotion
from pathlib import Path
from tqdm import tqdm

from sklearn import preprocessing
import torch
from torch.utils.data import Dataset

"""
preprocess dataset
"""
DATA_DIR = "/hdd/traffic_data_2019/"
PROC_DIR = "/hdd/traffic_data_2019/processed/"
JSON_DIR = "/hdd/traffic_data_2019/json/"

months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", 
            "october", "november", "december"]
month_short = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
# states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", 
#           "IN", "IA", "KS", "KY", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
#           "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", 
#           "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
states = ["MA"]
# states to ID: use index (i)
# states = ["AL"]


# Load useful stations
all_stations = "/hdd/traffic_data_2019/station_list.txt"
with open(all_stations) as f:
    lines = f.read().splitlines()

lines = [l.strip().split() for l in lines]
state_to_station = {}
for l in lines:
    state_to_station[l[0]] = l[1:]



def fill_missing():
    for i, m in enumerate(months):
        fold_path = DATA_DIR+m+"_2019/"
        Path(DATA_DIR+"processed/"+m+"_2019/").mkdir(parents=True, exist_ok=True)
        print("Processing folder:", fold_path)
        filenames = next(walk(fold_path), (None, None, []))[2]
        out_dir = DATA_DIR+"processed/"+m+"_2019/"
        for f in filenames:
            print(f)
            linear_interplotion(fold_path, out_dir, f, month_short[i])



def to_json():
    """
    load preprocessed monthly VOL data into json files in the /json folder
    """
    for i, m in enumerate(months):
        fold_path = PROC_DIR+m+"_2019/"
        Path(DATA_DIR+"json/"+m+"_2019/").mkdir(parents=True, exist_ok=True)
        print("Processing folder:", fold_path)
        filenames = next(walk(fold_path), (None, None, []))[2]
        out_dir = DATA_DIR+"json/"+m+"_2019/"
        for f in filenames:
            if f[0:2] in states:
                daily_vol = load_monthly_data("/hdd/traffic_data_2019/", "processed/"+m+"_2019/"+f, f[0:2])
                save_path = out_dir + f[0:2] + ".json"
                out_f = Path(save_path)
                out_f.touch(exist_ok=True)
                with open(out_f, "w") as outfile:
                    json.dump(daily_vol, outfile, indent=2)



# def save_as_dataset():
#     """
#     load monthly json files and combine for the whole year
#     """
#     for i,s in enumerate(states):
#         # Get total stations 
#         example = JSON_DIR+"january_2019/"+s+".json"
#         f = open(example)
#         data = json.load(f).keys()
#         unique_station = [*set(data)]
#         print("Total stations:", len(unique_station))
        
#         # Process one state for whole year
#         curr_state = []
#         for station in tqdm(unique_station):
#             curr_station = []
#             flag = True
#             print("Processing station", station)
#             if station[:-2] in state_to_station[s]:
#                 for j,m in enumerate(months):
#                     curr_file = JSON_DIR+m+"_2019/"+s+".json"
#                     f = open(curr_file)
#                     monthly_data = json.load(f)
#                     try:
#                         data = monthly_data[station]    # data for each station and month
#                         station_month_ls = find_station_month(data, j)
#                         curr_station.extend(station_month_ls)
#                     except:
#                         flag = False
#                 if flag:
#                     curr_state.append(curr_station)
#                     print(len(curr_station))

#         state_set = np.array(curr_state)
#         print("Saved dataset shape:", state_set.shape)
#         outfile = "/hdd/traffic_data_2019/dataset/"+s
#         np.save(outfile, state_set)



def save_as_dataset():
    """
    load monthly json files and combine for the whole year
    """
    for i,s in enumerate(states):
        # Get total stations 
        example = JSON_DIR+"january_2019/"+s+".json"
        f = open(example)
        data = json.load(f).keys()
        unique_station = [*set(data)]   # station ID = station ID number + lane + direction
        print("Total stations:", len(unique_station))
        
        # Process one state for whole year
        curr_state = []
        for station in tqdm(unique_station):
            curr_station = []
            flag = True
            print("Processing station", station)
            if station[:-2] in state_to_station[s]:
                for j,m in enumerate(months):
                    curr_file = JSON_DIR+m+"_2019/"+s+".json"
                    f = open(curr_file)
                    monthly_data = json.load(f)
                    try:
                        data = monthly_data[station]    # data for each station and month
                        station_month_ls = find_station_month(data, j, int(f"{states.index(s):02}"+station))
                        curr_station.extend(station_month_ls)
                    except:
                        flag = False
                if flag:
                    curr_state.append(curr_station)
                    print(len(curr_station))

        state_set = np.array(curr_state)
        print("Saved dataset shape:", state_set.shape)
        outfile = "/hdd/traffic_data_2019/dataset/"+s
        np.save(outfile, state_set)




def load_fed_dataset(dataset_path, client_id):
    """
    split preprocessed data in .npy files into features (x) and targets (y)
    """
    STA_PATH = '/hdd/traffic_data_2019/'
    # read in station cluster and location 
    df = pd.read_csv(STA_PATH+"station_cluster.csv", index_col=0)
    df["unique-station"] = df["state"] + df["station"]

    saved = np.load(dataset_path, allow_pickle=True)

    scaled = np.zeros_like(saved)
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(saved.T[0]).T  # shape: 777, 8760
    # selected_ind = 50   # select first 50 stations
    # scaled = scaled[:selected_ind] # TODO: testing: using one row
    # days = saved[:selected_ind, :, 1]
    days = saved[:, :, 1]
    stations = saved[:, :, 2]

    features = []
    target = []

    for ind, row in enumerate(scaled):
        index = 0
        while index < row.shape[0]-6*24:
            features.append(np.array((row[index:index+5*24], days[ind][index:index+5*24], stations[ind][index:index+5*24])))
            target.append(row[index+5*24:index+6*24])
            index += 24

    features = np.array(features)
    target = np.array(target)
    print(features.shape, target.shape) # print feature and target shape

    train, val = int(0.8*features.shape[0]), int(0.1*features.shape[0])    # split train/val/test = 0.8/0.1/0.1
    test = features.shape[0] - train - val  

    saving_dataset = {}
    saving_dataset["train_x"] = features[:train, :]
    saving_dataset["val_x"] = features[train:train+val, :]
    saving_dataset["test_x"] = features[train+val:, :]
    saving_dataset["train_y"] = target[:train, :]
    saving_dataset["val_y"] = target[train:train+val, :]
    saving_dataset["test_y"] = target[train+val:, :]

    for fold in ["train", "test", "val"]:
        x_outfile = "/hdd/traffic_data_2019/torch_dataset/"+fold+"_"+str(client_id)+"_x" # add client ID to saved x data
        np.save(x_outfile, saving_dataset[fold+"_x"])
        y_outfile = "/hdd/traffic_data_2019/torch_dataset/"+fold+"_"+str(client_id)+"_y" # add client ID to saved y data
        np.save(y_outfile, saving_dataset[fold+"_y"])




def load_dataset(dataset_path):
    """
    split preprocessed data in .npy files into features (x) and targets (y)
    """
    saved = np.load(dataset_path, allow_pickle=True)
    scaled = np.zeros_like(saved)
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(saved.T[0]).T  # shape: 777, 8760
    selected_ind = 50   # select first 50 stations
    scaled = scaled[:selected_ind] # TODO: testing: using one row
    days = saved[:selected_ind, :, 1]

    features = []
    target = []

    for ind, row in enumerate(scaled):
        index = 0
        while index < row.shape[0]-6*24:
            features.append(np.array((row[index:index+5*24], days[ind][index:index+5*24], np.full((5*24), ind))).T)
            target.append(row[index+5*24:index+6*24])
            index += 24

    features = np.array(features)
    target = np.array(target)
    print(features.shape, target.shape) # print feature and target shape

    train, val = int(0.9*features.shape[0]), int(0.05*features.shape[0])
    test = features.shape[0] - train - val

    saving_dataset = {}
    saving_dataset["train_x"] = features[:train, :]
    saving_dataset["val_x"] = features[train:train+val, :]
    saving_dataset["test_x"] = features[train+val:, :]
    saving_dataset["train_y"] = target[:train, :]
    saving_dataset["val_y"] = target[train:train+val, :]
    saving_dataset["test_y"] = target[train+val:, :]

    for fold in ["train", "test", "val"]:
        x_outfile = "/hdd/traffic_data_2019/torch_dataset/"+fold+"_x"
        np.save(x_outfile, saving_dataset[fold+"_x"])
        y_outfile = "/hdd/traffic_data_2019/torch_dataset/"+fold+"_y"
        np.save(y_outfile, saving_dataset[fold+"_y"])
    


def main():
    # step 1
    to_json()
    # step 2
    save_as_dataset()
    # step 3
    data_path = [
        '/hdd/traffic_data_2019/dataset/ME.npy',
        '/hdd/traffic_data_2019/dataset/MA.npy',]
    for i, dataset_path in enumerate(data_path):
        load_fed_dataset(dataset_path, i+13)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')
"""
Preprocessing dataset.
Filling missing data, converting raw txt data to json and numpy format.
Saving numpy files to training, val, and testing set.
"""
import os
import json
import random
import numpy as np
from os import walk
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn import preprocessing
import matplotlib.pyplot as plt
from utils.linear_interpolation import linear_interplotion
from utils.highway_traffic_monitoring import load_monthly_data, find_station_month



DATA_DIR = "/hdd/traffic_data_2019/"
PROC_DIR = "/hdd/FHWA_dataset/processed/"
JSON_DIR = "/hdd/FHWA_dataset/json/"

months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", 
            "october", "november", "december"]
month_short = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
# states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", 
#           "IN", "IA", "KS", "KY", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
#           "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", 
#           "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
states = ["AL"]
client_count = 100


all_stations = "/hdd/traffic_data_2019/station_list.txt"
with open(all_stations) as f:
    lines = f.read().splitlines()

lines = [l.strip().split() for l in lines]
state_to_station = {}   # list for every station ID in a state.
for l in lines:
    state_to_station[l[0]] = l[1:]



def fill_missing():
    for i, m in enumerate(months):
        fold_path = DATA_DIR+m+"_2019/"
        Path("/hdd/FHWA_dataset/processed/"+m+"_2019/").mkdir(parents=True, exist_ok=True)
        print("Processing folder:", fold_path)
        filenames = next(walk(fold_path), (None, None, []))[2]
        out_dir = "/hdd/FHWA_dataset/processed/"+m+"_2019/"
        for f in filenames:
            print(f)
            linear_interplotion(fold_path, out_dir, f, month_short[i])



def to_json():
    """
    Load preprocessed monthly VOL data into json files in the json/ folder.
    """
    for i, m in enumerate(months):
        fold_path = PROC_DIR+m+"_2019/"
        Path("/hdd/FHWA_dataset/json/"+m+"_2019/").mkdir(parents=True, exist_ok=True)
        print("Processing folder:", fold_path)
        filenames = next(walk(fold_path), (None, None, []))[2]
        out_dir = "/hdd/FHWA_dataset/json/"+m+"_2019/"
        for f in filenames:
            if f[0:2] in states:
                daily_vol = load_monthly_data("/hdd/FHWA_dataset/", "processed/"+m+"_2019/"+f, f[0:2])
                save_path = out_dir + f[0:2] + ".json"
                out_f = Path(save_path)
                out_f.touch(exist_ok=True)
                with open(out_f, "w") as outfile:
                    json.dump(daily_vol, outfile, indent=2)



def save_as_dataset():
    """
    Load monthly json files and combine for the whole year.
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
        outfile = "/hdd/FHWA_dataset/dataset/"+s
        np.save(outfile, state_set)




def generated_dataset():
    """
    Generate dataset into input and targets with sliding window.
    """
    for i,s in enumerate(states):
        example = JSON_DIR+"january_2019/"+s+".json"
        f = open(example)
        data = json.load(f).keys()
        unique_station = [*set(data)]   # station ID = station ID number + lane + direction
        print("Total stations:", len(unique_station))
        
        # Process one state for whole year
        for station in tqdm(unique_station):
            curr_station = []
            print("Processing station", station)
            if station[:-2] in state_to_station[s]:
                curr_file = JSON_DIR+"january_2019/"+s+".json"
                f = open(curr_file)
                monthly_data = json.load(f)
                example_key = next(iter(monthly_data[station]))
                example_data = np.array(monthly_data[station][example_key])[:,0]
                while example_data.max() > 200:
                    example_data = (example_data/2).astype(int)
                for i in range(1200):
                    mod_data = np.zeros_like(example_data)
                    perturb = random.uniform(-0.15, -0.05) if random.randint(0, 1) == 0 else random.uniform(0.05, 0.15)
                    mod_data = example_data+np.array(example_data*perturb).astype(int)
                    for i in range(len(mod_data)):
                        percent_change = random.uniform(-0.01, 0.01)
                        mod_data[i] += int(mod_data[i] * percent_change) 
                    curr_station.append(mod_data)

            station_set = np.array(curr_station)
            outfile = "/hdd/FHWA_dataset/dataset/"+str(station)
            np.savetxt("/hdd/FHWA_dataset/txt_dataset/"+str(station)+'.txt', station_set, fmt='%d') 
            np.save(outfile, station_set)




def load_fed_dataset(dataset_path, client_id):
    """
    Split preprocessed data in .npy files into features (x) and targets (y).
    """
    STA_PATH = '/hdd/traffic_data_2019/'
    df = pd.read_csv(STA_PATH+"station_cluster.csv", index_col=0)
    df["unique-station"] = df["state"] + df["station"]
    saved = np.load(dataset_path, allow_pickle=True)
    
    scaled = np.zeros_like(saved)
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(saved.T).T
    plt.plot()

    features = []
    target = []

    index = 0
    while index < scaled.shape[0]-5:
        features.append(np.array(scaled[index:index+5]).flatten())
        target.append(np.array(scaled[index+5:index+6]).flatten())
        index += 6

    features = np.array(features)
    target = np.array(target)

    train, val = int(0.9*features.shape[0]), int(0.05*features.shape[0])    # split train/val/test = 0.9/0.05/0.05
    test = features.shape[0] - train - val  

    saving_dataset = {}
    saving_dataset["train_x"] = features[:train, :]
    saving_dataset["val_x"] = features[train:train+val, :]
    saving_dataset["test_x"] = features[train+val:, :]
    saving_dataset["train_y"] = target[:train, :]
    saving_dataset["val_y"] = target[train:train+val, :]
    saving_dataset["test_y"] = target[train+val:, :]

    for fold in ["train", "test", "val"]:
        x_outfile = "/hdd/FHWA_dataset/torch_dataset/"+fold+"_"+str(client_id)+"_x" # add client ID to saved x data
        np.save(x_outfile, saving_dataset[fold+"_x"])
        y_outfile = "/hdd/FHWA_dataset/torch_dataset/"+fold+"_"+str(client_id)+"_y" # add client ID to saved y data
        np.save(y_outfile, saving_dataset[fold+"_y"])

    


def main():
    # fill_missing()
    # step 1
    # to_json()
    # step 2
    # generated_dataset()
    # step 3
    NPY_DATA = "/hdd/FHWA_dataset/dataset/"
    files = os.listdir(NPY_DATA)
    data_path = [os.path.join(NPY_DATA, f) for f in files]
    for i, dataset_path in enumerate(data_path):
        load_fed_dataset(dataset_path, i)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')
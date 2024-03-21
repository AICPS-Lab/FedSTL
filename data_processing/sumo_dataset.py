import numpy as np
import pandas as pd
from string import ascii_lowercase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join


data_path = "/hdd/SUMO_dataset/SUMO_dataset/" 
save_path = "/hdd/SUMO_dataset/learn_dataset/"


def load_fed_dataset(dataset_path, client_id):
    """
    split preprocessed data in .npy files into features (x) and targets (y)
    """
    saved = np.load(dataset_path, allow_pickle=True)    # shape: (400, 6)
    scaled = np.zeros_like(saved)
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(saved)

    features = []
    target = []
    
    index = 0
    while index < scaled.shape[0]-41:
        features.append(scaled[index:index+40])
        target.append(scaled[index+1:index+41])
        index += 1

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
        x_outfile = save_path+fold+"_"+str(client_id)+"_x" # add client ID to saved x data
        np.save(x_outfile, saving_dataset[fold+"_x"])
        y_outfile = save_path+fold+"_"+str(client_id)+"_y" # add client ID to saved y data
        np.save(y_outfile, saving_dataset[fold+"_y"])



def main():
    all_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    for i, data_file in enumerate(all_files):
        load_fed_dataset(data_path+data_file, i)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')
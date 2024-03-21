"""
The main file for training and evaluating FedAvg.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
from torch.utils.data import DataLoader
from IoTData import SequenceDataset
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init
import sys
import os
import copy
from tqdm import tqdm
from options import args_parser
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
from transformer import TimeSeriesTransformer
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import matplotlib.pyplot as plt

# save results to txt log file.
stdoutOrigin = sys.stdout 


def main():
    args = args_parser()
    args.device = get_device()
    sys.stdout = open("log/FedAvg"+str(args.model)+".txt", "a")
    
    client_dataset = {}
    for c in range(args.client):
        client_dataset[c] = {}
        train_loader_private, trainset_shared, val_loader, test_loader, dataset_len = get_shared_dataset(c, args.dataset)
        client_dataset[c]["train_private"] = train_loader_private
        client_dataset[c]["train_shared"] = trainset_shared
        client_dataset[c]["val"] = val_loader
        client_dataset[c]["test"] = test_loader
        client_dataset[c]["len"] = dataset_len
    print("Loaded client dataset.")

    ############################
    # loading shared model
    glob_model, clust_weight_keys = model_init(args)
    glob_model = to_device(glob_model, args.device)
    net_keys = [*glob_model.state_dict().keys()]

    ############################
    # generate list of local models for each user
    local_weights = {}      # client ID: weight: val
    for c in range(args.client):
        w_local_dict = {}
        for key in glob_model.state_dict().keys():
            w_local_dict[key] = glob_model.state_dict()[key]
        local_weights[c] = w_local_dict
    print("Loaded client models.")

    ############################
    # simple training algorithm
    if "train" in args.mode:
        train_loss = []         # glob train loss 
        eval_loss = []          # glob eval loss 

        # one communication round 
        for ix_epoch in range(1, args.epoch+1): 
            local_loss = []     # loss for all clients in this round
            glob_weight = {}    # glob weights in this round
            total_len = 0       # total dataset length

            m = max(int(args.frac * args.client), 1) # a fraction of all devices
            if ix_epoch == args.epoch: # in the last round, all users are selected 
                m = args.client
            idxs_users = np.random.choice(range(args.client), m, replace=False)          # select devices for this round
            print(f"Communication round  {ix_epoch}\n---------")
            print("Selected:", idxs_users)
            
            try:
                for c_ind, c in enumerate(idxs_users):  # client update iterations
                    local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                    net_local = copy.deepcopy(glob_model)   # init local net
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)
                    last = (ix_epoch == args.epoch)

                    w_local, loss, idx = local.train(net=net_local.to(args.device), 
                                                     idx=c, w_glob_keys=None, 
                                                     lr=args.max_lr, last=last)
                    local_loss.append(copy.deepcopy(loss))
                    total_len += client_dataset[c]["len"][0]

                    # update shared glob weights
                    if len(glob_weight) == 0:    # first update
                        glob_weight = copy.deepcopy(w_local)
                        for key in glob_model.state_dict().keys():
                            glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0]
                            local_weights[c][key] = w_local[key]
                    else:
                        for key in glob_model.state_dict().keys():
                            glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
                            local_weights[c][key] = w_local[key]

                    print(ix_epoch, idx, loss)

            except KeyboardInterrupt:
                break
            
            for k in glob_model.state_dict().keys():    # get weighted average for global weights
                glob_weight[k] = torch.div(glob_weight[k], total_len)
            
            w_local = glob_model.state_dict()
            for k in glob_weight.keys():
                w_local[k] = glob_weight[k]
            
            if args.epoch != ix_epoch:
                glob_model.load_state_dict(glob_weight)

            loss_avg = sum(local_loss) / len(local_loss)
            train_loss.append(loss_avg)

            for c in range(args.client):
                glob_model.load_state_dict(local_weights[c])
                local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c) 
                w_local, loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)  
                local_loss.append(copy.deepcopy(loss))
            eval_loss.append(sum(local_loss)/len(local_loss))
            print(sum(local_loss)/len(local_loss))

        model_path = "/hdd/saved_models/"
        save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_FedAvg", ix_epoch)

    else:
        print("FedAvg main: Mode should be set to `train`.")




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
        sys.stdout.close()
        sys.stdout = stdoutOrigin
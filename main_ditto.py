"""
The main file for training and evaluating Ditto.
"""
# modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# and: https://github.com/lgcollins/FedRep

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import os
import copy
import torch
import numpy as np
from options import args_parser
from utils.update import LocalUpdateDitto
from utils_training import get_device, to_device, save_model, get_shared_dataset, model_init
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import matplotlib.pyplot as plt

## save results to txt log file.
stdoutOrigin=sys.stdout 



def main():
    args = args_parser()
    args.device = get_device()
    sys.stdout = open("log/Ditto"+str(args.model)+".txt", "a")

    ############################
    # Load client dataset.
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
    # Load shared model.
    glob_model, _ = model_init(args)
    glob_model = to_device(glob_model, args.device)
    net_keys = [*glob_model.state_dict().keys()]

    ############################
    # Generate a list of local weights.
    local_weights = {}      # client ID: weight: val
    for c in range(args.client):
        w_local_dict = {}
        for key in glob_model.state_dict().keys():
            w_local_dict[key] = glob_model.state_dict()[key]
        local_weights[c] = w_local_dict
    print("Loaded client models.")

    ############################
    # Simple training rounds for Ditto.
    train_loss = []         # glob training loss 
    eval_loss = []          # glob eval loss 

    for ix_epoch in range(1, args.epoch+1): # one communication round 
        local_loss = []     # loss for all local clients in this round.
        glob_weight = {}    # glob weights in this round

        m = max(int(args.frac * args.client), 1) # a fraction of all devices
        if ix_epoch == args.epoch:               # In the last round, all users are selected 
            m = args.client
        # Select participating clients in this round.
        idxs_users = np.random.choice(range(args.client), m, replace=False)          
        
        try:
            print(f"Communication round {ix_epoch}\n---------")
            print("Selected:", idxs_users)

            for c_ind, c in enumerate(idxs_users):    # client update iterations
                last = (ix_epoch == args.epoch)
                local = LocalUpdateDitto(args=args, dataset=client_dataset[c], idxs=c)
                net_local = copy.deepcopy(glob_model)       # init local net with global weights
                w_glob_k = copy.deepcopy(net_local.state_dict())
                # train a client for some epochs
                w_k, loss, idx = local.train(net=net_local.to(args.device), 
                                            idx=c, w_glob_keys=None, 
                                            lr=args.max_lr, last=last, 
                                            w_ditto=None)   # train global
                w_local = copy.deepcopy(local_weights[c])   # load local weights
                net_local.load_state_dict(w_local)          # load local state dict
                w_local, loss, idx = local.train(net=net_local.to(args.device), 
                                                    idx=c, w_glob_keys=None, 
                                                    lr=args.max_lr, last=last, 
                                                    w_ditto=w_glob_k) # train local
                local_loss.append(copy.deepcopy(loss))

                # update shared glob weights
                if len(glob_weight) == 0:    # first update
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] = w_k[key]/m
                        local_weights[c][key] = w_local[key]
                else:
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] += w_k[key]/m
                        local_weights[c][key] = w_local[key]
                print(ix_epoch, idx, loss)

        except KeyboardInterrupt:   # break when keyboard interrupt
            break

        loss_avg = sum(local_loss) / len(local_loss)
        train_loss.append(loss_avg)
        glob_model.load_state_dict(glob_weight)

        for c in range(args.client):
            glob_model.load_state_dict(local_weights[c])
            local = LocalUpdateDitto(args=args, dataset=client_dataset[c], idxs=c) 
            w_local, loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)  
            local_loss.append(copy.deepcopy(loss))
        eval_loss.append(sum(local_loss)/len(local_loss))
        print(sum(local_loss)/len(local_loss))

    model_path = "/hdd/saved_models/"
    save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_Ditto", ix_epoch)
    for c in range(args.client):
        glob_model.load_state_dict(local_weights[c])
        model_path = "/hdd/saved_models/"
        save_model(model_path, net_local, str(args.dataset)+"_"+str(args.model)+"_Ditto_"+str(c), ix_epoch)





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
        sys.stdout.close()
        sys.stdout=stdoutOrigin
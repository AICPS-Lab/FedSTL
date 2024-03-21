"""
The main file for training and evaluating FedSTL format
with options to compare with other benchmarks.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from IoTData import SequenceDataset
from utils.update import LocalUpdate, LocalUpdateProp, compute_cluster_id, cluster_id_property, cluster_explore
from utils_training import get_device, to_device, save_model, get_client_dataset, get_shared_dataset, model_init
import copy
from tqdm import tqdm
from options import args_parser
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
import matplotlib.pyplot as plt

## save results to txt log file.
stdoutOrigin=sys.stdout


def get_dict_keys(cluster_id, idxs_users):
    d = {}
    for i in idxs_users:
        for key, val in cluster_id.items():
            if i in val:
                d[i] = key
    return d




def main():
    args = args_parser()
    args.device = get_device()
    sys.stdout = open("log/FedSTL"+str(args.model)+".txt", "a")
    
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
    # training with clusters
    if "train" in args.mode and args.cluster > 0:
        
        cluster_weights = {}
        for cluster in range(args.cluster):
            cluster_weights[cluster] = {}

        cluster_models = {}
        for cluster in range(args.cluster):
            cluster_models[cluster] = copy.deepcopy(glob_model)

        train_loss = [] # glob train loss 
        eval_loss = []  # glob eval loss 
        
        for ix_epoch in range(1, args.epoch+1):
            local_loss = []             # loss for all clients in this round
            glob_weight = {}            # glob weights in this round
            total_len = 0               # total dataset length
            cluster_len = [0] * args.cluster

            # a fraction of all devices
            m = max(int(args.frac * args.client), 1)        
            if ix_epoch == args.epoch:                          
                m = args.client
            idxs_users = np.random.choice(range(args.client), m, replace=False)  
            print(f"Communication round  {ix_epoch}\n---------")
            print("Selected:", idxs_users)     

            last = (ix_epoch == args.epoch)
            if last: 
                args.frac = 1
            cluster_id = cluster_id_property(cluster_models, client_dataset, args, idxs_users)  # cluster: clients
            client2cluster = get_dict_keys(cluster_id, idxs_users)                              # client:  cluster
            
            for c in idxs_users:
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                net_local = copy.deepcopy(cluster_models[client2cluster[c]]) 
                w_local = net_local.state_dict()
                for k in local_weights[c].keys():
                    if k not in clust_weight_keys:
                        w_local[k] = local_weights[c][k]
                net_local.load_state_dict(w_local)

                w_local, loss, idx = local.train(net=net_local.to(args.device), 
                                                 idx=c, w_glob_keys=clust_weight_keys, 
                                                 lr=args.max_lr, last=last)
                local_loss.append(copy.deepcopy(loss))
                total_len += client_dataset[c]["len"][0]
                cluster_len[client2cluster[c]] += client_dataset[c]["len"][0]

                # update shared cluster weights
                if len(cluster_weights[client2cluster[c]]) == 0:
                    cluster_temp = copy.deepcopy(w_local)
                    for k in clust_weight_keys:
                        cluster_weights[client2cluster[c]][k] = cluster_temp[k]*client_dataset[c]["len"][0]
                else:
                    for k in clust_weight_keys:
                        cluster_weights[client2cluster[c]][k] += w_local[k]*client_dataset[c]["len"][0]

                # update shared glob weights
                if len(glob_weight) == 0: 
                    glob_weight = copy.deepcopy(w_local)
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] = glob_weight[key]*client_dataset[c]["len"][0] # weighted update
                        local_weights[c][key] = w_local[key]
                else:
                    for key in glob_model.state_dict().keys():
                        glob_weight[key] += w_local[key]*client_dataset[c]["len"][0]
                        local_weights[c][key] = w_local[key]
                print(ix_epoch, idx, loss)
            
            # get weighted average global weights
            for k in glob_model.state_dict().keys():    
                glob_weight[k] = torch.div(glob_weight[k], total_len)

            # update locals 
            w_local = glob_model.state_dict()
            for k in glob_weight.keys():
                w_local[k] = glob_weight[k]
            
            # update globals
            if args.epoch != ix_epoch:
                glob_model.load_state_dict(glob_weight)

            # update clusters 
            for cluster in range(args.cluster):
                for k in clust_weight_keys:
                    cluster_weights[cluster][k] = torch.div(cluster_weights[cluster][k], cluster_len[cluster])

            for cluster in range(args.cluster):
                w_local = net_local.state_dict()
                for k in glob_weight.keys():
                    if k not in clust_weight_keys:
                        w_local[k] = glob_weight[k]
                    else:
                        w_local[k] = cluster_weights[cluster][k]
                cluster_models[cluster].load_state_dict(w_local)
        

            if args.mode == "train-logic":          # cluster model fine-tune
                for cluster in range(args.cluster):
                    net_local = copy.deepcopy(cluster_models[cluster]) 
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)

                    data_x = [client_dataset[i]["train_shared"][0] for i in cluster_id[cluster]]
                    data_y = [client_dataset[i]["train_shared"][1] for i in cluster_id[cluster]]
                    cluster_data_x = np.concatenate(data_x, axis=0)
                    cluster_data_y = np.concatenate(data_y, axis=0)
                    cluster_train_dataset = SequenceDataset(cluster_data_x, cluster_data_y)
                    cluster_train_loader = DataLoader(cluster_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

                    w_local, loss = cluster_explore(net=net_local.to(args.device), 
                                                    w_glob_keys=clust_weight_keys, 
                                                    lr=args.max_lr, args=args, 
                                                    dataloaders=cluster_train_loader)
                    for key in glob_weight.keys():
                        cluster_weights[cluster][key] = w_local[key]
                    cluster_models[cluster].load_state_dict(w_local)
                    print("cluster fine-tune:", ix_epoch, cluster, loss)
        
            loss_avg = sum(local_loss)/len(local_loss)
            train_loss.append(loss_avg)

            for c in range(args.client):
                glob_model.load_state_dict(local_weights[c])
                local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                w_local, loss, cons_loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)   
                local_loss.append(copy.deepcopy(loss))
            eval_loss.append(sum(local_loss)/len(local_loss))
            print(sum(local_loss)/len(local_loss))

        model_path = "/hdd/saved_models/"
        glob_model.load_state_dict(glob_weight)
        save_model(model_path, glob_model, 
                   str(args.dataset)+"_"+str(args.local_updates)+"_"+str(args.model)+"_FedSTL_"+str(args.property_type), 
                   ix_epoch)
        
        for c in range(args.cluster):
            glob_model.load_state_dict(cluster_weights[c])
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_FedSTL_Cluster_"+str(c), ix_epoch)
        
        for c in range(args.client):
            glob_model.load_state_dict(local_weights[c])
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_FedSTL_Client_"+str(c), ix_epoch)


    ############################
    # simple training without clusters
    elif "train" in args.mode:
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
            
            try:
                print(f"Communication round  {ix_epoch}\n---------")
                print("Selected:", idxs_users)

                # client update iterations
                for c_ind, c in enumerate(idxs_users):
                    if args.mode == "train":
                        local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)   # init local update modules
                    elif args.mode == "train-logic":
                        local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                    
                    net_local = copy.deepcopy(glob_model)   # init local net
                    w_local = net_local.state_dict()
                    net_local.load_state_dict(w_local)
                    last = (ix_epoch == args.epoch)

                    w_local, loss, idx = local.train(net=net_local.to(args.device), idx=c, w_glob_keys=None, lr=args.max_lr, last=last) # train local
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
                if args.mode == "train": 
                    local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c) 
                    ## TODO: commented out for now
                    w_local, loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)  
                elif args.mode == "train-logic":           
                    local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
                    w_local, loss, cons_loss, idx = local.test(net=glob_model.to(args.device), idx=c, w_glob_keys=None)   
                local_loss.append(copy.deepcopy(loss))
            eval_loss.append(sum(local_loss)/len(local_loss))

            print(sum(local_loss)/len(local_loss))

        model_path = "/hdd/saved_models/"
        if args.mode == "train":
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_glob", ix_epoch)
        elif args.mode == "train-logic":
            save_model(model_path, glob_model, str(args.dataset)+"_"+str(args.model)+"_"+str(args.property_type)+"_glob", ix_epoch)



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    
    finally:
        print('\nDone.')
        sys.stdout.close()
        sys.stdout=stdoutOrigin
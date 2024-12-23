#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819

import sys
import copy
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from .transformer_utils import generate_square_subsequent_mask
sys.path.append("..")
from find_trace import *
from generate_STL import generate_property, generate_property_test, get_robustness_score




def property_loss_simp(y_pred, property, loss_function):
    return torch.sum(loss_function(y_pred - property))



def property_loss_eventually(y_pred, property, loss_function, type):
    iterval = 2
    if type == "eventually-upper":
        diff_yp = y_pred - property
        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
        diff_min, ind = torch.min(loss_function(unsqueezed_diff), dim=2)
        return torch.sum(diff_min)
    elif type == "eventually-lower":
        diff_yp = property - y_pred
        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
        diff_min, ind = torch.min(loss_function(unsqueezed_diff), dim=2)
        return torch.sum(diff_min)



def property_loss(X, y_pred, property_by_station_day, loss_function, y, show):
    loss = torch.zeros(1).to('cuda')
    for ind, arr in enumerate(X):   # X shape: 64, 120, 3
        sensor = int(arr[-1,2].item())
        day = int(arr[-1,1].item())
        day = (day+1)%7
        loss += torch.sum(loss_function(y_pred[ind] - torch.tensor(property_by_station_day[sensor][day][0]).to('cuda')))
        loss += torch.sum(loss_function(torch.tensor(property_by_station_day[sensor][day][1]).to('cuda') - y_pred[ind]))
    if show:
        plt.plot(property_by_station_day[sensor][day][0])
        plt.plot(property_by_station_day[sensor][day][1])
        plt.plot(y[ind].detach().cpu())
        plt.plot(y_pred[ind].detach().cpu())
        plt.show()
    return loss



def repackage_hidden(h):

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)




def transformer_prox_train(dataloader, net, args, loss_func, lr, server_model=None):
    
    net.train()
    mu = 1

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()

    epoch_loss = []
    for it in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            w_0 = copy.deepcopy(net.state_dict())
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            if it>0:
                w_diff = torch.tensor(0., device=device)
                for w, w_t in zip(server_model.parameters(), net.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)
                loss += mu / 2. * w_diff

            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)




def transformer_ditto_train(dataloader, net, args, loss_func, lr, w_ditto=None, lam=1):
    
    net.train()

    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            w_0 = copy.deepcopy(net.state_dict())
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            if w_ditto is not None:
                w_net = copy.deepcopy(net.state_dict())
                for key in w_net.keys():
                    w_net[key] = w_net[key] - lr*lam*(w_0[key]-w_ditto[key])
                net.load_state_dict(w_net)
                optimizer.zero_grad()

            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)




def transformer_prop_train(dataloader, net, args, loss_func, lr, w_glob_keys=None):
    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()

    epoch_loss = []
    epoch_cons_loss = []
    for iter in range(local_eps):
        num_updates = 0
        for name, param in net.named_parameters():
            param.requires_grad = True 
        
        batch_loss = []
        batch_cons_loss = []
        for X, y in dataloader: ## batch first=True
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            pred_loss = loss_func(output.view(-1, 24), y)

            if args.property_type == 'constraint':
                property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)

            loss = pred_loss + cons_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            num_updates += 1
            batch_loss.append(loss.item())
            batch_cons_loss.append(cons_loss.item())
            
            if num_updates == args.local_updates: break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)



def transformer_prop_teacher_test(dataloader, net, args, loss_func, rho=False):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()
        
    for name, param in net.named_parameters():
        param.requires_grad = False 

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    for X, y in dataloader:
        net.eval()
        X = X.unsqueeze(2)
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        output = output.view(-1, 24)
    
        if args.property_type == 'constraint':
            property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
            teacher_pred = torch.min(output, corrected_trace_upper)
            teacher_pred = torch.max(output, corrected_trace_lower)
            output = teacher_pred
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
        elif args.property_type == 'eventually':
            property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
            property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
            cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
        else:
            raise NotImplementedError
        pred_loss = loss_func(output, y)
        batch_loss.append(pred_loss.item())
        batch_cons_loss.append(cons_loss.item())

        if rho==True:
            if args.property_type == 'constraint':
                batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_upper - output))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_lower))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                
            elif args.property_type == 'eventually':
                iterval = 2
                diff_yp = output - property_upper
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                diff_yp = property_lower - output
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
            else:
                raise NotImplementedError
    
    if rho==True:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss), sum(batch_rho)/len(batch_rho)
    else:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss)





def transformer_prop_test(dataloader, net, args, loss_func, rho=False):
    net.eval()
    m = nn.ReLU()
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()
        
    for name, param in net.named_parameters():
        param.requires_grad = False 

    batch_rho = []
    batch_loss = []
    batch_cons_loss = []
    
    for X, y in dataloader:
        net.eval()
        X = X.unsqueeze(2)
        output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
        
        if args.property_type == 'constraint':
            property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
            corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
            property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
            corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
            cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
        elif args.property_type == 'eventually':
            property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
            property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
            cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
        else:
            raise NotImplementedError
        
        pred_loss = loss_func(output.view(-1, 24), y)
        batch_loss.append(pred_loss.item())
        batch_cons_loss.append(cons_loss.item())

        if rho==True:
            if args.property_type == 'constraint':
                batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
            
            elif args.property_type == 'eventually':
                iterval = 2
                diff_yp = output - property_upper
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                diff_yp = property_lower - output
                unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
            else:
                raise NotImplementedError
    
    if rho==True:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss), sum(batch_rho)/len(batch_rho)
    else:
        return net, sum(batch_loss)/len(batch_loss), sum(batch_cons_loss)/len(batch_cons_loss)




def transformer_train(dataloader, net, args, loss_func, lr, w_glob_keys=None):
    bias_p, weight_p = [], []
    for name, p in net.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    
    optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
    local_eps = args.client_iter
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        if args.method == 'FedRep' and iter < args.head_iter:
            for name, param in net.named_parameters():
                if name in w_glob_keys:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif args.method == 'FedRep' and iter >= args.head_iter:
            for name, param in net.named_parameters():
                if name in w_glob_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in net.named_parameters():
                param.requires_grad = True 
        
        batch_loss = []
        for batch_idx, (X, y) in enumerate(dataloader): ## batch first=True
            net.train()
            optimizer.zero_grad()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)




def transformer_test(dataloader, net, args, loss_func):
    net.eval()

    local_eps = args.client_iter  # local update epochs
    src_mask = generate_square_subsequent_mask(dim1=24, dim2=120).cuda()
    tgt_mask = generate_square_subsequent_mask(dim1=24, dim2=24).cuda()

    epoch_loss = []
    for iter in range(local_eps):
        num_updates = 0
        
        for name, param in net.named_parameters():
            param.requires_grad = False 
        
        batch_loss = []
        for X, y in dataloader:
            net.eval()
            X = X.unsqueeze(2)
            output = net(src=X, tgt=X[:,-24:,:], src_mask=src_mask, tgt_mask=tgt_mask)
            loss = loss_func(output.view(-1, 24), y)
            
            num_updates += 1
            batch_loss.append(loss.item())
            
            if num_updates == args.local_updates:
                break

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    return net, sum(epoch_loss)/len(epoch_loss)



class LocalUpdateProx(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs  # client index

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, mu=0.1, server_model=None):
        
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prox_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, server_model)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([
                {'params': weight_p, 'weight_decay':0.0001}, 
                {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            
            local_eps = self.args.client_iter  # local update epochs
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = True
            
            for it in range(local_eps):   # for # total local ep
                num_updates = 0
                batch_loss = []

                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)

                    if it>0:
                        w_diff = torch.tensor(0., device=device)
                        for w, w_t in zip(server_model.parameters(), net.parameters()):
                            w_diff += torch.pow(torch.norm(w - w_t), 2)
                        loss += mu / 2. * w_diff

                    loss.backward()
                    optimizer.step()
                    num_updates += 1
                    batch_loss.append(loss.item())
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
    
    
    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):
        
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_test(self.ldr_train, net, self.args, self.loss_func)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.eval()
            epoch_loss = []
            num_updates = 0
            
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False 

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []
            for X, y in self.ldr_train:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                num_updates += 1
                batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs  # client index

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, w_ditto=None, lam=1):
        
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_ditto_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_ditto=w_ditto, lam=lam)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.train()
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([
                {'params': weight_p, 'weight_decay':0.0001}, 
                {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            
            local_eps = self.args.client_iter  # local update epochs
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = True
            
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0
                batch_loss = []

                for batch_idx, (X, y) in enumerate(self.ldr_train):
                    w_0 = copy.deepcopy(net.state_dict())
                    net.train()
                    net.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    if w_ditto is not None:
                        w_net = copy.deepcopy(net.state_dict())
                        for key in w_net.keys():
                            w_net[key] = w_net[key] - lr*lam*(w_0[key]-w_ditto[key])
                        net.load_state_dict(w_net)
                        optimizer.zero_grad()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs
        

    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):

        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_test(self.ldr_train, net, self.args, self.loss_func)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            net.eval()
            epoch_loss = []
            num_updates = 0
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False 

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []
            for X, y in self.ldr_train:
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                num_updates += 1
                batch_loss.append(loss.item())

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs




# local updates for FedRep, FedAvg
class LocalUpdate(object):
    """
    Federated learning updating class for a single agent.
    """
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs


    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            local_eps = self.args.client_iter  # local update epochs
            
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0
                
                if self.args.method == 'FedRep' and iter < self.args.head_iter:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                elif self.args.method == 'FedRep' and iter >= self.args.head_iter:
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    for name, param in net.named_parameters():
                        param.requires_grad = True 

                batch_loss = []
                for X, y in self.ldr_train:
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



    def train_cluster(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_train(self.ldr_train, net, self.args, self.loss_func, self.args.max_lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]
            
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            local_eps = self.args.client_iter  # local update epochs
            epoch_loss = []
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):   # for # total local ep
                num_updates = 0

                for name, param in net.named_parameters():
                    param.requires_grad = True 

                batch_loss = []
                for X, y in self.ldr_train:
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    loss = self.loss_func(output, y)
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(loss.item())
                    
                    if num_updates == self.args.local_updates:
                        break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



    def test(self, net, w_glob_keys, dataset_test=None, ind=-1, idx=-1):
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_test(self.ldr_train, net, self.args, self.loss_func)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            epoch_loss = []
            num_updates = 0

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_loss = []
            batch_y_test = []
            batch_pred_test = []
            for X, y in self.ldr_train:
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                loss = self.loss_func(output, y)

                batch_y_test.append(y.detach().cpu().numpy())
                batch_pred_test.append(output.detach().cpu().numpy())
                
                net.zero_grad()
                num_updates += 1
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs





def compute_cluster_id(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)
    
    # load cluster models 
    for cluster in range(args.cluster):
        for c in idxs_users:
            local = LocalUpdate(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []

    # cluster_id[cluster] = [0]
    # return cluster_id
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < int(args.client*args.frac/args.cluster) and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id



def compute_cluster_id_eval(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)
    
    # load cluster models 
    for cluster in range(args.cluster):
        print(cluster)
        for c in idxs_users:
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, cons_loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < int(args.client*args.frac/args.cluster) and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id




def cluster_id_property(cluster_models, client_dataset, args, idxs_users):
    cluster_loss = np.full((args.cluster, args.client), np.inf)
    
    for cluster in range(args.cluster):
        print(cluster)
        for c in idxs_users:
            local = LocalUpdateProp(args=args, dataset=client_dataset[c], idxs=c)
            w_local, loss, cons_loss, idx = local.test(net=cluster_models[cluster] .to(args.device), idx=c, w_glob_keys=None)
            cluster_loss[cluster][c] = cons_loss
    
    cluster_id = {}
    for cluster in range(args.cluster):
        cluster_id[cluster] = []
    
    cluster_id[cluster] = [0]
    return cluster_id

    
    client_lst = [c for c in idxs_users]
    i = 0
    while i < len(idxs_users):
        print(i)
        min_index = np.argwhere(cluster_loss == np.min(cluster_loss))
        if len(cluster_id[min_index[0][0]]) < int(args.client*args.frac/args.cluster) and min_index[0][1] in client_lst:
            cluster_id[min_index[0][0]].append(min_index[0][1])
            i += 1
            client_lst.remove(min_index[0][1])
        cluster_loss[min_index[0][0], min_index[0][1]] = np.inf

    return cluster_id



def cluster_explore(net, w_glob_keys, lr, args, dataloaders):

    loss_func = nn.MSELoss()

    if net.model_type == 'transformer':
        net, avg_ep_loss = transformer_prop_train(dataloaders, net, args, loss_func, lr, w_glob_keys=w_glob_keys)
        return net.state_dict(), avg_ep_loss
    
    if net.model_type != 'transformer':
        bias_p, weight_p = [], []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        m = nn.ReLU()
        optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, 
                                    {'params': bias_p, 'weight_decay':0}], 
                                    lr=lr, momentum=0.5)
        epoch_loss = []
        epoch_cons_loss = []
        hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
        for name, param in net.named_parameters():
            if name in w_glob_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for iter in range(args.cluster_fine_tune_iter):
            num_updates = 0
            batch_loss = []
            batch_cons_loss = []
            for batch_idx, (X, y) in enumerate(dataloaders):
                net.train()
                optimizer.zero_grad()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                pred_loss = loss_func(output, y)
                
                if args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    cons_loss = loss_func(output, corrected_trace_upper) + loss_func(output, corrected_trace_lower)
                
                elif args.property_type == 'eventually':
                    property_upper = generate_property_test(X, property_type = "eventually-upper")
                    property_lower = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, m, "eventually-upper") + property_loss_eventually(output, property_lower, m, "eventually-lower")
                
                else:
                    raise NotImplementedError
                
                loss = pred_loss + cons_loss
                loss.backward()
                optimizer.step()

                num_updates += 1
                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss)
                
                if num_updates == args.local_updates: break
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
    
    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)



class LocalUpdateProp(object):
    """
    Federated learning updating class for a single agent with property mining.
    """
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.MSELoss(reduction='mean')
        self.m = nn.ReLU()
        self.ldr_train = dataset["train_private"]
        self.ldr_val = dataset["val"]
        self.ldr_test = dataset["test"]
        self.idxs = idxs


    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001):
        if net.model_type == 'transformer':
            net, avg_ep_loss = transformer_prop_train(self.ldr_train, net, self.args, self.loss_func, lr, w_glob_keys=w_glob_keys)
            return net.state_dict(), avg_ep_loss, self.idxs
        
        else:
            bias_p, weight_p = [], []
            for name, p in net.named_parameters():
                if 'bias' in name: 
                    bias_p += [p]
                else: 
                    weight_p += [p]
        
            optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay':0.0001}, {'params': bias_p, 'weight_decay':0}], lr=lr, momentum=0.5)
            local_eps = self.args.client_iter   # local update epochs
            
            epoch_loss = []
            epoch_cons_loss = []

            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()
            
            for iter in range(local_eps):
                num_updates = 0
                
                for name, param in net.named_parameters():
                    param.requires_grad = True 

                batch_loss = []
                batch_cons_loss = []
                for X, y in self.ldr_train:
                    net.train()
                    optimizer.zero_grad()
                    hidden_1 = repackage_hidden(hidden_1)
                    hidden_2 = repackage_hidden(hidden_2)
                    output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                    pred_loss = self.loss_func(output, y)
                    
                    if self.args.property_type == 'constraint':
                        property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                        corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                        property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                        corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                        cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)

                    elif self.args.property_type == 'eventually':
                        property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                        property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                        cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                    
                    else:
                        raise NotImplementedError
                    
                    loss = pred_loss + cons_loss
                    loss.backward()
                    optimizer.step()

                    num_updates += 1
                    batch_loss.append(pred_loss.item())
                    batch_cons_loss.append(cons_loss.item())
                    
                    if num_updates == self.args.local_updates: break

                epoch_loss.append(sum(batch_loss)/len(batch_loss))
                epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
        
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.idxs



    def test(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False):
        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho= transformer_prop_test(self.ldr_train, net, self.args, self.loss_func, rho=True)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_test(self.ldr_train, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs

        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            
            for X, y in self.ldr_train:
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                pred_loss = self.loss_func(output, y)
                batch_loss.append(pred_loss.item())
                
                if self.args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)

                elif self.args.property_type == 'eventually':
                    property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                    property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                else:
                    raise NotImplementedError
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_lower - output))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_upper))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                    
                    elif self.args.property_type == 'eventually':
                        iterval = 2
                        diff_yp = output - property_upper
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                        diff_yp = property_lower - output
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                    else:
                        raise NotImplementedError

                num_updates += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho)
        else:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs
        

    def test_teacher(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.001, rho=False):
        m = nn.ReLU()
        epoch_rho = []
        epoch_loss = []
        epoch_cons_loss = []
        num_updates = 0

        if net.model_type == 'transformer':
            if rho == True:
                net, ep_ls, ep_cons_ls, ep_rho= transformer_prop_teacher_test(self.ldr_val, net, self.args, self.loss_func, rho=True)
                return ep_ls, ep_cons_ls, self.idxs, ep_rho
            else:
                net, ep_ls, ep_cons_ls = transformer_prop_teacher_test(self.ldr_val, net, self.args, self.loss_func, rho=False)
                return net.state_dict(), ep_ls, ep_cons_ls, self.idxs
        
        else:
            hidden_1, hidden_2 = net.init_hidden(), net.init_hidden()

            for name, param in net.named_parameters():
                param.requires_grad = False

            batch_rho = []
            batch_loss = []
            batch_cons_loss = []
            for X, y in self.ldr_val:
                net.eval()
                hidden_1 = repackage_hidden(hidden_1)
                hidden_2 = repackage_hidden(hidden_2)
                output, hidden_1, hidden_2 = net(X, hidden_1, hidden_2)
                
                if self.args.property_type == 'constraint':
                    property_upper, stl_lib_upper = generate_property_test(X, property_type = "upper")
                    corrected_trace_upper = convert_best_trace(stl_lib_upper, output)
                    property_lower, stl_lib_lower = generate_property_test(X, property_type = "lower")
                    corrected_trace_lower = convert_best_trace(stl_lib_lower, output)
                    teacher_pred = torch.min(output, corrected_trace_upper)
                    teacher_pred = torch.max(output, corrected_trace_lower)
                    output = teacher_pred
                    cons_loss = self.loss_func(output, corrected_trace_upper) + self.loss_func(output, corrected_trace_lower)
                elif self.args.property_type == 'eventually':
                    property_upper, _ = generate_property_test(X, property_type = "eventually-upper")
                    property_lower, _ = generate_property_test(X, property_type = "eventually-lower")
                    cons_loss = property_loss_eventually(output, property_upper, self.m, "eventually-upper") + property_loss_eventually(output, property_lower, self.m, "eventually-lower")
                else:
                    raise NotImplementedError
                pred_loss = self.loss_func(output, y)
                batch_loss.append(pred_loss.item())
                batch_cons_loss.append(cons_loss.item())

                if rho==True:
                    if self.args.property_type == 'constraint':
                        batch_rho.append( 1-torch.count_nonzero(m(corrected_trace_upper - output))/len(corrected_trace_upper)/corrected_trace_upper.shape[1] )
                        batch_rho.append( 1-torch.count_nonzero(m(output - corrected_trace_lower))/len(corrected_trace_lower)/corrected_trace_lower.shape[1] )
                    
                    elif self.args.property_type == 'eventually':
                        iterval = 2
                        diff_yp = output - property_upper
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                        diff_yp = property_lower - output
                        unsqueezed_diff = diff_yp.view(diff_yp.shape[0], int(diff_yp.shape[1]//iterval), iterval)
                        diff_min, ind = torch.min(m(unsqueezed_diff), dim=2)
                        batch_rho.append( 1-torch.count_nonzero(diff_min) / len(diff_min) / diff_min.shape[1] )
                    else:
                        raise NotImplementedError
                num_updates += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_cons_loss.append(sum(batch_cons_loss)/len(batch_cons_loss))
            if rho==True:
                epoch_rho.append(sum(batch_rho)/len(batch_rho))
            
        if rho==True:
            return sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs, sum(epoch_rho)/len(epoch_rho)
        else:
            return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_cons_loss)/len(epoch_cons_loss), self.idxs

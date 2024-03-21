import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from generate_STL import generate_property
import os
from tqdm import tqdm
from options import args_parser
from model import ShallowRegressionLSTM
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


import matplotlib.pyplot as plt



def generate_train_property(arr):
    """
    not used
    """
    data_by_station = {}
    for idx, line in enumerate(arr):
        try:
            data_by_station[line[0,2]] = np.concatenate((data_by_station[line[0,2]], line), axis=0)
        except KeyError:
            data_by_station[line[0,2]] = line

    for key, item in data_by_station.items():
        tensor_sep = item.reshape(int(item.shape[0]/24), 24, 3)
        data_by_station[key] = tensor_sep[:, :, :2]

    data_by_station_day = {}    # dict: station - day - property
    for key, item in data_by_station.items():
        station = {}
        for i in range(7):
            station[i] = []
        for signal in item:
            station[signal[0,1]].append(signal[:,0])
        for key_2, item_2 in station.items():
            station[key_2] = np.array(item_2)
        data_by_station_day[key] = station
    
    property_by_station_day = {}
    for station_id, station_data in data_by_station_day.items():
        daily_property = {}
        for day, daily_data in station_data.items():
            conf_interval, property_upper = generate_property(daily_data, property_type = "value-upper", mining_range = 2)
            _, property_lower = generate_property(daily_data, property_type = "value-lower", mining_range = 2)
            daily_property[day] = np.stack((property_upper, property_lower), axis=0)
        property_by_station_day[station_id] = daily_property
    
    return property_by_station_day



def generate_train_property(arr):
    """
    not used
    """
    data_by_station = {}
    for idx, line in enumerate(arr):
        try:
            data_by_station[line[0,2]] = np.concatenate((data_by_station[line[0,2]], line), axis=0)
        except KeyError:
            data_by_station[line[0,2]] = line

    for key, item in data_by_station.items():
        tensor_sep = item.reshape(int(item.shape[0]/24), 24, 3)
        data_by_station[key] = tensor_sep[:, :, :2]

    data_by_station_day = {}    # dict: station - day - property
    for key, item in data_by_station.items():
        station = {}
        for i in range(7):
            station[i] = []
        for signal in item:
            station[signal[0,1]].append(signal[:,0])
        for key_2, item_2 in station.items():
            station[key_2] = np.array(item_2)
        data_by_station_day[key] = station
    
    property_by_station_day = {}
    for station_id, station_data in data_by_station_day.items():
        daily_property = {}
        for day, daily_data in station_data.items():
            conf_interval, property_upper = generate_property(daily_data, property_type = "value-upper", mining_range = 2)
            _, property_lower = generate_property(daily_data, property_type = "value-lower", mining_range = 2)
            daily_property[day] = np.stack((property_upper, property_lower), axis=0)
        property_by_station_day[station_id] = daily_property
    
    return property_by_station_day



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)


class ToDeviceLoader:
    """
    Wrap a dataloader to move data to a device
    """
    def __init__(self,data,device):
        self.data = data
        self.device = device
        
    def __iter__(self):
        for batch in self.data:
            yield to_device(batch,self.device)
            
    def __len__(self):
        """
        Number of batches
        """
        return len(self.data)



def save_model(path, model, model_name, epoch):
    if not os.path.isdir(path):
        os.makedirs(path)
    save_prefix = os.path.join(path, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close() 



class SequenceDataset(Dataset):
    def __init__(self, x, y, device='cuda'):
        self.x = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i): 
        return self.x[i], self.y[i]



def train(model, data_loader, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    for X, y in tqdm(data_loader):
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss





def generate_train_property(arr):
    data_by_station = {}
    for idx, line in enumerate(arr):
        try:
            data_by_station[line[0,2]] = np.concatenate((data_by_station[line[0,2]], line), axis=0)
        except KeyError:
            data_by_station[line[0,2]] = line

    for key, item in data_by_station.items():
        tensor_sep = item.reshape(int(item.shape[0]/24), 24, 3)
        data_by_station[key] = tensor_sep[:, :, :2]

    data_by_station_day = {}    # dict: station - day - property
    for key, item in data_by_station.items():
        station = {}
        for i in range(7):
            station[i] = []
        for signal in item:
            station[signal[0,1]].append(signal[:,0])
        for key_2, item_2 in station.items():
            station[key_2] = np.array(item_2)
        data_by_station_day[key] = station
    
    property_by_station_day = {}
    for station_id, station_data in data_by_station_day.items():
        daily_property = {}
        for day, daily_data in station_data.items():
            conf_interval, property_upper = generate_property(daily_data, property_type = "value-upper", mining_range = 2)
            _, property_lower = generate_property(daily_data, property_type = "value-lower", mining_range = 2)
            # plt.plot(property_upper)
            # plt.plot(property_lower)
            # plt.plot(conf_interval[:,1])
            # plt.plot(conf_interval[:,2])
            # for ln in daily_data:
            #     plt.scatter([i for i in range(24)], ln, s=5)
            # plt.show()
            daily_property[day] = np.stack((property_upper, property_lower), axis=0)
        property_by_station_day[station_id] = daily_property
    
    return property_by_station_day

    

def property_loss_simp(y_pred, property, loss_function):
    return torch.sum(loss_function(y_pred - property))



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




def train_logic(model, data_loader, loss_function, optimizer, property_by_station_day):
    num_batches = len(data_loader)
    total_loss = 0
    rule_loss = 0
    model.train()
    m = nn.ReLU()
    
    count = 0
    for X, y in tqdm(data_loader):
        show = False

        # count+=1
        # if count % 100 == 0:
        #     show = True
        
        output = model(X)
        # property_upper = generate_property(X, property_type = "value-upper")
        # property_lower = generate_property(X, property_type = "value-lower")
        # con_loss1 = property_loss(output, property_upper, m)
        # con_loss2 = property_loss(property_lower, output, m)
        cons_loss = property_loss(X, output, property_by_station_day, m, y, show)
        pred_loss = loss_function(output, y)
        while cons_loss > pred_loss:
            cons_loss = cons_loss/10
        loss = pred_loss + cons_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += pred_loss.item()  # only return the predictive loss for easier comparison
        rule_loss += cons_loss.item() 

        
    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    print(f"Property loss: {rule_loss / num_batches}")
    return avg_loss






def test(model, data_loader, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    gt, y_pred = [], []

    with torch.no_grad():
        for X, y in tqdm(data_loader):
            output = model(X)
            x = [i for i in range(97, 97+24)]
            gt.extend(y.detach().cpu()[0])
            y_pred.extend(output.detach().cpu()[0])

            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Validation loss: {avg_loss}")
    return avg_loss, gt, y_pred




def main():
    # parse args
    args = args_parser()
    args.device = get_device()

    dataset_array = {}
    dataset_path = "/hdd/traffic_data_2019/torch_dataset/"
    for fold in ["train", "test", "val"]:
        # x and y file changes for synthetic data
        x_file = dataset_path+fold+"_x.npy"
        y_file = dataset_path+fold+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    # TODO: dataset too large
    train_dataset = SequenceDataset(dataset_array["train_x"], dataset_array["train_y"])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    if args.mode == "train":
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_loss = []
        eval_loss = []
        for ix_epoch in range(1, args.epoch+1):
            try:
                print(f"Epoch {ix_epoch}\n---------")
                train_loss.append(train(model, train_loader, loss_function, optimizer=optimizer))
                loss_, gt, y_pred = test(model, val_loader, loss_function)
                eval_loss.append(loss_)
                print()
            except KeyboardInterrupt:
                break

        plt.plot(train_loss)
        plt.plot(eval_loss)
        plt.show()

        model_path = "/hdd/traffic_data_2019/run/"
        save_model(model_path, model, "lstm", ix_epoch)


    if args.mode == "train-logic":
        print("--------- Generating property ---------")
        property_by_station_day = generate_train_property(dataset_array["train_x"])
        print("--------- Finished: generating property ---------")

        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)   

        train_loss = []
        eval_loss = []
        for ix_epoch in range(1, args.epoch+1):
            try:
                print(f"Epoch {ix_epoch}\n---------")
                train_loss.append(train_logic(model, train_loader, loss_function, optimizer, property_by_station_day))
                loss_, gt, y_pred = test(model, val_loader, loss_function)
                eval_loss.append(loss_)
                print()
            except KeyboardInterrupt:
                break

        plt.plot(train_loss)
        plt.plot(eval_loss)
        plt.show()

        model_path = "/hdd/traffic_data_2019/run/"
        save_model(model_path, model, "lstm_logic", ix_epoch)



    elif args.mode == "eval":
        print("Testing lstm")
        model_path = "/hdd/traffic_data_2019/run/lstm_epoch_15.pt"
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        loss_function = nn.MSELoss()
        eval_loss, gt, y_pred = test(model, test_loader, loss_function)

        plt.plot(gt)
        plt.plot(y_pred)
        plt.show()

        
        print("Testing logic-lstm")
        model_path = "/hdd/traffic_data_2019/run/lstm_logic_epoch_15.pt"
        model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
        model = to_device(model, args.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        loss_function = nn.MSELoss()
        eval_loss, gt, y_pred = test(model, test_loader, loss_function)
        plt.plot(gt)
        plt.plot(y_pred)
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone.')
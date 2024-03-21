"""
Utils and settings for training. 
Implementations for getting device, 
loading data to device, saving models, 
and loading dataset.
"""
import torch
import numpy as np
from IoTData import SequenceDataset
from torch.utils.data import DataLoader
from network import ShallowRegressionLSTM, ShallowRegressionGRU, ShallowRegressionRNN, MultiRegressionLSTM, MultiRegressionGRU, MultiRegressionRNN
from transformer import TimeSeriesTransformer
import os



weight_keys_mapping = {
    'lstm': ['lstm_1.weight_ih', 'lstm_1.weight_hh', 'lstm_1.bias_ih', 'lstm_1.bias_hh', 
             'lstm_2.weight_ih', 'lstm_2.weight_hh', 'lstm_2.bias_ih', 'lstm_2.bias_hh'],
    'gru': ['gru_1.weight_ih', 'gru_1.weight_hh', 'gru_1.bias_ih', 'gru_1.bias_hh', 
            'gru_2.weight_ih', 'gru_2.weight_hh', 'gru_2.bias_ih', 'gru_2.bias_hh'],
    'rnn': ['rnn_1.weight_ih', 'rnn_1.weight_hh', 'rnn_1.bias_ih', 'rnn_1.bias_hh', 
            'rnn_2.weight_ih', 'rnn_2.weight_hh', 'rnn_2.bias_ih', 'rnn_2.bias_hh'],
    'transformer': ['encoder_input_layer.weight', 'encoder_input_layer.bias',
                    'decoder_input_layer.weight', 'decoder_input_layer.bias',
                    'linear_mapping.weight', 'linear_mapping.bias']
}


def model_init(args):
    if args.dataset == 'fhwa':
        if args.model == 'LSTM':
            glob_model = ShallowRegressionLSTM(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
        elif args.model == 'GRU':
            glob_model = ShallowRegressionGRU(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
        elif args.model == 'RNN':
            glob_model = ShallowRegressionRNN(input_dim=2, batch_size=args.batch_size, time_steps=96, sequence_len=24, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
        elif args.model == 'transformer':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys = weight_keys_mapping['transformer']
        else:
            print("Model type:", args.model, "not implemented")

    elif args.dataset == 'sumo':
        if args.model == 'GRU':
            glob_model = MultiRegressionGRU(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['gru']
        elif args.model == 'LSTM':
            glob_model = MultiRegressionLSTM(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['lstm']
        elif args.model == 'RNN':
            glob_model = MultiRegressionRNN(input_dim=6, batch_size=args.batch_size, time_steps=40, sequence_len=10, hidden_dim=16)
            clust_weight_keys = weight_keys_mapping['rnn']
        elif args.model == 'transformer':
            glob_model = TimeSeriesTransformer()
            clust_weight_keys = weight_keys_mapping['transformer']
        else:
            print("Model type:", args.model, "not implemented")

    return glob_model, clust_weight_keys



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)



def save_model(path, model, model_name, epoch):
    if not os.path.isdir(path):
        os.makedirs(path)
    save_prefix = os.path.join(path, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close() 



def get_client_dataset(client_id, dataset_name):
    """
    Getting client dataset files by dataset name. 
    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        dataset_path = "/hdd/FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "/hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    train_dataset = SequenceDataset(dataset_array["train_x"], dataset_array["train_y"])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])
    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, drop_last=True)

    return train_loader, val_loader, test_loader, dataset_len



def get_shared_dataset(client_id, dataset_name):
    """
    Getting client and shared dataset files by dataset name. 
    Training data is seperated to a small shared group and a large private group.
    """
    dataset_array = {}
    if dataset_name == 'fhwa':
        dataset_path = "/hdd/FHWA_dataset/torch_dataset/"
    elif dataset_name == 'sumo':
        dataset_path = "/hdd/SUMO_dataset/learn_dataset/"
    for fold in ["train", "test", "val"]:
        x_file = dataset_path+fold+"_"+str(client_id)+"_x.npy"
        y_file = dataset_path+fold+"_"+str(client_id)+"_y.npy"
        dataset_array[fold+"_x"] = np.load(x_file, allow_pickle=True)
        dataset_array[fold+"_y"] = np.load(y_file, allow_pickle=True)
    
    public_len = int(0.5*len(dataset_array["train_x"]))
    train_dataset = SequenceDataset(dataset_array["train_x"][public_len:], dataset_array["train_y"][public_len:])
    val_dataset = SequenceDataset(dataset_array["val_x"], dataset_array["val_y"])
    test_dataset = SequenceDataset(dataset_array["test_x"], dataset_array["test_y"])

    dataset_len = [len(train_dataset), len(val_dataset), len(test_dataset)]
    train_loader_private = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=5, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=5, drop_last=True)

    return train_loader_private, [dataset_array["train_x"][:public_len], dataset_array["train_y"][:public_len]], val_loader, test_loader, dataset_len
"""
Prediction model implementations in PyTorch.
Models include vanilla RNN, GRU, LSTM, and time series Transformers.

Reference: Transformer implementation: https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MultiRegressionRNN(nn.Module):
    def __init__(self, input_dim=6, batch_size=64, time_steps=40, sequence_len=10, hidden_dim=16):
        super().__init__()
        self.model_type = 'rnn'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.rnn_1 = nn.RNNCell(input_size=self.input_dim, hidden_size=self.hidden_dim)
        self.rnn_2 = nn.RNNCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim)

    def forward(self, x, h_1, h_2):
        output_seq = torch.empty((self.batch_size, self.time_steps, self.input_dim)).to("cuda")

        for t in range(self.time_steps):
            h_1 = self.rnn_1(x[:,t,:], h_1)
            h_2 = self.rnn_2(h_1, h_2)
            output_seq[:,t,:] = self.fc(h_2).view(self.batch_size, self.input_dim)
        return output_seq, h_1, h_2

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_dim, device="cuda")



class MultiRegressionGRU(nn.Module):
    def __init__(self, input_dim=6, batch_size=64, time_steps=40, sequence_len=10, hidden_dim=16):
        super().__init__()
        self.model_type = 'gru'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.gru_1 = nn.GRUCell(input_size=self.input_dim, hidden_size=self.hidden_dim)
        self.gru_2 = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim)

    def forward(self, x, h_1, h_2):
        output_seq = torch.empty((self.batch_size, self.time_steps, self.input_dim)).to("cuda")

        for t in range(self.time_steps):
            h_1 = self.gru_1(x[:,t,:], h_1)
            h_2 = self.gru_2(h_1, h_2)
            output_seq[:,t,:] = self.fc(h_2).view(self.batch_size, self.input_dim)
        return output_seq, h_1, h_2

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_dim, device="cuda")



class MultiRegressionLSTM(nn.Module):
    def __init__(self, input_dim=6, batch_size=64, time_steps=40, sequence_len=10, hidden_dim=16):
        super().__init__()
        self.model_type = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.lstm_1 = nn.LSTMCell(input_size=self.input_dim, hidden_size=self.hidden_dim)
        self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=self.input_dim)

    def forward(self, x, hc_1, hc_2):
        output_seq = torch.empty((self.batch_size, self.time_steps, self.input_dim)).to("cuda")

        for t in range(self.time_steps):
            hc_1 = self.lstm_1(x[:,t,:], hc_1)
            h_1, _ = hc_1
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, _ = hc_2
            output_seq[:,t,:] = self.fc(h_2).view(self.batch_size, self.input_dim)
        return output_seq, hc_1, hc_2

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim, device="cuda"), 
                torch.zeros(self.batch_size, self.hidden_dim, device="cuda"))



class ShallowRegressionLSTM(nn.Module):
    def __init__(self, input_dim=2, batch_size=64, time_steps=96, sequence_len=24, hidden_dim=16):
        super().__init__()
        self.model_type = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.lstm_1 = nn.LSTMCell(input_size=self.time_steps, hidden_size=self.hidden_dim)  # TODO: fix input size
        self.lstm_2 = nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x, hc_1, hc_2):
        output_seq = torch.empty((self.batch_size, self.sequence_len)).to("cuda")
        for t in range(self.sequence_len):
            hc_1 = self.lstm_1(x[:,t:t+self.time_steps], hc_1)    # takes in traffic volume data only
            h_1, _ = hc_1
            hc_2 = self.lstm_2(h_1, hc_2)
            h_2, _ = hc_2
            output_seq[:,t] = self.fc(h_2).view(self.batch_size)
        return output_seq, hc_1, hc_2

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim, device="cuda"), 
                torch.zeros(self.batch_size, self.hidden_dim, device="cuda"))



class ShallowRegressionGRU(nn.Module):
    def __init__(self, input_dim=2, batch_size=64, time_steps=96, sequence_len=24, hidden_dim=16):
        super().__init__()
        self.model_type = 'gru'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.gru_1 = nn.GRUCell(input_size=self.time_steps, hidden_size=self.hidden_dim)  # TODO: fix input size
        self.gru_2 = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x, h_1, h_2):
        output_seq = torch.empty((self.batch_size, self.sequence_len)).to("cuda")
        for t in range(self.sequence_len):
            h_1 = self.gru_1(x[:,t:t+self.time_steps], h_1)    # takes in traffic volume data only
            h_2 = self.gru_2(h_1, h_2)
            output_seq[:,t] = self.fc(h_2).view(self.batch_size)
        return output_seq, h_1, h_2

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_dim, device="cuda")



class ShallowRegressionRNN(nn.Module):
    def __init__(self, input_dim=2, batch_size=64, time_steps=96, sequence_len=24, hidden_dim=16):
        super().__init__()
        self.model_type = 'rnn'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.time_steps = time_steps

        self.rnn_1 = nn.RNNCell(input_size=self.time_steps, hidden_size=self.hidden_dim)
        self.rnn_2 = nn.RNNCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)

    def forward(self, x, h_1, h_2):
        output_seq = torch.empty((self.batch_size, self.sequence_len)).to("cuda")
        for t in range(self.sequence_len):
            h_1 = self.rnn_1(x[:,t:t+self.time_steps], h_1)    # takes in traffic volume data only
            h_2 = self.rnn_2(h_1, h_2)
            output_seq[:,t] = self.fc(h_2).view(self.batch_size)
        return output_seq, h_1, h_2

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_dim, device="cuda")
    





## transformer for sequential prediction
class Transformer(nn.Module):
    def __init__(self, ntoken: int=96, d_model: int=200, nhead: int=2, d_hid: int=200,
                 nlayers: int=1, dropout: float=0.2):
        """
        Arguments:
            ntoken : input dimension of the model
            d_model : dimensionality of the embedding layer 
            nhead : number of attention heads in the multi-head attention layer 
            d_hid : dimensionality of feedforward layer in encoder and decoder blocks
            nlayers : number of encoder and decoder blocks 
        """
        super().__init__()
        self.d_model = d_model
        self.model_type = 'transformer'
        self.embed = nn.Embedding(ntoken, d_model)      
        self.pos_encoder = PositionalEncoding(d_model, dropout)     
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)    ## encoding layer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)      
        self.decoder = nn.Linear(d_model, ntoken)       
        self.init_weights()

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embed(src) * math.sqrt(self.d_model)     ## embedding
        src = self.pos_encoder(src)                         ## positional encoding
        output = self.transformer_encoder(src, src_mask)    ## transformer encoder
        output = self.decoder(output)                       ## decoder
        return output
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)



class PositionalEncoding(nn.Module):
    """
    injects information about relative or absolute position of tokens
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        Add positional encoding to the embedding vector
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    



def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
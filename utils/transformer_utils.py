import os
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path



def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, for both src and tgt masking, this must be target sequence
              length
        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 
    Return:
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
        """
        Produce all the start and end index positions of all sub-sequences.
        The indices will be used to split the data into sub-sequences on which 
        the models will be trained. 
        Returns a tuple with four elements:
        1) The index position of the first element to be included in the input sequence
        2) The index position of the last element to be included in the input sequence
        3) The index position of the first element to be included in the target sequence
        4) The index position of the last element to be included in the target sequence
        
        Args:
            num_obs (int): Number of observations in the entire dataset for which
                            indices must be generated.
            input_len (int): Length of the input sequence (a sub-sequence of 
                             of the entire data sequence)
            step_size (int): Size of each step as the data sequence is traversed.
                             If 1, the first sub-sequence will be indices 0-input_len, 
                             and the next will be 1-input_len.
            forecast_horizon (int): How many index positions is the target away from
                                    the last index position of the input sequence?
                                    If forecast_horizon=1, and the input sequence
                                    is data[0:10], the target will be data[11:taget_len].
            target_len (int): Length of the target / output sequence.
        """

        input_len = round(input_len) # just a precaution
        start_position = 0
        stop_position = num_obs-1 # because of 0 indexing
        
        subseq_first_idx = start_position
        subseq_last_idx = start_position + input_len
        target_first_idx = subseq_last_idx + forecast_horizon
        target_last_idx = target_first_idx + target_len 
        print("target_last_idx is {}".format(target_last_idx))
        print("stop_position is {}".format(stop_position))
        indices = []
        while target_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx, target_first_idx, target_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size
            target_first_idx = subseq_last_idx + forecast_horizon
            target_last_idx = target_first_idx + target_len

        return indices

def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        """
        Produce all the start and end index positions that is needed to produce
        the sub-sequences. 
        Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
        sequence. These tuples should be used to slice the dataset into sub-
        sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences. 
        
        Args:
            num_obs (int): Number of observations (time steps) in the entire 
                           dataset for which indices must be generated, e.g. 
                           len(data)
            window_size (int): The desired length of each sub-sequence. Should be
                               (input_sequence_length + target_sequence_length)
                               E.g. if you want the model to consider the past 100
                               time steps in order to predict the future 50 
                               time steps, window_size = 100+50 = 150
            step_size (int): Size of each step as the data sequence is traversed 
                             by the moving window.
                             If 1, the first sub-sequence will be [0:window_size], 
                             and the next will be [1:window_size].
        Return:
            indices: a list of tuples
        """

        stop_position = len(data)-1 # 1- because of 0 indexing
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        subseq_last_idx = window_size
        indices = []
        
        while subseq_last_idx <= stop_position:
            indices.append((subseq_first_idx, subseq_last_idx))
            subseq_first_idx += step_size
            subseq_last_idx += step_size

        return indices
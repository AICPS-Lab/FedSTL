from telex import synth
from telex import inputreader
import telex.scorer as scorer
import telex.tensor_scorer as tensor_scorer
import telex.parametrizer as parametrizer
import telex.stl as stl
import numpy as np
import scipy.stats
import torch
from pprint import pprint
import matplotlib.pyplot as plt


trace_dir = "/hdd/traffic_data_2019/stl_trace"



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
device = get_device()



STL_templates = {
    'lower': [
        "G[0,1](x >= a? -500;500 )",
        "G[2,3](x >= a? -500;500 )",
        "G[4,5](x >= a? -500;500 )",
        "G[6,7](x >= a? -500;500 )",
        "G[8,9](x >= a? -500;500 )",
        "G[10,11](x >= a? -500;500 )",
        "G[12,13](x >= a? -500;500 )",
        "G[14,15](x >= a? -500;500 )",
        "G[16,17](x >= a? -500;500 )",
        "G[18,19](x >= a? -500;500 )",
        "G[20,21](x >= a? -500;500 )",
        "G[22,23](x >= a? -500;500 )",
    ],
    'upper': [
        "G[0,1](x < a? 0;500 )",
        "G[2,3](x < a? 0;500 )",
        "G[4,5](x < a? 0;500 )",
        "G[6,7](x < a? 0;500 )",
        "G[8,9](x < a? 0;500 )",
        "G[10,11](x < a? 0;500 )",
        "G[12,13](x < a? 0;500 )",
        "G[14,15](x < a? 0;500 )",
        "G[16,17](x < a? 0;500 )",
        "G[18,19](x < a? 0;500 )",
        "G[20,21](x < a? 0;500 )",
        "G[22,23](x < a? 0;500 )",
    ],
    'eventually-lower': [
        "F[0,1](x >= a? -5;5 )",
        "F[2,3](x >= a? -5;5 )",
        "F[4,5](x >= a? -5;5 )",
        "F[6,7](x >= a? -5;5 )",
        "F[8,9](x >= a? -5;5 )",
        "F[10,11](x >= a? -5;5 )",
        "F[12,13](x >= a? -5;5 )",
        "F[14,15](x >= a? -5;5 )",
        "F[16,17](x >= a? -5;5 )",
        "F[18,19](x >= a? -5;5 )",
        "F[20,21](x >= a? -5;5 )",
        "F[22,23](x >= a? -5;5 )"
    ],
    'eventually-upper': [
        "F[0,1](x <= a? -5;5 )",
        "F[2,3](x <= a? -5;5 )",
        "F[4,5](x <= a? -5;5 )",
        "F[6,7](x <= a? -5;5 )",
        "F[8,9](x <= a? -5;5 )",
        "F[10,11](x <= a? -5;5 )",
        "F[12,13](x <= a? -5;5 )",
        "F[14,15](x <= a? -5;5 )",
        "F[16,17](x <= a? -5;5 )",
        "F[18,19](x <= a? -5;5 )",
        "F[20,21](x <= a? -5;5 )",
        "F[22,23](x <= a? -5;5 )"
    ],
    'until': [
        'U[0,23]( x1 >= a? -5;5, x2 <= b? -5;5)',
        'U[0,23]( x1 >= a? -5;5, x2 <= b? -5;5)',
        'U[0,23]( x1 <= a? -5;5, x2 >= b? -5;5)',
        'U[0,23]( x1 <= a? -5;5, x2 <= b? -5;5)',
    ],
    'corr-lower': [
        "G[0,2]({ x1 - x2 } >= a? -5;5 )",
        "G[3,5]({ x1 - x2 } >= a? -5;5 )",
        "G[6,8]({ x1 - x2 } >= a? -5;5 )",
        "G[9,11]({ x1 - x2 } >= a? -5;5 )",
        "G[12,14]({ x1 - x2 } >= a? -5;5 )",
        "G[15,17]({ x1 - x2 } >= a? -5;5 )",
        "G[18,20]({ x1 - x2 } >= a? -5;5 )",
        "G[21,23]({ x1 - x2 } >= a? -5;5 )",
        "G[24,26]({ x1 - x2 } >= a? -5;5 )",
        "G[27,29]({ x1 - x2 } >= a? -5;5 )",
        "G[30,32]({ x1 - x2 } >= a? -5;5 )",
        "G[33,34]({ x1 - x2 } >= a? -5;5 )",
        "G[35,37]({ x1 - x2 } >= a? -5;5 )",
        "G[38,39]({ x1 - x2 } >= a? -5;5 )",
    ], 
    'corr': [
        "G[0,1]({ x1 - x2 } >= a? -5;5 )",
        "G[2,3]({ x1 - x2 } >= a? -5;5 )",
        "G[4,5]({ x1 - x2 } >= a? -5;5 )",
        "G[6,7]({ x1 - x2 } >= a? -5;5 )",
        "G[8,9]({ x1 - x2 } >= a? -5;5 )",
        "G[10,11]({ x1 - x2 } >= a? -5;5 )",
        "G[12,13]({ x1 - x2 } >= a? -5;5 )",
        "G[14,15]({ x1 - x2 } >= a? -5;5 )",
        "G[16,17]({ x1 - x2 } >= a? -5;5 )",
        "G[18,19]({ x1 - x2 } >= a? -5;5 )",
        "G[20,21]({ x1 - x2 } >= a? -5;5 )",
        "G[22,23]({ x1 - x2 } >= a? -5;5 )",
        "G[24,25]({ x1 - x2 } >= a? -5;5 )",
        "G[26,27]({ x1 - x2 } >= a? -5;5 )",
        "G[28,29]({ x1 - x2 } >= a? -5;5 )",
        "G[30,31]({ x1 - x2 } >= a? -5;5 )",
        "G[32,33]({ x1 - x2 } >= a? -5;5 )",
        "G[34,35]({ x1 - x2 } >= a? -5;5 )",
        "G[36,37]({ x1 - x2 } >= a? -5;5 )",
        "G[38,39]({ x1 - x2 } >= a? -5;5 )",
    ], 
    'corr-1': [
        "G[0,1](x >= a? -5;5 )",
        "G[2,3](x >= a? -5;5 )",
        "G[4,5](x >= a? -5;5 )",
        "G[6,7](x >= a? -5;5 )",
        "G[8,9](x >= a? -5;5 )",
        "G[10,11](x >= a? -5;5 )",
        "G[12,13](x >= a? -5;5 )",
        "G[14,15](x >= a? -5;5 )",
        "G[16,17](x >= a? -5;5 )",
        "G[18,19](x >= a? -5;5 )",
        "G[20,21](x >= a? -5;5 )",
        "G[22,23](x >= a? -5;5 )",
        "G[24,25](x >= a? -5;5 )",
        "G[26,27](x >= a? -5;5 )",
        "G[28,29](x >= a? -5;5 )",
        "G[30,31](x >= a? -5;5 )",
        "G[32,33](x >= a? -5;5 )",
        "G[34,35](x >= a? -5;5 )",
        "G[36,37](x >= a? -5;5 )",
        "G[38,39](x >= a? -5;5 )",
    ], 
    'corr-test': [
        "G[0,1](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[2,3](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[4,5](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[6,7](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[8,9](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[10,11](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[12,13](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[14,15](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[16,17](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[18,19](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[20,21](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[22,23](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[24,25](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[26,27](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[28,29](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[30,31](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
        "G[32,33](x1 <= 100 -> G[0,5](x2 >= a? -50;50))",
    ], 
    'corr-test1': [
        "G[0,1](x1 <= 100 -> x2 >= a? -50;50)",
        "G[2,3](x1 <= 100 -> x2 >= a? -50;50)",
        "G[4,5](x1 <= 100 -> x2 >= a? -50;50)",
        "G[6,7](x1 <= 100 -> x2 >= a? -50;50)",
        "G[8,9](x1 <= 100 -> x2 >= a? -50;50)",
        "G[10,11](x1 <= 100 -> x2 >= a? -50;50)",
        "G[12,13](x1 <= 100 -> x2 >= a? -50;50)",
        "G[14,15](x1 <= 100 -> x2 >= a? -50;50)",
        "G[16,17](x1 <= 100 -> x2 >= a? -50;50)",
        "G[18,19](x1 <= 100 -> x2 >= a? -50;50)",
        "G[20,21](x1 <= 100 -> x2 >= a? -50;50)",
        "G[22,23](x1 <= 100 -> x2 >= a? -50;50)",
        "G[24,25](x1 <= 100 -> x2 >= a? -50;50)",
        "G[26,27](x1 <= 100 -> x2 >= a? -50;50)",
        "G[28,29](x1 <= 100 -> x2 >= a? -50;50)",
        "G[30,31](x1 <= 100 -> x2 >= a? -50;50)",
        "G[32,33](x1 <= 100 -> x2 >= a? -50;50)",
        "G[34,35](x1 <= 100 -> x2 >= a? -50;50)",
        "G[36,37](x1 <= 100 -> x2 >= a? -50;50)",
        "G[38,39](x1 <= 100 -> x2 >= a? -50;50)",
    ], 
    'corr-upper': [
        "G[0,2]({ x1 - x2 } <= a? -5;5 )",
        "G[3,5]({ x1 - x2 } <= a? -5;5 )",
        "G[6,8]({ x1 - x2 } <= a? -5;5 )",
        "G[9,11]({ x1 - x2 } <= a? -5;5 )",
        "G[12,14]({ x1 - x2 } <= a? -5;5 )",
        "G[15,17]({ x1 - x2 } <= a? -5;5 )",
        "G[18,20]({ x1 - x2 } <= a? -5;5 )",
        "G[21,23]({ x1 - x2 } <= a? -5;5 )",
        "G[24,26]({ x1 - x2 } <= a? -5;5 )",
        "G[27,29]({ x1 - x2 } <= a? -5;5 )",
        "G[30,32]({ x1 - x2 } <= a? -5;5 )",
        "G[33,34]({ x1 - x2 } <= a? -5;5 )",
        "G[35,37]({ x1 - x2 } <= a? -5;5 )",
        "G[38,39]({ x1 - x2 } <= a? -5;5 )",
    ], 
}


def get_max_window(arr, window_length=2):
    result = []
    for i in range(len(arr)):
        begin = i-window_length+1
        end = i+window_length
        if i-window_length+1 < 0:
            begin = 0
        elif i+window_length >= len(arr):
            end = len(arr)
        result.append(max(arr[begin:end]))   
    return result



def get_min_window(arr, window_length=2):
    result = []
    for i in range(len(arr)):
        begin = i-window_length+1
        end = i+window_length
        if i-window_length+1 < 0:
            begin = 0
        elif i+window_length >= len(arr):
            end = len(arr)
        result.append(min(arr[begin:end]))   
    return result



def test_stl(tlStr, optmethod = "gradient"):
    print("Got template:",tlStr) # STL template 
    (stlsyn, value, dur) = synth.synthSTLParam(tlStr, trace_dir, optmethod)
    print("Synthesized STL formula: {}\n Theta Optimal Value: {}\n Optimization time: {}\n".format(stlsyn, value, dur))
    (bres, qres) = synth.verifySTL(stlsyn, trace_dir)
    print("Test result of synthesized STL on each trace: {}\n Robustness Metric Value: {}\n".format(bres, qres))
    return stlsyn, value, dur



def delta(data):
    return np.diff(data) / data[:,1:] * 100



def cut_trace_list(trace_dir):
    tracenamelist = synth.find_filenames(trace_dir, suffix=".csv")
    tracelist = []
    for tracename in tracenamelist:
        tracelist.append(inputreader.readtracefile(tracename))
        
    for trace in tracelist:
        trace_len = 24
        list_chunked = []
        for i in range(len(trace['x'])//trace_len):
            curr_list = []
            for j in range(trace_len):
                curr_list.append(trace['x'][24*i+j])
            list_chunked.append(curr_list)
    return list_chunked




def cut_trace(tensor_arr):
    # cut tensor arrays into size of batch*day*24 hrs
    return tensor_arr.view(-1, 3, int(tensor_arr.shape[2]/24), 24)



def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



def cut_trace_by_day(tensor_arr):
    # cut tensor arrays into size of batch*day*24 hrs
    tensor_sep = tensor_arr.view(-1, int(tensor_arr.shape[1]/24), 24, 2)
    days = tensor_sep[:, -1, -1, 1] # the day of the predictions 
    raise NotImplementedError



def torch_confidence_interval(data: torch.Tensor, confidence: float = 0.90) -> torch.Tensor:
    """
    Computes the confidence interval for a given survey of a data set.
    """
    n = len(data)
    mean: torch.Tensor = data.mean()
    se: torch.Tensor = data.std(unbiased=True) / (n**0.5)
    t_p: float = float(scipy.stats.t.ppf((1 + confidence) / 2., n-1))
    ci = t_p * se
    return mean, mean-ci, mean+ci





def generate_property_test(tensor_arr, property_type = "upper", mining_range = 2):
    optmethod = "gradient"
    up_bound    = torch.zeros(tensor_arr.shape[1]).to(device)        # shape: batch*hrs
    low_bound   = torch.zeros(tensor_arr.shape[1]).to(device)        # shape: batch*hrs
    property    = torch.zeros([tensor_arr.shape[0], tensor_arr.shape[1]]).to(device)
    
    stlsyn_lib = []
    for j in range(tensor_arr.shape[1]):
        up_bound = torch.max(tensor_arr, 0).values.cpu().detach().numpy()
        low_bound = torch.min(tensor_arr, 0).values.cpu().detach().numpy()

    if property_type == "upper":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            try:
                (stlsyn, value, dur) = synth.synthSTLParam(templ, up_bound, optmethod)
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound
                stlsyn_lib.append(stlsyn)
            except:
                stlsyn_lib.append('')

    elif property_type == "lower":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            try:
                (stlsyn, value, dur) = synth.synthSTLParam(templ, low_bound, optmethod)
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound
                stlsyn_lib.append(stlsyn)
            except:
                stlsyn_lib.append('')

    elif property_type == "corr":
        x_test_0 = torch.max(tensor_arr[:,:,0], 0).values.cpu().detach().numpy()
        x_test_1 = torch.max(tensor_arr[:,:,1], 0).values.cpu().detach().numpy()
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            # val_dict = {'x1': x1_bound, 'x2': x2_bound}
            (stlsyn, value, dur) = synth.synthSTLParam(templ, x_test_0-x_test_1, optmethod)
            stlsyn_lib.append(stlsyn)

    elif property_type == "until":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            conf_interval_dict = {
                'x1': conf_interval[:,1].cpu().numpy(), # lower
                'x2': conf_interval[:,2].cpu().numpy(), # upper
            }
            (stlsyn, value, dur) = synth.synthSTLParam(templ, [conf_interval_dict], optmethod)

    elif property_type == "eventually-upper":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            try:
                (stlsyn, value, dur) = synth.synthSTLParam(templ, conf_interval[:,2].cpu().numpy(), optmethod)
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound
            except:
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = sliding_window[mining_range*temp_idx:mining_range*temp_idx+mining_range, 2]

    elif property_type == "eventually-lower":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            try:
                (stlsyn, value, dur) = synth.synthSTLParam(templ, conf_interval[:,1].cpu().numpy(), optmethod)
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound
            except:
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = sliding_window[mining_range*temp_idx:mining_range*temp_idx+mining_range, 1]

    return property, stlsyn_lib










def cumscoretracelist(tlstl, paramvalue, tracelist, scorerfun):
    score = 0
    pstl = stl.parse(tlstl)
    paramlist = parametrizer.getParams(pstl) # eg. [b? -3.0;0.0 , a? 0.0;3.0 ]
    valmap = {}
    i = 0   # number of params
    for param in paramlist:
        valmap[param.name] = paramvalue[i]
        i = i + 1
    stlcand = parametrizer.setParams(pstl, valmap)
    if isinstance(tracelist, list):
        for trace in tracelist:
            try:
                quantscore = scorerfun(stlcand, trace, 0)
            except ValueError:
                quantscore = -10000         # proxy for -inf, putting -inf makes optimizers angry
            score = (score + quantscore) 
    else:
        try:
            quantscore = scorerfun(stlcand, tracelist, 0)
        except ValueError:
            quantscore = -10000             # proxy for -inf, putting -inf makes optimizers angry
        score = (score + quantscore) 
    return score




def get_robustness_score(tensor_arr, pred, property_type = "upper", mining_range = 2):
    # Usage: 
    # cons_loss = get_robustness_score(X, output, property_type = "lower") + get_robustness_score(X, output, property_type = "upper")
    # cons_loss = get_robustness_score(X, output, property_type = "eventually")

    scorefun_1  = scorer.smartscore
    scorefun_2  = tensor_scorer.smartscore
    optmethod = "gradient"
    
    tensor_chunked = cut_trace(tensor_arr)                               # (batch_size, 3, days, hours) = (64, 3, 5, 24)
    conf_interval  = torch.zeros(tensor_chunked.shape[3], 3).to(device)  # batch * 1 * hrs * (mean, lowerbound, upperbound)
    property_list  = []
    
    for j in range(tensor_chunked.shape[3]):
        mean, conf_low, conf_up = torch_confidence_interval(tensor_chunked[:,0,:,j])
        conf_interval[j,0] = mean
        conf_interval[j,1] = conf_low
        conf_interval[j,2] = conf_up

    if property_type == "upper":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            (stlsyn, value, dur) = synth.synthSTLParam(templ, conf_interval[:,2].cpu().numpy(), optmethod)
            property_list.append(stlsyn.subformula.bound)

    elif property_type == "lower":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            (stlsyn, value, dur) = synth.synthSTLParam(templ, conf_interval[:,1].cpu().numpy(), optmethod)
            property_list.append(stlsyn.subformula.bound)     
    
    elif property_type == "until":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            conf_interval_dict = {
                'x1': conf_interval[:,1].cpu().numpy(), # lower
                'x2': conf_interval[:,2].cpu().numpy(), # upper
            }
            (stlsyn, value, dur) = synth.synthSTLParam(templ, [conf_interval_dict], optmethod)
            property_list.append(stlsyn.left.bound)
            property_list.append(stlsyn.right.bound)

    elif property_type == "eventually":
        (stlsyn, value, dur) = synth.synthSTLParam(STL_templates[property_type][0], conf_interval[:,1].cpu().numpy(), optmethod)
        property_list.append(stlsyn.subformula.bound)
        (stlsyn, value, dur) = synth.synthSTLParam(STL_templates[property_type][1], conf_interval[:,2].cpu().numpy(), optmethod)
        property_list.append(stlsyn.subformula.bound)  
            
    
    # get robustness score
    prop_score = torch.zeros(1).cuda()
    if property_type == "until":
        for i in range(0, len(property_list), 2):
            trace_score = 0
            for trace in pred.cpu().detach().numpy():
                score = cumscoretracelist(STL_templates[property_type][i//2], [property_list[i], property_list[i+1]], trace, scorefun)
                trace_score += score
            prop_score += trace_score

    elif property_type == "eventually":
        trace_score = 0
        for trace in pred.cpu().detach().numpy():
            trace_score += cumscoretracelist(STL_templates[property_type][0], [property_list[0]], trace, scorefun)
            trace_score += cumscoretracelist(STL_templates[property_type][1], [property_list[1]], trace, scorefun)
        prop_score += trace_score

    else:       # upper or lower
        for ind, param in enumerate(property_list):
            trace_score = torch.zeros(1).cuda()
            for trace in pred:
                score = cumscoretracelist(STL_templates[property_type][ind], [param], trace, scorefun_2)
                trace_score += score.cuda()
            prop_score += trace_score

    return prop_score



def generate_property(tensor_arr, property_type = "corr", mining_range = 2):
    optmethod = "gradient"
    up_bound    = torch.zeros(tensor_arr.shape[1]).to(device)        # shape: batch*hrs
    low_bound   = torch.zeros(tensor_arr.shape[1]).to(device)        # shape: batch*hrs
    property    = torch.zeros([tensor_arr.shape[0], tensor_arr.shape[1]]).to(device)
    
    for j in range(tensor_arr.shape[1]):
        up_bound = torch.max(tensor_arr, 0)
        low_bound = torch.min(tensor_arr, 0)

    if property_type == "upper":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            try:
                (stlsyn, value, dur) = synth.synthSTLParam(templ, conf_interval[:,2], optmethod)
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound
            except:
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = sliding_window[mining_range*temp_idx:mining_range*temp_idx+mining_range, 2]

    elif property_type == "lower":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            try:
                (stlsyn, value, dur) = synth.synthSTLParam(templ, conf_interval[:,1], optmethod)
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound
            except:
                property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = sliding_window[mining_range*temp_idx:mining_range*temp_idx+mining_range, 1]
    
    elif property_type == "until":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            conf_interval_dict = {
                'x1': torch.max(tensor_arr[:,:,0], 0).values.cpu().detach().numpy(), # lower
                'x2': torch.max(tensor_arr[:,:,0], 0).values.cpu().detach().numpy(), # upper
            }
            (stlsyn, value, dur) = synth.synthSTLParam(templ, [conf_interval_dict], optmethod)

    elif property_type == "corr":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            val_dict = {
                'x1': torch.max(tensor_arr[:,:,0], 0).values.cpu().detach().numpy(), # lower
                'x2': torch.max(tensor_arr[:,:,0], 0).values.cpu().detach().numpy(), # upper
            }
            (stlsyn, value, dur) = synth.synthSTLParam(templ, [val_dict], optmethod)
            property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound

    elif property_type == "corr-upper":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            val_dict = {
                'x1': conf_interval[:,2].cpu().numpy(),       
                'x2': conf_interval_guide[:,2].cpu().numpy(),  
            }
            (stlsyn, value, dur) = synth.synthSTLParam(templ, [val_dict], optmethod)
            property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound

    elif property_type == "corr-lower":
        for temp_idx, templ in enumerate(STL_templates[property_type]):
            val_dict = {
                'x1': conf_interval[:,1].cpu().numpy(),       
                'x2': conf_interval_guide[:,1].cpu().numpy(),  
            }
            (stlsyn, value, dur) = synth.synthSTLParam(templ, [val_dict], optmethod)
            property[:, mining_range*temp_idx:mining_range*temp_idx+mining_range] = stlsyn.subformula.bound

    else:
        raise NotImplementedError
    
    return property
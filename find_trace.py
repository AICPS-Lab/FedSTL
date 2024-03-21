"""
Correcting a prediction by STL specifications.
Process different conditions by STL operators.
"""
import numpy as np
import torch
import telex.stl as stl
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
torch.set_printoptions(sci_mode=False, precision=3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def best_trace_helper():
    pass


def convert_best_trace(stl_lib, trace: torch.Tensor, multi: bool = False):
    target = torch.zeros_like(trace).to(device)
    corrected_trace = torch.zeros_like(trace).to(device)
    
    for stl_form in stl_lib:
        if isinstance(stl_form, stl.Globally):
            if multi:
                left_t = int(stl_form.interval.left)
                right_t = int(stl_form.interval.right)
                right_bound = stl_form.subformula.bound
                right_relop = stl_form.subformula.relop
                target[:,left_t:right_t+1,1] = right_bound+target[:,left_t:right_t+1,0]
                if '>' in right_relop:
                    corrected_trace = torch.where(target < trace, trace, target)
                elif '<' in right_relop:
                    corrected_trace = torch.where(target > trace, trace, target)
            else:
                left_t = int(stl_form.interval.left)
                right_t = int(stl_form.interval.right)
                target[:,left_t:right_t+1] = stl_form.subformula.bound
                if '>' in stl_form.subformula.relop:
                    corrected_trace = torch.where(target < trace, trace, target)
                elif '<' in stl_form.subformula.relop:
                    corrected_trace = torch.where(target > trace, trace, target)
                else:
                    raise NameError('In convert_best_trace: unknown atomic logic operator.')

        elif isinstance(stl_form, stl.Future):
            left_t = int(stl_form.interval.left)
            right_t = int(stl_form.interval.right)
            target[:,left_t:right_t+1] = stl_form.subformula.bound
            if '>' in stl_form.subformula.relop:
                mask = target > trace
                corrected_trace = torch.where(mask, target, trace)
            elif '<' in stl_form.subformula.relop:
                mask = target < trace
                corrected_trace = torch.where(mask, target, trace)
            else:
                raise NameError('In convert_best_trace: unknown atomic logic operator.')

        elif isinstance(stl_form, stl.Until):
            left_form = stl_form.left
            right_form = stl_form.right
            begin_t = int(stl_form.interval.left)
            end_t = int(stl_form.interval.right)
            if '>' in stl_form.subformula.relop:
                ...
            elif '<' in stl_form.subformula.relop:
                ...
            else:
                raise NameError('In convert_best_trace: unknown atomic logic operator.')

        elif isinstance(stl_form, stl.Or):
            left_corrected  = convert_best_trace(stl_form.left, trace)
            right_corrected = convert_best_trace(stl_form.right, left_corrected)
            return right_corrected

        elif isinstance(stl_form, stl.And):
            left_corrected  = convert_best_trace(stl_form.left, trace)
            right_corrected = convert_best_trace(stl_form.right, left_corrected)
            return right_corrected

        # else:
        #     raise RuntimeError('In convert_best_trace: unknown STL operator type.')
        
    return corrected_trace
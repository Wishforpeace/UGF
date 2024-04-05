import os
import random
import numpy as np
import torch

def check_dir(path, make_dir=True):
    if not os.path.exists(path):  #
        os.makedirs(path)


def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')





def check_and_save(model, result, check, save_model=True, parallel=False):

    for key in check.keys():
        
        if key == 'MAE' or key == 'Loss':
            if check[key] > result[key]:
                
                if save_model:
                    if parallel:
                        model.module.save_model()
                    else:
                        model.save_model()
                check[key] = result[key]
        else:
            if check[key] < result[key]:
                if save_model:
                    if parallel:
                        model.module.save_model()
                    else:
                        model.save_model()
                check[key] = result[key]
    return check
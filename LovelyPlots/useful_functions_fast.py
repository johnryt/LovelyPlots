import numpy as np
import pandas as pd

def write_to_log(string, log_file_path='outputs/log.txt', reinitialize=False):
    if reinitialize:
        file = open(log_file_path, 'w')
    else:
        file = open(log_file_path, 'a')
    file.write(string+'\n')
    file.close()
    
def logit(x):
    return 1/(1+np.exp(-x))

def pickel_object(variable, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def get_memory_usage(locs):
    """pass locals() or globals(), or other dict of values, to get the sizes of all its pandas dataframes/series in ~MB"""
    sizes = pd.Series()
    for i in locs:
        if hasattr(locs[i],'memory_usage') and i not in ['np']:
            ph = locs[i].memory_usage(deep=True)
            if hasattr(ph,'sum'):
                sizes[i] = ph.sum()
            else:
                sizes[i] = ph
    return sizes.sort_values()/1e6

def can_be_int(i):
    try:
        int(i)
        return True
    except:
        return False
    
def can_be_float(i):
    try:
        float(i)
        return True
    except:
        return False

import numpy as np
import pandas as pd
from os import listdir

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


def read_parquet_subset(file_path, filters=None, columns=None):
    """
    Read a parquet file with optional column-based filtering.
    
    Parameters
    ----------
    file_path : str
        Path to the parquet file.
    filters : dict or None
        Dictionary where keys are column names and values are the criteria to match.
        Values can be a single value or a list of values.
        E.g. {'Reporter ISO Alpha-3': 'USA', 'Period': [2019, 2020]}
    columns : list or None
        List of column names to include - see readme in Data - Trade Data Monitor folder for list of columns available.
    
    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd
    import pyarrow.parquet as pq
    
    if filters is None and columns is None:
        return pd.read_parquet(file_path)
    if type(columns)==str:
        columns = [columns]
    
    columns_available = get_parquet_columns(file_path)
    columns_want = (list(columns) if columns is not None else []) + (list(filters.keys()) if filters is not None else [])
    if columns_want:
        # Validate that all specified columns exist
        missing_cols = set(columns) - set(columns_available)
        if missing_cols:
            raise ValueError(f"The following columns do not exist in the parquet file: {missing_cols}. \nOptions include {columns_available}")

    # Build pyarrow filter expressions for row-group-level pushdown
    filter_conditions = []
    for col, values in filters.items():
        if not isinstance(values, (list, tuple, set, np.ndarray, pd.arrays.ArrowStringArray)):
            values = [values]
        filter_conditions.append((col, 'in', list(values)))
    
    df = pd.read_parquet(file_path, filters=filter_conditions, columns=columns)
    return df

def get_parquet_columns(file_path):
    """Get column names from a parquet file without loading the entire file."""
    import pyarrow.parquet as pq
    from os import listdir
    if not file_path.endswith('.parquet'):
        # get first parquet file from folder
        file_path = f"{file_path}/{[i for i in listdir(file_path) if i.endswith('.parquet')][0]}"
    schema = pq.read_schema(file_path)
    return schema.names

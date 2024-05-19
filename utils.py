import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client
from funcs import *


_parameter_constraints: dict = {
    "file_type": ["csv", "pqt", "h5"]
} 


def read_data(client: Client, data_loc: str, file_type: str, key: str = None, n_blocks: int = None) -> None:

    if not isinstance(file_type, str):
        raise TypeError(
            f"""File type must be a string. 
            Type given was {type(file_type)}"""
            )

    file_type_clean = ''.join(char for char in file_type if char.isalnum())

    if file_type_clean not in _parameter_constraints["file_type"]:
        raise ValueError(
            f"""Invalid file type. For instance, the data is of type "{file_type_clean}". 
            Input data must be of type {_parameter_constraints}"""
            )
    
    if file_type_clean.lower() == "csv":
        data = dd.read_csv(data_loc)

    elif file_type_clean.lower() == "pqt":
        data = dd.read_parquet(data_loc)

    elif file_type_clean.lower() == "h5":

        if key is None:
            raise ValueError(
                f"""File key required for reading HDFs."""
            )
        
        data = dd.read_hdf(data_loc, key = key, lock = True, sorted_index = True)

    if n_blocks is not None:
        data.repartition(npartitions = n_blocks)

    X = data.to_dask_array(lengths = True)

    if X.shape[0] % X.chunksize[0] != 0:
        #inappropriate error for this 
        raise ValueError(
            f"""Data partition shapes do not match. Pad data with zeroes or repartition. 
            For instance, the data is of shape {X.shape} and partitions of shape {X.chunksize}."""
            )
    
    X_persisted = client.persist(X)

    X = X_persisted.partitions.ravel()
    X = client.persist(X)

    return X

def preprocess_data(client: Client, data_partitions: list):

    X = client.map(get_autoscaled_data, data_partitions)
    X = client.gather(X)
    X_persisted = client.persist(X)
    
    return X_persisted

def progress_bar(curr_iter: int, max_iter: int, curr_component: int, n_components: int, max_eps: np.float64):

    length = 60

    complete_chars = "#"*(curr_iter+1)*(length//max_iter)
    incomplete_chars = "-"*(length - (curr_iter+1)*(length//max_iter))
    
    progress_str = f"Fitting component {curr_component + 1}/{n_components} [" + complete_chars + incomplete_chars + f" {curr_iter+1}/{max_iter}] Eps: " + f"{max_eps :.3f}."

    return progress_str
    
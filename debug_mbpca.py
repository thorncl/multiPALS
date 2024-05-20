from mbpca import MBPCA
import numpy as np
import dask.dataframe as dd
import dask.array as da
from utils import *
import config
from dask.distributed import Client, LocalCluster


def main():

    cluster = LocalCluster()
    client = Client(cluster)
    pca = MBPCA(client, 5, 0.001, 10)
    raw_data = read_data(client, config.BRIDGE_DATA_PATH, "h5", "data")
    X = preprocess_data(client, raw_data)
    pca.fit(X)


    print('end')

if __name__ == "__main__":

    main()






from mbpca import MBPCA
import numpy as np
import dask.dataframe as dd
import dask.array as da
from utils import *
import config
from dask.distributed import Client, LocalCluster
import random

def main():

    cluster = LocalCluster()
    client = Client(cluster)
    pca = MBPCA(client, 2, 1e-18, 100)
    X = read_data(client, config.RANDOM_DATA_PATH, "h5", key = "data")
    # X = da.array([[[random.random() for j in range(4)] for i in range(10)] for k in range(6)])
    # X = client.persist(X.rechunk((1, 10, 4)))
    model = pca.fit(X)
    p_b, t_b, t_T, X_hat, E = pca.predict(X)

    print('end')

if __name__ == "__main__":

    main()






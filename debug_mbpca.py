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
    pca = MBPCA(client, 2, 1e-18, 100)
    raw_data = read_data(client, config.RANDOM_DATA_PATH, "h5", key = "data")
    X = preprocess_data(client, raw_data)
    model = pca.fit(X)
    X_transformed = pca.transform(X)
    X_hat = pca.predict()

    print('end')

if __name__ == "__main__":

    main()






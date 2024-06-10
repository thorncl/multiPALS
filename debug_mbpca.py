from mbpca import MBPCA
from utils import *
import config
from dask.distributed import Client, LocalCluster

def main():

    cluster = LocalCluster()
    client = Client(cluster)
    pca = MBPCA(client, 2, 1e-18, 100)
    X = read_data(client, config.RANDOM_DATA_PATH, "h5", key = "data")
    model = pca.fit(X)
    p_b, t_b, t_T, X_hat, E = pca.predict(X)

if __name__ == "__main__":

    main()






import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import copy
from dask_ml.preprocessing import StandardScaler
from dask.distributed import Client, as_completed
from utils import *

#TODO:

# SPE
# PRESS
# Total variance???
# Hotelling's T2

def _infer_dimensions(X):

    n_samples = X[0].shape[0]
    n_features = X[0].shape[1]
    n_partitions = len(X)

    return n_samples, n_features, n_partitions
    
class MBPCA:

    def __init__(
                self,
                client: Client,
                n_components: int = None,
                tol: float = 0.0,
                iter: int = 0,
                solver: str = "nipals"
            ):
        
        self.client: Client = client
        self.n_components: int = n_components
        self.tol: float = tol
        self.iter: int = iter
        self.solver = solver
        self._n_samples: int
        self._n_features: int
        self._n_partitions: int
        self.X: list = []
        self.mean: da
        self.std: da
        self.residuals: da 
        self.super_scores: da 
        self.block_loadings: da
        self.block_scores: da
        self.super_weights: da
        self.block_var_exp: da
        self.curr_component: int = 0

    def preprocess_data(self, X: list, mean: da = None, std: da = None):

        X_masked = da.where(~da.isfinite(X), np.nan, X)

        if all(arg is None for arg in [mean, std]):
            mean = da.nanmean(X_masked, axis = 1)
            std = da.sqrt(da.nanvar(X, axis = 1, ddof = 1))
        
        for i in range(X_masked.shape[0]):
            X_masked[i, :, :] = (X_masked[i, :, :] - mean[i, :])/std[i, :]

        X = X_masked.reshape(X_masked.shape[0]*X_masked.shape[1], X_masked.shape[2])
        X = X.partitions.ravel()

        return self.client.persist(X), mean, std

    def _init_model(self):

        self.residuals = da.empty((self._n_partitions, self._n_samples, self._n_features, self.n_components))
        self.super_scores = da.empty((self._n_samples, self.n_components))
        self.block_loadings = da.empty((self._n_partitions, self._n_features, self.n_components))
        self.block_scores = da.empty((self._n_samples, self._n_partitions, self.n_components))
        self.super_weights = da.empty((self._n_partitions, self.n_components))
        self.block_var_exp = da.empty((self._n_partitions, self.n_components))

    def _update_model(self, E: list, t_T: da, p_b: list, t_b: da, w_T: da, var_exp: list):

        self.residuals[:, :, :, self.curr_component] = da.asarray(E)
        self.super_scores[:, self.curr_component] = t_T.flatten()
        self.block_loadings[:, :, self.curr_component] = da.asarray(p_b).squeeze()
        self.block_scores[:, :, self.curr_component] = t_b
        self.super_weights[:, self.curr_component] = w_T.flatten()
        self.block_var_exp[:, self.curr_component] = da.asarray(var_exp)

    def set_initial_super_score(self):

        X_array = da.asarray(self.X)
        col_var = da.nanvar(X_array, axis = 1, ddof = 1)
        block_idx, col_idx = da.unravel_index(da.argmax(col_var, axis = None), col_var.shape)

        return self.X[block_idx][:, col_idx].reshape(-1,1)

    def set_super_score_block(self, t_T):

        return self.client.persist([t_T for i in range(len(self.X))])

    def get_block_loadings(self, t_T_block, norm: bool = True):
        
        p_b = self.client.map(get_p_b, self.X, t_T_block, [norm for i in range(len(self.X))])
        return self.client.gather(p_b)

    def get_block_scores(self, p_b):

        t_b = self.client.map(get_t_b, self.X, p_b)
        t_b = self.client.gather(t_b)
        t_b = da.hstack(t_b)
        return self.client.persist(t_b)

    def get_super_weights(self, t_b, t_T):

        w_T = self.client.submit(get_w_T, t_b, t_T)
        return w_T.result()

    def get_super_score(self, t_b, w_T):

        t_T_new = self.client.submit(get_t_T_new, t_b, w_T)
        return self.client.persist(t_T_new.result())

    def get_eps(self, t_T_new, t_T):

        eps = t_T_new - t_T
        
        return ((eps.T @ eps) / (t_T.shape[0] * (t_T_new.T @ t_T_new))).compute().item()

    def has_converged(self, eps):

        return eps < self.tol

    def deflate(self, t_T_block, p_b):

        E = self.client.map(get_residuals, self.X, t_T_block, p_b)
        E = self.client.gather(E)
        return self.client.persist(E)

    def get_variance_explained(self, E):

        tr_E = self.client.map(eig_sum, E)

        var_exp = self.client.map(get_var_exp, self.tr_X, tr_E)
        var_exp = self.client.gather(var_exp)
        return self.client.persist(var_exp)
    
    def fit(self, X: list):

        self.X = X
        self._n_samples, self._n_features, self._n_partitions = _infer_dimensions(self.X)
        self._init_model()

        self._fit(X)

        return self

    def _fit(self, X: list):

        self.X, self.mean, self.std = self.preprocess_data(X)

        if self.solver == "nipals":
            self.tr_X = self.client.map(eig_sum, self.X)
            self._fit_nipals()

    def _fit_nipals(self):

        for i in range(self.n_components):

            t_T = self.set_initial_super_score()
            t_T_block = self.set_super_score_block(t_T)

            for j in range(self.iter):

                p_b = self.get_block_loadings(t_T_block, norm = True)
                t_b = self.get_block_scores(p_b)
                w_T = self.get_super_weights(t_b, t_T)
                t_T_new = self.get_super_score(t_b, w_T)
                eps = self.get_eps(t_T_new, t_T)
                t_T = t_T_new
                t_T_block = self.set_super_score_block(t_T)

                print(progress_bar(j, self.iter, i, self.n_components, eps), end = '\r', flush = True)

                if self.has_converged(eps):
                    break 

            p_b = self.get_block_loadings(t_T_block, norm = False)
            E = self.deflate(t_T_block, p_b)
            var_exp = self.get_variance_explained(E)

            self._update_model(E, t_T, p_b, t_b, w_T, var_exp)
            
            self.X = E
            self.curr_component += 1 

    def predict(self, X_new: list):

        p_b = da.empty((self._n_partitions, self._n_features, self.n_components))
        t_T = da.empty((self._n_samples, self.n_components))
        X_hat = da.empty((self._n_partitions, self._n_samples, self._n_features))
        E = da.empty((self._n_partitions, self._n_samples, self._n_features))

        X_new_scaled, _, _ = self.preprocess_data(X_new, mean = self.mean, std = self.std)
        X_new_scaled = da.asarray(X_new_scaled)

        for i in range(self.n_components):

            p_b_i = da.asarray([self.block_loadings[j, :, i]/da.linalg.norm(self.block_loadings[j, :, i]) for j in range(self._n_partitions)])
            p_b[:, :, i] = p_b_i
     
        t_b = da.transpose(X_new_scaled @ p_b, axes = (1, 0, 2)) / da.sqrt(self._n_features)

        for i in range(self.n_components):

            t_T[:, i] = t_b[:, :, i] @ self.super_weights[:,i]

        for k in range(self._n_partitions):
            
            X_hat[k, :, :] = t_T @ p_b[k, :, :].T
            E[k, :, :] = X_new_scaled[k, :, :] - X_hat[k, :, :]

        return p_b, t_b, t_T, X_hat, E

    


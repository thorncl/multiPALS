import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import copy
from dask_ml.preprocessing import StandardScaler
from dask.distributed import Client, as_completed
from utils import *

#TODO:

# Look into computation time for var explained and super score convergence -> currently requires large amount of memory.
# Look into refactoring code to support async operations; otherwise, maybe no need for the separate mapping functions.

# Transform method
# Predict method
# SPE
# PRESS
# Total variance???
# Hotelling's T2

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
        self.X: list = []
        self.X_orig: list = []
        self.E: list = []
        self.t_T: da
        self.p_b: list = []
        self.t_b: da
        self.w_T: da
        self.t_T_new: da
        self.t_T_block: list = []
        self.variance: list = []
        self.curr_component: int = 0

    def set_initial_super_score(self):

        X_array = da.asarray(self.X)
        col_var = da.nanvar(X_array, axis = 1, ddof = 1)
        block_idx, col_idx = da.unravel_index(da.argmax(col_var, axis = None), col_var.shape)

        self.t_T = self.X[block_idx][:, col_idx].reshape(-1,1)

        del X_array, col_var, block_idx, col_idx 

    def set_super_score_block(self):

        self.t_T_block = self.client.persist([self.t_T for i in range(len(self.X))])

    def block_loadings(self, norm: bool = True):
        
        self.p_b = self.client.map(get_p_b, self.X, self.t_T_block, [norm for i in range(len(self.X))])
        self.p_b = self.client.gather(self.p_b)

    def block_scores(self):

        self.t_b = self.client.map(get_t_b, self.X, self.p_b)
        self.t_b = self.client.gather(self.t_b)
        self.t_b = da.hstack(self.t_b)
        self.t_b = self.client.persist(self.t_b)

    def super_weights(self):

        self.w_T = self.client.submit(get_w_T, self.t_b, self.t_T)
        self.w_T = self.w_T.result()

    def super_score(self):

        self.t_T_new = self.client.submit(get_t_T_new, self.t_b, self.w_T)
        self.t_T_new = self.client.persist(self.t_T_new.result())

    def get_eps(self):

        eps = self.t_T_new - self.t_T
        
        self.eps = ((eps.T @ eps) / (self.t_T.shape[0] * (self.t_T_new.T @ self.t_T_new))).compute().item()

    def has_converged(self):

        return self.eps < self.tol

    def deflate(self):

        self.E = self.client.map(get_residuals, self.X, self.t_T_block, self.p_b)
        self.E = self.client.gather(self.E)
        self.E = self.client.persist(self.E)

    def variance_explained(self):

        tr_E = self.client.map(eig_sum, self.E)

        var_exp = self.client.map(get_var_exp, self.tr_X, tr_E)
        var_exp = self.client.gather(var_exp)
        self.variance.append(self.client.persist(var_exp))
    
    def fit(self, X):

        self.X = X
        self._fit()

        return self

    def _fit(self):

        if self.solver == "nipals":
            self.tr_X = self.client.map(eig_sum, self.X)
            self._fit_nipals()

    def _fit_nipals(self):

        for i in range(self.n_components):

            self.set_initial_super_score()
            self.set_super_score_block()

            for j in range(self.iter):
                self.block_loadings(norm = True)
                self.block_scores()
                self.super_weights()
                self.super_score()
                self.get_eps()
                self.t_T = self.t_T_new
                self.set_super_score_block()

                print(progress_bar(j, self.iter, i, self.n_components, self.eps), end = '\r', flush = True)

                if self.has_converged():
                    break 

            self.block_loadings(norm = False)
            self.deflate()
            self.variance_explained()
            self.X = self.E
            self.curr_component += 1
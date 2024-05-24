import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import copy
from dask.distributed import Client, as_completed
from utils import *

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

    def set_initial_super_score(self, X):

        return X[0][:,0].reshape(-1,1)
        #self.set_super_score_block()

    def set_super_score_block(self, t_T, X):

        return self.client.persist([t_T for i in range(len(X))])

    def block_loadings(self, X, t_T_block):

        return self.client.map(get_p_b, X, t_T_block)
        
    def block_scores(self, X, p_b):

        t_b = self.client.map(get_t_b, X, p_b)
        t_b = self.client.gather(t_b)
        t_b = da.hstack(t_b)
        t_b = self.client.persist(t_b)
        return t_b

    def super_weights(self, t_b, t_T):

        return self.client.submit(get_w_T, t_b, t_T)

    def super_score(self, t_b, w_T):

        t_T_new = self.client.submit(get_t_T_new, t_b, w_T)
        t_T_new = self.client.persist(t_T_new.result())
        return t_T_new

    def get_eps(self, t_T_new, t_T):

        return (abs(t_T_new - t_T)).compute()
    
    def has_converged(self, eps):

        return (eps <= self.tol).all()

    def deflate(self, X, t_T_block):

        p_b = self.block_loadings(X, t_T_block)

        E = self.client.map(get_residuals, X, t_T_block, p_b)
        E = self.client.gather(E)
        E = self.client.persist(E)
        return E

    def variance_explained(self, X_orig, E):

        tr_X = self.client.map(eig_sum, X_orig)
        tr_E = self.client.map(eig_sum, E)

        var_exp = self.client.map(get_var_exp, tr_X, tr_E)
        var_exp = self.client.gather(var_exp)
        self.variance.append(self.client.persist(var_exp))

        return E
    
    def fit(self, X):

        self._fit(X, copy.deepcopy(X))

        return self

    def _fit(self, X, X_orig):

        if self.solver == "nipals":
            self._fit_nipals(X, X_orig)

    def _fit_nipals(self, X, X_orig):

        for i in range(self.n_components):

            super_score = self.set_initial_super_score(X)
            wide_super_score = self.set_super_score_block(super_score, X)

            for j in range(self.iter):
                block_loadings = self.block_loadings(X, wide_super_score)
                block_scores = self.block_scores(X, block_loadings)
                super_weights = self.super_weights(block_scores, super_score)
                super_score_new = self.super_score(block_scores, super_weights)
                eps = self.get_eps(super_score_new, super_score)

                super_score = super_score_new
                wide_super_score = self.set_super_score_block(super_score, X)

                print(progress_bar(j, self.iter, i, self.n_components, eps.max()), end = '\r', flush = True)

                if self.has_converged(eps):
                    break 
            
            E = self.deflate(X, wide_super_score)
            X = self.variance_explained(X_orig, E)
            self.curr_component += 1
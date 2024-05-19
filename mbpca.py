import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask_ml.preprocessing import StandardScaler
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
        self.X: list
        self.E: list
        self.t_T: da
        self.p_b: list
        self.t_b: da
        self.w_T: da
        self.t_T_new: da
        self.t_T_block: list

    def set_initial_super_score(self):

        self.t_T = self.X[0][:,0].reshape(-1,1)
        self.set_super_score_block()

    def set_super_score_block(self):

        self.t_T_block = self.client.persist([self.t_T for i in range(len(self.X))])

    def block_loadings(self):

        self.p_b = self.client.map(get_p_b, self.X, self.t_T_block, pure = False)
        
    def block_scores(self):

        self.t_b = self.client.map(get_t_b, self.X, self.p_b)
        self.t_b = self.client.gather(self.t_b)
        self.t_b = da.hstack(self.t_b)
        self.t_b = self.client.persist(self.t_b)

    def super_weights(self):

        self.w_T = self.client.submit(get_w_T, self.t_b, self.t_T, pure = False)

    def super_score(self):

        self.t_T_new = self.client.submit(get_t_T_new, self.t_b, self.w_T)
        self.t_T_new = self.client.persist(self.t_T_new.result())

    def check_convergence(self):

        self.eps = (abs(self.t_T_new - self.t_T)).compute()
        self.has_converged = (self.eps <= self.tol).all()
        self.max_eps = np.max(self.eps)

        self.t_T = self.t_T_new
        del self.t_T_new
        self.set_super_score_block()

    def deflate(self):

        self.E = self.client.map(get_residuals, self.X, self.t_T_block, self.p_b, pure = False)
        self.E = self.client.gather(self.E)
        self.E = self.client.persist(self.E)

        self.variance_explained()

        self.X = self.E
        del self.E

        self.has_converged = False

    def variance_explained(self):

        tr_X = self.client.map(eig_sum, self.X)
        tr_E = self.client.map(eig_sum, self.E)

        self.var_exp = self.client.map(get_var_exp, tr_X, tr_E)
        self.var_exp = self.client.gather(self.var_exp)
        self.var_exp = self.client.persist(self.var_exp)

        return
    
    def fit(self, X):

        self.X = X
        self._fit()

        return self

    def _fit(self):

        if self.solver == "nipals":
            self._fit_nipals()

    def _fit_nipals(self):

        self.set_initial_super_score()

        for i in range(self.n_components):
            for j in range(self.iter):
                self.block_loadings()
                self.block_scores()
                self.super_weights()
                self.super_score()
                self.check_convergence()
                print(progress_bar(j, self.iter, i, self.n_components, self.max_eps), end = '\r', flush = True)

                if self.has_converged:
                    break 
            
            self.deflate()

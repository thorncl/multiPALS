import dask.array as da
import numpy as np

def get_p_b(X_b: da, t_T: da, norm: bool):
    
    p_b = (X_b.T @ t_T) / (t_T.T @ t_T)

    if norm:
       p_b /= da.linalg.norm(p_b)

    return p_b

def get_t_b(X_b: da, p_b: da):

    return (X_b @ p_b) / da.sqrt(X_b.shape[1])

def get_w_T(t_b: da, t_T: da):

    w_T = (t_b.T @ t_T) / (t_T.T @ t_T)
    return w_T / da.linalg.norm(w_T)

def get_t_T_new(t_b: da, w_T: da):

    return t_b @ w_T

def get_residuals(X_b: da, t_T: da, p_b: da):

    return (X_b - (t_T @ p_b.T))

def eig_sum(X_b: da):
    
    sum_of_squares = X_b.T @ X_b
    return da.trace(sum_of_squares)

def get_var_exp(tr_X_b: da, tr_E_b: da):

    return abs(1 - (tr_E_b / tr_X_b))*100
    
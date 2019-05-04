
import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import math
import pickle
import time

from scipy.special import expit
from utilities import *
from gpr_pymc3 import * 

if __name__ == "__main__":
    N = 200
    p_arr = [200, 500]
    m0 = 5
    snr_arr = [2, 5]
    induce_arr = [20, 50, 100]
    for snr in snr_arr:
        for p in p_arr:
            print('Simulating for N = {0} and p = {1}'.format(N, p))
            X, y = make_simple_interaction_data(N, p, m0, snr)
            np.save('../data/synthetic/X_N_{0}_M_{1}_scale_{2}'.format(N, M, snr), X)
            np.save('../data/synthetic/y_N_{0}_M_{1}_scale_{2}'.format(N, M, snr), y)
            # Exact
            skim_model, skim_run, skim_gp, skim_time = SKIM_exact(X, y, m0, cores=2)
            dump_pymc3_model('../model/exact_N_{0}_M_{1}_scale_{2}'.format(N, M, snr), skim_run, skim_time)
            for n_induce in induce_arr:
                # Induce + Kmeans
                print('Simulating for N = {0} and p = {1} and n_induce = {2}'.format(N, p, n_induce))
                Xu, skim_model_induce, skim_run_induce, skim_gp_induce, skim_time_induce = SKIM_inducing(X, y, m0, 20, "FITC", False)
                np.save('../data/synthetic/Xu_N_{0}_M_{1}_scale_{2}_induce_{3}'.format(N, M, snr, n_induce), Xu)
                dump_pymc3_model('../model/fitc_N_{0}_M_{1}_scale_{2}_induce_{3}'.format(N, M, snr, n_induce), skim_run_induce, skim_time_induce)
                # Subsample
                skim_model_sub, skim_run_sub, skim_gp_sub, skim_time_sub = SKIM_exact(X[:n_induce, :], y[:n_induce], m0)
                dump_pymc3_model('../model/subsamp_N_{0}_M_{1}_scale_{2}_induce_{3}'.format(N, M, snr, n_induce), skim_run_sub, skim_time_sub)



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
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

import os
sys.path.insert(0, os.getcwd())
main_dir = os.path.basename(sys.modules['__main__'].__file__)
IS_RUN_WITH_SPHINX_GALLERY = main_dir != os.getcwd()

def do_all_exact(N, p, snr, m0=5):
    X = np.load('../data/synthetic/X_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    y = np.load('../data/synthetic/y_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    skim_model, skim_run, skim_gp, skim_time = SKIM_exact(X, y, m0)
    dump_pymc3_model(N, p, m0, '../model/exact_N_{0}_p_{1}_scale_{2}.pkl'.format(N, p, snr), skim_run, skim_time)


def do_all_induce(N, p, snr, n_induce, m0=5):
    X = np.load('../data/synthetic/X_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    y = np.load('../data/synthetic/y_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    # Induce + Kmeans
    print('Simulating for N = {0} and p = {1} and n_induce = {2}'.format(N, p, n_induce))
    Xu, skim_model_induce, skim_run_induce, skim_gp_induce, skim_time_induce = SKIM_inducing(X, y, m0, n_induce, "FITC", False)
    np.save('../data/synthetic/Xu_N_{0}_p_{1}_scale_{2}_induce_{3}'.format(N, p, snr, n_induce), Xu)
    dump_pymc3_model(N, p, m0, '../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce), skim_run_induce, skim_time_induce)
    # Subsample
    skim_model_sub, skim_run_sub, skim_gp_sub, skim_time_sub = SKIM_exact(X[:n_induce, :], y[:n_induce], m0)
    dump_pymc3_model(N, p, m0, '../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce), skim_run_sub, skim_time_sub)

if __name__ == "__main__":
    N_arr = [50]
    p_arr = [20, 50]
    m0 = 5
    snr_arr = [2]
    # induce_arr = [50, 100, 200]
    induce_arr = [20, 30]
    for N in N_arr:
        for snr in snr_arr:
            for p in p_arr:
                print('Simulating for N = {0} and p = {1}'.format(N, p))
                X, y = make_simple_interaction_data(N, p, m0, snr)
                np.save('../data/synthetic/X_N_{0}_p_{1}_scale_{2}'.format(N, p, snr), X)
                np.save('../data/synthetic/y_N_{0}_p_{1}_scale_{2}'.format(N, p, snr), y)
            

    print('==== Finished Generating datasets ====')
    Parallel(n_jobs=-2)(delayed(do_all_induce)(N=N, p=p, snr=snr, n_induce=n_induce) for N in N_arr for p in p_arr for snr in snr_arr for n_induce in induce_arr)
    Parallel(n_jobs=-2)(delayed(do_all_exact)(N=N, p=p, snr=snr) for N in N_arr for p in p_arr for snr in snr_arr)

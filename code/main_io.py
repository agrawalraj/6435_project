
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

import os
import sys

if __name__ == "__main__":
    N = int(sys.argv[1])
    p = int(sys.argv[2])
    snr = int(sys.argv[3])
    n_induce = int(sys.argv[4])
    m0 = 5
    exists = os.path.isfile('../data/synthetic/X_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    if exists:
        print('=== Loading existing dataset for N = {0}, p = {1}, snr = {2} ==='.format(N, p, snr))
        X = np.load('../data/synthetic/X_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
        y = np.load('../data/synthetic/y_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    else:
        print('=== Generating new dataset for N = {0}, p = {1}, snr = {2} ==='.format(N, p, snr))
        X, y = make_simple_interaction_data(N, p, m0, snr)
        np.save('../data/synthetic/X_N_{0}_p_{1}_scale_{2}'.format(N, p, snr), X)
        np.save('../data/synthetic/y_N_{0}_p_{1}_scale_{2}'.format(N, p, snr), y)
    if n_induce == 0:
        print('== Doing exact inference == ')
        skim_model, skim_run, skim_gp, skim_time = SKIM_exact(X, y, m0)
        dump_pymc3_model(N, p, m0, '../model/exact_N_{0}_p_{1}_scale_{2}.pkl'.format(N, p, snr), skim_run, skim_time)
    else:
        print('== Doing approx inference == ')
        print('Simulating for N = {0} and p = {1} and n_induce = {2}'.format(N, p, n_induce))
        Xu, skim_model_induce, skim_run_induce, skim_gp_induce, skim_time_induce = SKIM_inducing(X, y, m0, n_induce, "FITC", False)
        np.save('../data/synthetic/Xu_N_{0}_p_{1}_scale_{2}_induce_{3}'.format(N, p, snr, n_induce), Xu)
        dump_pymc3_model(N, p, m0, '../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce), skim_run_induce, skim_time_induce)
        # Subsample
        skim_model_sub, skim_run_sub, skim_gp_sub, skim_time_sub = SKIM_exact(X[:n_induce, :], y[:n_induce], m0)
        dump_pymc3_model(N, p, m0, '../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce), skim_run_sub, skim_time_sub)


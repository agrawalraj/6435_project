
import pymc3 as pm
import pandas as pd
import numpy as np
import theano.tensor as tt
import math
import pickle
import time

from scipy.special import expit
from utilities import *

def square_scale_input(x, scale_vec):
    return tt.mul(x, tt.mul(x, scale_vec))

def scale_input(x, scale_vec):
    return tt.mul(x, scale_vec)

def dump_pymc3_model(dump_file_name, run, run_time):
    output = open(dump_file_name, 'wb')
    pickle.dump([run, run_time], output)
    output.close()
    print('Saved PyMC3 run data in ' + dump_file_name)

def load_pymc3_run(file_name):
    pkl_file = open(file_name, 'rb')
    run, run_time = pickle.load(pkl_file)
    return run, run_time

def dump_pymc3_run(dump_file_name, run, N, p, m0, slab_scale=3):
    pymc3_dict = {}
    lambda_local_scale = run['lambda_local_scale']
    pymc3_dict['sigma'] = run['sigma']
    pymc3_dict['psi'] = run['psi']
    eta_1_base = run['eta_1_base']
    m_base = run['m_base']
    phi = (m0 / (p - m0)) * (sigma / math.sqrt(1.0 * N))
    pymc3_dict['c'] = run['c']
    pymc3_dict['eta_1'] = phi * eta_1_base
    pymc3_dict['m_sq'] = slab_scale ** 2 * m_base
    pymc3_dict['kappa'] = np.sqrt(np.multiply(pymc3_dict['m_sq'], (lambda_local_scale ** 2).T) / (pymc3_dict['m_sq'] + np.multiply(pymc3_dict['eta_1'] ** 2, (lambda_local_scale ** 2).T))).T
    output = open(dump_file_name, 'wb')
    pickle.dump(pymc3_dict, output)
    output.close()
    print('Saved PyMC3 run data in ' + dump_file_name)

def SKIM_exact(X, y, m0, slab_scale=3, slab_df=25, n_iter=1000, chains=2):
    N, p = X.shape
    half_slab_df = 0.5 * slab_df
    alpha = 0 # No quadratic effects
    X2 = X ** 2
    with pm.Model() as model:
        # Sampled latent parameters
        lambda_local_scale = pm.HalfCauchy('lambda_local_scale', beta=1, shape=(p, ))
        sigma = pm.HalfNormal('sigma', sd=1)
        psi = pm.InverseGamma('psi', alpha=half_slab_df, beta=half_slab_df)
        eta_1_base = pm.HalfCauchy('eta_1_base', beta=1)
        m_base = pm.InverseGamma('m_base', alpha=half_slab_df, beta=half_slab_df) 
        c = pm.Normal('c', mu=0, sd=1)

        # Transformed parameters
        phi = (m0 / (p - m0)) * (sigma / math.sqrt(1.0 * N))
        eta_1 = phi * eta_1_base # Global scale for linear effects
        m_sq = slab_scale ** 2 * m_base
        kappa = tt.sqrt(m_sq * lambda_local_scale ** 2 / (m_sq + eta_1 ** 2 * lambda_local_scale ** 2))
        eta_2 = eta_1 ** 2 / m_sq * psi # Global prior variance of interaction terms

        # SKIM covariance function
        eta_2_sq = eta_2 ** 2
        cov1 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=2, offset=1.0)
        cov1_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov1) 
        cov2 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=c ** 2 - .5 * eta_2_sq - 1)
        cov2_scaled = pm.gp.cov.WarpedInput(p, warp_func=square_scale_input, args=(kappa), cov_func=cov2)
        cov3 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=0)
        cov3_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov3) 
        cov_final = .5 * eta_2_sq * cov1_scaled + (alpha ** 2 - .5 * eta_2_sq) * cov2_scaled + (eta_1 ** 2 - eta_2_sq) * cov3_scaled
        gp = pm.gp.Marginal(cov_func=cov_final)
        
        # Observed
        y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

        # Sample values
        t0 = time.time()
        trace = pm.sample(n_iter, chains=chains) # defaults to NUTS
        tot = time.time() - t0
        # trace = pm.sample(n_iter, chains=chains)
        # trace = pm.find_MAP(maxeval=5000)
    return model, trace, gp, tot

def SKIM_inducing(X, y, m0, n_inducing, inducing_method, learn_induce_pts, slab_scale=3, slab_df=25, n_iter=1000, chains=2):
    N, p = X.shape
    half_slab_df = 0.5 * slab_df
    alpha = 0 # No quadratic effects
    X2 = X ** 2
    with pm.Model() as model:
        # Sampled latent parameters
        lambda_local_scale = pm.HalfCauchy('lambda_local_scale', beta=1, shape=(p, ))
        sigma = pm.HalfNormal('sigma', sd=1)
        psi = pm.InverseGamma('psi', alpha=half_slab_df, beta=half_slab_df)
        eta_1_base = pm.HalfCauchy('eta_1_base', beta=1)
        m_base = pm.InverseGamma('m_base', alpha=half_slab_df, beta=half_slab_df) 
        c = pm.Normal('c', mu=0, sd=1)

        # Transformed parameters
        phi = (m0 / (p - m0)) * (sigma / math.sqrt(1.0 * N))
        eta_1 = phi * eta_1_base # Global scale for linear effects
        m_sq = slab_scale ** 2 * m_base
        kappa = tt.sqrt(m_sq * lambda_local_scale ** 2 / (m_sq + eta_1 ** 2 * lambda_local_scale ** 2))
        eta_2 = eta_1 ** 2 / m_sq * psi # Global prior variance of interaction terms

        # SKIM covariance function
        eta_2_sq = eta_2 ** 2
        cov1 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=2, offset=1.0)
        cov1_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov1) 
        cov2 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=c ** 2 - .5 * eta_2_sq - 1)
        cov2_scaled = pm.gp.cov.WarpedInput(p, warp_func=square_scale_input, args=(kappa), cov_func=cov2)
        cov3 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=0)
        cov3_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov3) 
        cov_final = .5 * eta_2_sq * cov1_scaled + (alpha ** 2 - .5 * eta_2_sq) * cov2_scaled + (eta_1 ** 2 - eta_2_sq) * cov3_scaled
        gp = pm.gp.MarginalSparse(cov_func=cov_final, approx=inducing_method)
        
        # Observed
        if learn_induce_pts == False:
            Xu = pm.gp.util.kmeans_inducing_points(n_inducing, X)
        else:
            Xu = pm.Flat("Xu", shape=(n_inducing, p))
        y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, noise=sigma)

        # Sample values
        t0 = time.time()
        trace = pm.sample(n_iter, chains=chains) # defaults to NUTS
        tot = time.time() - t0
    return Xu, model, trace, gp, tot 

def SKIM_GLM(X, y, m0, slab_scale=3, slab_df=25, n_iter=1000, chains=2, cores=1):
    N, p = X.shape
    half_slab_df = 0.5 * slab_df
    alpha = 0 # No quadratic effects
    X2 = X ** 2
    with pm.Model() as model:
        # Sampled latent parameters
        lambda_local_scale = pm.HalfCauchy('lambda_local_scale', beta=1, shape=(p, ))
        sigma = pm.HalfNormal('sigma', sd=1)
        psi = pm.InverseGamma('psi', alpha=half_slab_df, beta=half_slab_df)
        eta_1_base = pm.HalfCauchy('eta_1_base', beta=1)
        m_base = pm.InverseGamma('m_base', alpha=half_slab_df, beta=half_slab_df) 
        c = pm.Normal('c', mu=0, sd=1)

        # Transformed parameters
        phi = (m0 / (p - m0)) * (sigma / math.sqrt(1.0 * N))
        eta_1 = phi * eta_1_base # Global scale for linear effects
        m_sq = slab_scale ** 2 * m_base
        kappa = tt.sqrt(m_sq * lambda_local_scale ** 2 / (m_sq + eta_1 ** 2 * lambda_local_scale ** 2))
        eta_2 = eta_1 ** 2 / m_sq * psi # Global prior variance of interaction terms

        # SKIM covariance function
        eta_2_sq = eta_2 ** 2
        cov1 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=2, offset=1.0)
        cov1_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov1) 
        cov2 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=c ** 2 - .5 * eta_2_sq - 1)
        cov2_scaled = pm.gp.cov.WarpedInput(p, warp_func=square_scale_input, args=(kappa), cov_func=cov2)
        cov3 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=0)
        cov3_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov3) 
        cov_final = .5 * eta_2_sq * cov1_scaled + (alpha ** 2 - .5 * eta_2_sq) * cov2_scaled + (eta_1 ** 2 - eta_2_sq) * cov3_scaled
        gp = pm.gp.Latent(cov_func=cov_final)
        
        # Observed
        f = gp.prior("f", X=X)
        probs = pm.Deterministic("probs", pm.math.invlogit(f))
        y_ = pm.Bernoulli("y", p=probs, observed=y)

        # Sample value
        t0 = time.time()
        trace = pm.sample(n_iter, chains=chains, cores=cores) # defaults to NUTS
        tot = time.time() - t0
    return model, trace, gp, tot 

# def main_effect_prediction_helper(p, run, run_gp):
#     X_main_effects = np.zeros((2 * p, p))
#     for i in range(p):
#         X_main_effects[2*i, i] = 1
#         X_main_effects[2*i + 1, i] = -1
#     mean_X_main_arr = []
#     cov_X_main_arr = []
#     for i, point in enumerate(run.points()):
#         mean_X_main, cov_X_main = run_gp.predict(X_main_effects, point, diag=True)
#         if i % 50 == 0:
#             print('At iteration {0} for main effects'.format(i))
#     return mean_X_main_arr, cov_X_main_arr

# N = 50
# p = 20
# m0 = 5
# snr_stength = 5
# X, y = make_simple_interaction_data(N, p, m0, snr_stength)
# skim_model, skim_run, skim_gp, skim_time = SKIM_exact(X, y, m0, cores=2)
# pm.summary(skim_run)

# points = []
# for point in skim_run.points():
#     points.append(point)

# skim_gp.predict(X_main_effects, points[0])


# Xu, skim_model_induce, skim_run_induce, skim_gp_induce, skim_time_induce = SKIM_inducing(X, y, m0, 20, "FITC", False, cores=4)

# dump_pymc3_run('delete.pkl', skim_run_induce, N, p, m0)


# a = main_effect_predict_SKIM(skim_run)

# skim_model, skim_run = SKIM_FITC(X, y, m0, gp_scheme=pm.gp.MarginalSparse)


# skim_model_vfe, skim_run_vfe = SKIM_VFE(X, y, m0)

# N = 100
# p = 20
# m0 = 5
# snr_stength = 5
# X, y = make_simple_binary_data(N, M, m0, snr_stength)
# skim_glm_model, skim_glm_run = SKIM_GLM(X, y, m0)

# b = pm.summary(skim_glm_run)

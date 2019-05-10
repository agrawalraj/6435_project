
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

def dump_pymc3_model(N, p, m0, dump_file_name, run, run_time, slab_scale=3):
    output = open(dump_file_name, 'wb')
    run_dict = dump_pymc3_run_helper(run, N, p, m0, slab_scale)
    pickle.dump([run_dict, pm.summary(run), run_time], output)
    output.close()
    print('Saved PyMC3 run data in ' + dump_file_name)

def load_pymc3_run(file_name):
    pkl_file = open(file_name, 'rb')
    run, samp_stats, run_time = pickle.load(pkl_file)
    return run, samp_stats, run_time

def dump_pymc3_run_helper(run, N, p, m0, slab_scale=3):
    pymc3_dict = {}
    lambda_local_scale = run['lambda_local_scale']
    pymc3_dict['sigma'] = run['sigma']
    pymc3_dict['psi'] = run['psi']
    eta_1_base = run['eta_1_base']
    m_base = run['m_base']
    phi = (m0 / (p - m0)) * (pymc3_dict['sigma'] / math.sqrt(1.0 * N))
    pymc3_dict['c'] = run['c']
    pymc3_dict['eta_1'] = np.multiply(phi, eta_1_base)
    pymc3_dict['m_sq'] = slab_scale ** 2 * m_base
    pymc3_dict['kappa'] = np.sqrt(np.multiply(pymc3_dict['m_sq'], (lambda_local_scale ** 2).T) / (pymc3_dict['m_sq'] + np.multiply(pymc3_dict['eta_1'] ** 2, (lambda_local_scale ** 2).T))).T
    return pymc3_dict

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

def get_marginal_joint_gauss(A, mean_vec, cov_mat):
    # y = Ax + b, x ~ MVN(mean, cov_mat)
    new_mean = A.dot(mean_vec)[0]
    new_cov_mat = A.dot(cov_mat.dot(A.T))[0][0]
    return (new_mean, new_cov_mat)

def skim_induce_gp_pred(X, y, Xu, induce, c, kappa, eta_1, m_sq, psi, sigma, alpha=0, induce_method='FITC'):
    N, p = X.shape
    alpha = 0 # No quadratic effects
    X2 = X ** 2
    with pm.Model() as model:
        eta_2 = eta_1 ** 2 / m_sq * psi
        eta_2_sq = eta_2 ** 2
        cov1 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=2, offset=1.0)
        cov1_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov1) 
        cov2 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=c ** 2 - .5 * eta_2_sq - 1)
        cov2_scaled = pm.gp.cov.WarpedInput(p, warp_func=square_scale_input, args=(kappa), cov_func=cov2)
        cov3 = pm.gp.cov.Polynomial(p, c=tt.zeros(p), d=1, offset=0)
        cov3_scaled = pm.gp.cov.WarpedInput(p, warp_func=scale_input, args=(kappa), cov_func=cov3) 
        cov_final = float(.5 * eta_2_sq) * cov1_scaled + float(alpha ** 2 - .5 * eta_2_sq) * cov2_scaled + float(eta_1 ** 2 - eta_2_sq) * cov3_scaled
        if induce:
            gp = pm.gp.MarginalSparse(cov_func=cov_final, approx=induce_method)
            y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, noise=sigma)
        else:
            gp = pm.gp.Marginal(cov_func=cov_final)
            y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)
    X_main_effects = np.zeros((2*p, p))
    for i in range(p):
        X_main_effects[2*i, i] = 1
        X_main_effects[2*i + 1, i] = -1
    with model:
        f_mean, f_cov = gp.predict(X_main_effects)
    means = []
    model = 0 # free memory
    variances = []
    for i in range(p):
        A = np.array([[.5, -.5]])
        param_mean, param_var = get_marginal_joint_gauss(A, f_mean[(2*i):(2*i + 2)], f_cov[(2*i):(2*i + 2), (2*i):(2*i + 2)])
        means.append(param_mean)
        variances.append(param_var)
    return means, variances

def get_main_effects(X, y, mcmc_file_name, Xu=None, induce=False, sig_thresh=1.96, induce_method='FITC', thin_factor=10):
    N, p = X.shape
    pkl_file = open(mcmc_file_name, 'rb')
    mcmc_dict = pickle.load(pkl_file)[0]
    c = mcmc_dict['c']
    kappa = mcmc_dict['kappa']
    eta_1 = mcmc_dict['eta_1']
    m_sq = mcmc_dict['m_sq']
    psi = mcmc_dict['psi']
    sigma = mcmc_dict['sigma']
    n_samps = sigma.shape[0]
    samp_means_main = []
    samp_vars_main = []
    print('WARNING - THINNING BY A Factor {0}'.format(thin_factor))
    for i in range(0, n_samps, thin_factor):
        try:
            pred_mean, pred_var = skim_induce_gp_pred(X, y, Xu, induce, mcmc_dict['c'][i], mcmc_dict['kappa'][i], mcmc_dict['eta_1'][i], mcmc_dict['m_sq'][i], mcmc_dict['psi'][i], mcmc_dict['sigma'][i], induce_method=induce_method)
            samp_means_main.append(pred_mean)
            samp_vars_main.append(pred_var)
        except:
            print('Probably PD Error...check if error keeps coming up')
            continue
        print('At iteration {0} for main effects'.format(i))
    means_main_mat = np.array(samp_means_main)
    sd_main_mat = np.sqrt(np.array(samp_vars_main))
    avg_main_effects = np.nanmean(means_main_mat, axis=0)
    avg_main_sds = np.nanmean(sd_main_mat, axis=0)
    significant_idcs = get_significant_indcs(avg_main_effects, avg_main_sds, sig_thresh)
    return (means_main_mat, sd_main_mat, significant_idcs)

if __name__ == "__main__":
    N = 1000
    p = 500
    m0 = 5
    snr = 1
    induce_arr = [100, 200, 500]
    X = np.load('../data/synthetic/X_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    y = np.load('../data/synthetic/y_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr))
    #mcmc_run_path_exact = '../model/exact_N_{0}_p_{1}_scale_{2}.pkl'.format(N, p, snr)
    #mcmc_exact_params = get_main_effects(X, y, mcmc_run_path_exact, Xu=None, induce=False)
    #np.save('../summary_stats/exact_master_params_N_{0}_p_{1}_scale_{2}'.format(N, p, snr), mcmc_exact_params)
    #print('== Finished exact ==')
    for n_induce in induce_arr:
        print('== Doing for n_induce = {0} =='.format(n_induce))
        print('== Doing subsampled ==')
        mcmc_run_path_subsam = '../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce)
        mcmc_sub_params = get_main_effects(X[:n_induce, :], y[:n_induce], mcmc_run_path_subsam, Xu=None, induce=False)
        np.save('../summary_stats/subsamp_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}'.format(N, p, snr, n_induce), mcmc_sub_params)
        print('== Doing induce ==')
        Xu = np.load('../data/synthetic/Xu_N_{0}_p_{1}_scale_{2}_induce_{3}.npy'.format(N, p, snr, n_induce))
        mcmc_run_path_fitc = '../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce)
        mcmc_induce_params = get_main_effects(X, y, mcmc_run_path_fitc, Xu=Xu, induce=True)
        np.save('../summary_stats/fitc_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}'.format(N, p, snr, n_induce), mcmc_induce_params)

# N = 500
# p = 200
# m0 = 5
# snr_stength = 2
# X, y = make_simple_interaction_data(N, p, m0, snr_stength)
# skim_model, skim_run, skim_gp, skim_time = SKIM_exact(X, y, m0, n_iter=5)
# pm.summary(skim_run)

# points = []
# for point in skim_run.points():
#     points.append(point)

# skim_gp.predict(X_main_effects, points[0])


# Xu, skim_model_induce, skim_run_induce, skim_gp_induce, skim_time_induce = SKIM_inducing(X, y, m0, 20, "FITC", False)


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

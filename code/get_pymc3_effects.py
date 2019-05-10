
import torch
import pickle

torch.set_default_dtype(torch.float64) # For numerical stability 
torch.set_default_tensor_type(torch.DoubleTensor)

import numpy as np
import pyro
import pyro.contrib.gp as gp
from torch.nn import Parameter
from pyro.params import param_with_module_name
from pyro.contrib.gp.kernels import Polynomial, Sum
from utilities import get_significant_indcs

import os
import sys

class ScaledPolynomialKernel(Polynomial):
    def __init__(self, input_dim, variance=None, bias=None, degree=1, active_dims=None,
                 name="ScaledPolynomial"):
        super().__init__(input_dim, variance, bias, degree, active_dims, name)
    def forward(self, scale, X, Z=None, diag=False):
        if Z is None:
            Z = X
        return super(ScaledPolynomialKernel, self).forward(torch.mul(X, scale), torch.mul(Z, scale), diag)

class PairwiseInteractionKernelFixed(Polynomial):
    def __init__(self, input_dim, alpha, eta, phi, psi, c, scale,
        variance=None, active_dims=None, bias=None, name="pairwise_kernel_fixed"):
        super().__init__(input_dim, variance, bias, 1, active_dims, name)
        self.scale = scale
        self.kern1 = ScaledPolynomialKernel(input_dim, variance, bias=torch.tensor(1.), degree=2, active_dims=active_dims)
        self.kern2 = ScaledPolynomialKernel(input_dim, variance, bias=torch.tensor(0.), degree=1, active_dims=active_dims)
        self.kern3 = ScaledPolynomialKernel(input_dim, variance, bias=torch.tensor(0.), degree=1, active_dims=active_dims)
        self.phi = phi
        self.psi = psi
        self.input_dim = input_dim
        self.c = c
        self.alpha = self.phi ** 2 / self.c ** 2 * alpha
        self.eta = self.phi ** 2 / self.c ** 2 * eta
        self.scales = [] # DELETE
    def forward(self, X, Z=None, diag=False):
        scale = self.get_param('scale')
        kappa = torch.sqrt(scale)
        if Z is None:
            Z = X
        return self.eta ** 2 / 2 * self.kern1.forward(kappa, X, Z, diag) + (self.alpha ** 2 - self.eta ** 2 / 2) * self.kern2.forward(kappa, X ** 2, Z ** 2, diag) + (self.phi ** 2 - self.eta ** 2) * self.kern3.forward(kappa, X, Z, diag) + (self.psi ** 2 - self.eta ** 2 / 2)

def get_marginal_joint_gauss(A, mean_vec, cov_mat):
    # y = Ax + b, x ~ MVN(mean, cov_mat)
    new_mean = A.mv(mean_vec)[0]
    new_cov_mat = A.mm(cov_mat.mm(A.t()))[0][0]
    return (new_mean, new_cov_mat)

def get_main_effects_fast(X_train, y_train, Xu, induce, alpha, eta, phi, psi, c, sigma_sq, scale_temp, induce_method='FITC'):
    input_dim = X_train.shape[1]
    pairwise_interaction_kernel_temp = PairwiseInteractionKernelFixed(input_dim, alpha, eta, phi, psi, c, scale_temp)
    if induce == False:
        gpr_temp = gp.models.GPRegression(X_train, y_train, pairwise_interaction_kernel_temp, jitter=sigma_sq)
    else:
        gpr_temp = gp.models.SparseGPRegression(X_train, y_train, pairwise_interaction_kernel_temp, Xu, jitter=sigma_sq, approx=induce_method)
    means = []
    variances = []
    X_test = torch.zeros((2 * input_dim, input_dim))
    for i in range(input_dim):
        X_test[2*i, i] = 1
        X_test[2*i + 1, i] = -1
    f_mean, f_cov = gpr_temp(X_test, full_cov=True)
    for i in range(input_dim):
        A = torch.tensor([[.5, -.5]])
        param_mean, param_var = get_marginal_joint_gauss(A, f_mean[(2*i):(2*i + 2)], f_cov[(2*i):(2*i + 2), (2*i):(2*i + 2)])
        means.append(np.float(param_mean))
        variances.append(np.float(param_var))
    return (means, variances)

def gpr_predict(X_train, y_train, X_test, alpha, eta, phi, psi, c, sigma_sq, scale_temp):
    input_dim = X_train.shape[1]
    pairwise_interaction_kernel_temp = PairwiseInteractionKernelFixed(input_dim, alpha, eta, phi, psi, c, scale_temp)
    gpr_temp = gp.models.GPRegression(X_train, y_train, pairwise_interaction_kernel_temp, jitter=sigma_sq)
    param_mean, param_var = gpr_temp(X_test, full_cov=False)
    param_mean = [np.float(e) for e in param_mean]
    param_var = [np.float(e) for e in param_var]
    return (param_mean, param_var)

def pymc3_variable_selection(X, y, mcmc_file_name, Xu=None, induce=False, sig_thresh=1.96, induce_method='FITC'):
    # A really weird bug w/ pyro where have to call MCMC run
    # Load in sampled parameters 
    pkl_file = open(mcmc_file_name, 'rb')
    mcmc_dict = pickle.load(pkl_file)[0] 
    c_samps = torch.tensor(np.sqrt(mcmc_dict['m_sq']))
    num_samps = c_samps.shape[0]
    scale_samps = torch.tensor(mcmc_dict['kappa'] ** 2)
    eta_base_samps = torch.tensor(mcmc_dict['psi'])
    alpha_samps = torch.tensor(torch.zeros(num_samps)) # No quadratic effects
    phi_samps = torch.tensor(mcmc_dict['eta_1'])
    psi_samps = torch.tensor(mcmc_dict['c'])
    sigma_sq = torch.tensor(mcmc_dict['sigma'] ** 2)
    samp_means_main = []
    samp_vars_main = []
    for i in range(num_samps):
        try:
            means, variances = get_main_effects_fast(X, y, Xu, induce, alpha_samps[i], eta_base_samps[i], phi_samps[i], psi_samps[i], c_samps[i], sigma_sq[i], scale_samps[i], induce_method)
            samp_means_main.append(means)
            samp_vars_main.append(variances)
        except:
            print('Probably PD Error...check if error keeps coming up')
            continue
        if i % 100 == 0:
            print('At iteration {0} for main effects'.format(i))
    means_main_mat = np.array(samp_means_main)
    sd_main_mat = np.sqrt(np.array(samp_vars_main))
    avg_main_effects = np.nanmean(means_main_mat, axis=0)
    avg_main_sds = np.nanmean(sd_main_mat, axis=0)
    significant_idcs = get_significant_indcs(avg_main_effects, avg_main_sds, sig_thresh)
    return (means_main_mat, sd_main_mat, significant_idcs)

# def pymc3_prediction(X_train, y_train, X_test, mcmc_file_name, sig_thresh=2.58, psi=1.):
#     # A really weird bug w/ pyro where have to call MCMC run
#     # Load in sampled parameters 
#     pkl_file = open(mcmc_file_name, 'rb')
#     mcmc_dict = pickle.load(pkl_file)[0] 
#     c_samps = torch.tensor(np.sqrt(mcmc_dict['m_sq']))
#     num_samps = c_samps.shape[0]
#     scale_samps = torch.tensor(mcmc_dict['kappa'] ** 2)
#     eta_base_samps = torch.tensor(mcmc_dict['psi'])
#     alpha_samps = torch.tensor(torch.zeros(num_samps)) # No quadratic effects
#     phi_samps = torch.tensor(mcmc_dict['eta_1']) j
#     psi_samps = torch.tensor(mcmc_dict['c'])
#     sigma_sq = torch.tensor(mcmc_dict['sigma'] ** 2)
#     samp_means_main = []
#     samp_vars_main = []
#     for i in range(num_samps):
#         try:
#             means, variances = gpr_predict(X_train, y_train, X_test, alpha_samps[i], eta_base_samps[i], phi_samps[i], psi_samps[i], c_samps[i], sigma_sq[i], scale_samps[i])
#             samp_means_main.append(means)
#             samp_vars_main.append(variances)
#         except:
#             print('Probably PD Error...check if error keeps coming up')
#             continue
#         if i % 100 == 0:
#             print(i)
#     return np.array(samp_means_main), np.array(samp_vars_main)

if __name__ == "__main__":
N = 1000
p = 500
m0 = 5
snr = 1
induce_arr = [50, 100, 200, 500]
X = torch.tensor(np.load('../data/synthetic/X_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr)))
y = torch.tensor(np.load('../data/synthetic/y_N_{0}_p_{1}_scale_{2}.npy'.format(N, p, snr)))
# mcmc_run_path_exact = '../model/exact_N_{0}_p_{1}_scale_{2}.pkl'.format(N, p, snr)
# mcmc_exact_params = pymc3_variable_selection(X, y, mcmc_run_path_exact, Xu=None, induce=False)
# np.save('../summary_stats/exact_master_params_N_{0}_p_{1}_scale_{2}'.format(N, p, snr), mcmc_exact_params)
# print('== Finished exact ==')
for n_induce in induce_arr:
    print('== Doing for n_induce = {0} =='.format(n_induce))
    print('== Doing induce ==')
    # Xu = torch.tensor(np.load('../data/synthetic/Xu_N_{0}_p_{1}_scale_{2}_induce_{3}.npy'.format(N, p, snr, n_induce)))
    # mcmc_run_path_fitc = '../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce)
    # mcmc_induce_params = pymc3_variable_selection(X, y, mcmc_run_path_fitc, Xu=Xu, induce=True)
    # np.save('../summary_stats/fitc_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}'.format(N, p, snr, n_induce), mcmc_induce_params)
    print('== Doing subsampled ==')
    mcmc_run_path_subsam = '../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(N, p, snr, n_induce)
    mcmc_sub_params = pymc3_variable_selection(X[:n_induce, :], y[:n_induce], mcmc_run_path_subsam, Xu=None, induce=False)
    np.save('../summary_stats/subsamp_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}'.format(N, p, snr, n_induce), mcmc_sub_params)
    
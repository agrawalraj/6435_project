
import numpy as np
import pandas as pd
from scipy.stats.stats import linregress
import matplotlib.pyplot as plt
import pickle
import os

def pairwise_interaction_map_no_quad(X):
    N, p = X.shape
    num_interactions = int(p * (p - 1) / 2)
    p_new = int(num_interactions + p + 1)
    X_new = np.zeros((N, p_new))
    count = 0
    for i in range(p-1):
        for j in range(i+1, p):
            X_new[:, count] = X[:, i] * X[:, j]
            count += 1
    for i in range(p):
        X_new[:, (num_interactions + i)] = X[:, i]
    X_new[:, (p_new-1)] = 1
    return X_new

def get_significant_indcs(avg_main_effects, avg_main_sds, sig_thresh, tol=.05):
  signficant_indcs = []
  for i in range(avg_main_effects.shape[0]):
    lower = avg_main_effects[i] - avg_main_sds[i] * sig_thresh
    upper = avg_main_effects[i] + avg_main_sds[i] * sig_thresh
    if np.sign(lower) == np.sign(upper): # 0 not contained in interval --> signficant parameter
      if np.abs(avg_main_effects[i]) > tol:
        signficant_indcs.append(i)
  return np.array(signficant_indcs)

def make_simple_interaction_data(N, M, m0, scale, noise_scale=None):
  X = np.random.normal(size=(N, M), scale=scale)
  true_main_effects = np.zeros(M)
  true_main_effects[:m0] = 1
  true_main_effects = true_main_effects.reshape((1, M))
  all_effects = pairwise_interaction_map_no_quad(true_main_effects)
  beta_true = all_effects.flatten()
  if noise_scale == None:
    noise_scale = m0
  y = pairwise_interaction_map_no_quad(X).dot(beta_true) + np.random.normal(scale=noise_scale, size = (N, ))
  return (X, y)

def make_simple_binary_data(N, M, m0, scale):
  X, scores = make_simple_interaction_data(N, M, m0, scale, 0)
  probs = expit(scores)
  y = np.array([np.random.binomial(1, prob) for prob in probs])
  return (X, y)

 



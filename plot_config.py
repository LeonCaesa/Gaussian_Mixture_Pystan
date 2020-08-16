#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Sat Jun  6 14:54:44 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
from itertools import cycle
regex= r'.*?([0-9].*?)\.pickle'
pat=re.compile(regex)
import seaborn as sns

def az_v_theta_plot(stan_fit):
        """
        Function to demonstrate pystan theta convergence result through R_hat table, autocorrelation (3 chians), and trace plot
        """
        az.plot_trace(stan_fit, var_names=['v','theta'], filter_vars="like")
        print(stan_fit.stansummary())
        az.plot_autocorr(stan_fit, var_names=["v",'theta'])
        az.plot_pair(stan_fit, var_names=["v",'theta'], divergences=True)
        
def az_mu_sigma_plot(stan_fit):
        """
        Function to demonstrate pystan theta convergence result through R_hat table, autocorrelation (3 chians), and trace plot
        """
        az.plot_trace(stan_fit, var_names=['sigma2','mu'], filter_vars="like")
        az.plot_autocorr(stan_fit, var_names=['sigma2',"mu"])
        az.plot_pair(stan_fit, var_names=['sigma2',"mu"], divergences=True)
        

def plot(fit, threshold=0.02):
    """
        Function to plot the posterior P(K=j,|Y)s whose values are greater than threshold
        params: fit, stan fitted result with setup in Robust Bayesian Inference via Coarsening Jeffrey W. Miller & David B. Dunson (2018)
        5.1. Simulation Example: Perturbed Mixture of Gaussians
        return: predicted posterior
    """
    theta_list = fit['fit'].extract('theta')['theta']
    flag = np.mean(theta_list, axis=0) > threshold
    pd.DataFrame(theta_list[:, flag]).plot()
    plt.title('P(K=j|Y)')
    plt.xlabel('Step')
    plt.ylabel('P(K=j|Y)')

    pred_y = []
    for j in range(np.shape(fit['fit'].to_dataframe())[0]):

        k_temp = np.random.multinomial(1, fit['fit'].extract('theta')['theta'][j, :])
        mu_temp = fit['fit'].extract('mu')['mu'][j, :]
        sigma_temp = fit['fit'].extract('sigma2')['sigma2'][j, :]

        pos_mu = np.dot(mu_temp.T, k_temp)
        pos_sigma = np.dot(sigma_temp.T, k_temp)
        pred_y.append(np.random.normal(pos_mu, np.sqrt(pos_sigma)))

    return pred_y


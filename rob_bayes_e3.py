#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Fri May 22 14:42:54 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import pystan
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import choice
import seaborn as sns
import pandas as pd
# np.set_printoptions(precision=2)


def mixture_generation(config):
    """
        Function to sample from true gaussian mixture model
        params: w0, weights of probability
        params: mu0, weights of component gaussian mean
        params: sigma0, weights of component gaussian sigma  
        return: a sample of Gaussian Mixture
    """
    w0 = np.array(config['w0'])
    mu0 = np.array(config['mu0'])
    sigma0 = config['sigma0']
    samples = norm(mu0, sigma0).rvs()
    return choice(samples, p=w0)


def base_measure(): return mixture_generation(w0, mu0, sigma0)


def CRP(n_sample, alpha=5):
    """
        Function to sample a partition from Chinese Resturant Process
        params: n_sample, #of data
        params: alpha, divergence parameter        
        return: a list of partition 
    """

    count = []
    n = 0
    while n < n_sample:
        prob = np.zeros(len(count) + 1)

        for i in range(len(count)):  # for the exiesting tables:
            prob[i] = count[i] / (n + alpha)  # prob of i-th table assignment

        prob[-1] = alpha / (n + alpha)  # new table prob
        prob = prob / sum(prob)

        assignment = choice(range(len(prob)), p=prob)

        if assignment == len(count):  # new table created
            count.append(0)
        count[assignment] += 1
        n += 1

    return count


def sample_perturbation(config):
    """
        Function to sample perturbation Po by taking a random draw of a Polya urn scheme 
        Dirichlet process mixture with base distribution PθI , default concentration parameter 500.
        params: w0, weights of probability
        params: mu0, weights of component gaussian mean
        params: sigma0, weights of component gaussian sigma  
        params: n_sample # of data
        parama: alpha, CRP divergence parameter
        return: pertubated sample
    """
    n_sample = int(config['n_sample'])
    alpha = float(config['alpha'])

    count = CRP(n_sample, alpha)

    unique_sample = [mixture_generation(
        config) for _ in range(len(count))]

    repeat_sample = np.repeat(unique_sample, count)

    return np.random.normal(repeat_sample, 0.25)



if __name__ == '__main__':
    w0 = np.array([0.5, 0.5])
    mu0 = np.array([-2, 2])
    sigma0 = np.array([0.7, 0.8])
    n_sample = 2000
    alpha = 500

    pertubation_data = sample_perturbation(w0, mu0, sigma0, n_sample, alpha)
    true_data = [mixture_generation(w0, mu0, sigma0) for i in range(n_sample)]
    sns.distplot(pertubation_data, label='Pertub_Data')
    sns.distplot(true_data, label='True_Data')
    plt.title('Pertubation Data(K=2)')
    plt.legend()
    plt.show()

    w0 = np.array([0.25, 0.3, 0.25, 0.2])
    mu0 = np.array([-3.5, 0, 3, 6])
    sigma0 = np.array([0.25, 0.3, 0.25, 0.2])
    pertubation_data = sample_perturbation(w0, mu0, sigma0, n_sample, alpha)
    true_data = [mixture_generation(w0, mu0, sigma0) for i in range(n_sample)]
    sns.distplot(pertubation_data, label='Pertub_Data')
    sns.distplot(true_data, label='True_Data')
    plt.title('Pertubation Data(K=4)')
    plt.legend()
    plt.show()

    w0 = np.array([0.5, 0.5])
    mu0 = np.array([-2, 2])
    sigma0 = np.array([0.7, 0.8])
    alpha = 500
    cluster_maximum = 10
    power = 794
    n_sample = 1000

    pertubation_data = sample_perturbation(w0, mu0, sigma0, n_sample, alpha)
    true_data = [mixture_generation(w0, mu0, sigma0) for i in range(n_sample)]
    sns.distplot(pertubation_data, label='Pertub_Data')
    sns.distplot(true_data, label='True_Data')
    plt.title('Pertubation Data(K=2)')
    plt.legend()
    plt.show()

    mixture_dat = {'N': n_sample,
                   'y': pertubation_data.reshape([n_sample, 1]),
                   'D': 1,
                   'K': cluster_maximum,
                   'alpha': 0.05 / cluster_maximum * np.ones([cluster_maximum, ]),
                   'power': power

                   }
    mixture_model_correct = """
    data {
     int D; //number of dimensions
     int K; //number of gaussians
     int N; //number of data
     vector[D] y[N]; //data
     vector<lower=0>[K] alpha; // components weight prior
     real power;// power posterior
    }
    
    parameters {
     simplex[K] theta; //mixing proportions
     ordered[D] mu[K]; //mixture component means
     vector<lower=0>[K] sigma; //mixture component sigma
    }
    
    model {
     
    
    real ps[K];
    
     theta ~ dirichlet(alpha);
     
     for(k in 1:K){
     mu[k] ~ normal(0,25);
     sigma[k] ~inv_gamma(1, 1);
     
     }
     
    
     for (n in 1:N){
     for (k in 1:K){
     ps[k] =log(theta[k])+normal_lpdf(y[n] | mu[k], sigma[k]); //increment log probability of the gaussian
     }
     target += power/(power+N)*log_sum_exp(ps);
     }
    
    }
    
    generated quantities {                                                                               
      vector[N] log_p_y_tilde; 
      real ps[K];
      for (n in 1:N){
      for (k in 1:K){
     ps[k] =log(theta[k])+normal_lpdf(y[n] | mu[k], sigma[k]); //increment log probability of the gaussian
     }
     log_p_y_tilde[n] = log_sum_exp(ps);
    }
    }                                                                                                    
    
    """

    sm_exact = pystan.StanModel(model_code=mixture_model_correct)

    fit_exact = sm_exact.sampling(data=mixture_dat, iter=100, chains=4)

    fit_exact.plot()

    fit = fit_exact
    pred_y = []
    for j in range(200):

        k_temp = np.random.multinomial(1, fit.extract('theta')['theta'][j, :])
        mu_temp = fit.extract('mu')['mu'][j, :]
        sigma_temp = fit.extract('sigma')['sigma'][j, :]

        pos_mu = np.dot(mu_temp.T, k_temp)
        pos_sigma = np.dot(sigma_temp.T, k_temp)
        pred_y.append(np.random.normal(pos_mu, pos_sigma))

    sns.distplot(pred_y)
    sns.distplot(true_data)

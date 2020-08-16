#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author：Leon Wang
*   Date：Thu Jun 11 16:24:04 2020
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import pystan
import pickle
from rob_bayes_e3 import *
import os
import numpy as np
import sys


data_config = dict(
    alpha=500,
    n_sample=5000,
    w0=[0.7, 0.3],
    mu0=[-2, 2],
    sigma0=[0.7, 0.8],
)


model_setup = dict(
    power=int(10**float(sys.argv[1])),
    gem_alpha=0.5,
)


mcmc_setup = dict(
    cluster_maximum=7,
    power=int(10**float(sys.argv[1])),
    iterations=4000,
    n_chains=4,
)


config_dict = dict(data_config=data_config,
                   mcmc_setup=mcmc_setup,
                   model_setup=model_setup,
                   )

mixture_model_standard = """
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 vector[D] y[N]; //data
 real alpha; // components weight prior
 real power;// power posterior
}

parameters {
 real <lower=0,upper=1> v[K]; // betas
 vector[D] mu[K]; //mixture component means
 vector<lower=0>[K] sigma2; //mixture component sigma
}

transformed parameters{
  simplex[K] theta; //mixing proportions
  theta[1] = v[1];
  // stick-break process 
  for(j in 2:(K-1)){
      theta[j]= v[j]*(1-v[j-1])*theta[j-1]/v[j-1]; 
  }
  theta[K]=1-sum(theta[1:(K-1)]); // to make a simplex.
}

model {
real ps[K]; 

 for(k in 1:K){
 mu[k] ~ normal(0,5);
 sigma2[k] ~ inv_gamma(1, 1);
 v[k] ~ beta(1, alpha);   
 } 
 for (n in 1:N){
 for (k in 1:K){
 ps[k] =log(theta[k])+normal_lpdf(y[n] | mu[k], sqrt(sigma2[k])); //increment log probability of the gaussian
 }
 target += log_sum_exp(ps);
 }
}
"""


mixture_model_correct = """
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 vector[D] y[N]; //data
 real alpha; // components weight prior
 real power;// power posterior
}

parameters {
 real <lower=0,upper=1> v[K]; // betas
 vector[D] mu[K]; //mixture component means
 vector<lower=0>[K] sigma2; //mixture component sigma
}

transformed parameters{
  simplex[K] theta; //mixing proportions
  theta[1] = v[1];
  // stick-break process 
  for(j in 2:(K-1)){
      theta[j]= v[j]*(1-v[j-1])*theta[j-1]/v[j-1]; 
  }
  theta[K]=1-sum(theta[1:(K-1)]); // to make a simplex.
}

model {
real ps[K]; 

 for(k in 1:K){
 mu[k] ~ normal(0,5);
 sigma2[k] ~ inv_gamma(1, 1);
 v[k] ~ beta(1, alpha);   
 }
 
 for (n in 1:N){
 for (k in 1:K){
 ps[k] =log(theta[k])+normal_lpdf(y[n] | mu[k], sqrt(sigma2[k])); //increment log probability of the gaussian
 }
 target += power/(power+N)*log_sum_exp(ps);
 }
}
"""


mixture_model_approx = """
data {
 int D; //number of dimensions
 int K; //number of gaussians
 int N; //number of data
 vector[D] y[N]; //data
 real alpha; // components weight prior
 real power;// power posterior
}

parameters {
 real <lower=0,upper=1> v[K]; // betas
 vector[D] mu[K]; //mixture component means
 vector<lower=0>[K] sigma2; //mixture component sigma
}

transformed parameters{
  simplex[K] theta; //mixing proportions
  theta[1] = v[1];
  // stick-break process 
  for(j in 2:(K-1)){
      theta[j]= v[j]*(1-v[j-1])*theta[j-1]/v[j-1]; 
  }
  theta[K]=1-sum(theta[1:(K-1)]); // to make a simplex.
}

model {
real ps[K]; 

 for(k in 1:K){
 mu[k] ~ normal(0,5);
 sigma2[k] ~ inv_gamma(1, 1);
 v[k] ~ beta(1, alpha);   
 }
 for (n in 1:N){
 for (k in 1:K){
 ps[k] =log(theta[k])+power/(power+N)*normal_lpdf(y[n] | mu[k], sqrt(sigma2[k])); //increment log probability of the gaussian
 }
 target += log_sum_exp(ps);
 }
}
"""

# Adding initialization
mu_init = np.zeros(mcmc_setup['cluster_maximum'])
mu_init[0] = 2
mu_init[1] = -2
sigma_init = np.ones(mcmc_setup['cluster_maximum'])
init_list=[]
for i_ in range(mcmc_setup['n_chains']):
    init_v_list = np.random.beta(11,5, mcmc_setup['cluster_maximum'])
#    init_v_list [0] = 0.97
#    init_v_list [1] = 0.3
    temp_dict={
        'mu':np.random.normal(mu_init, sigma_init, [mcmc_setup['cluster_maximum'], 1]),
 #       'v': init_v_list,
    }
    init_list.append(temp_dict)
    
    
    
    

def model_run(model_names, path, config_dict):

    data_config = config_dict['data_config']
    mcmc_setup = config_dict['mcmc_setup']
    model_setup = config_dict['model_setup']

    pertubation_data = sample_perturbation(data_config)
    true_data = [mixture_generation(data_config)
                 for i in range(data_config['n_sample'])]

    with open(path + '/data' + str(mcmc_setup['power']) + '.pickle', "wb") as f:
        pickle.dump(
            {'true': true_data, 'pertubation': pertubation_data}, f, protocol=-1)

    mixture_dat = {'N': data_config['n_sample'],
                   'y': pertubation_data.reshape([data_config['n_sample'], 1]),
                   'D': 1,
                   'K': mcmc_setup['cluster_maximum'],
                   'alpha': model_setup['gem_alpha'],
                   'power': model_setup['power']
                   }
    if 'fit_standard' in model_names:
        sm_standard = pystan.StanModel(model_code=mixture_model_standard)
        fit_standard = sm_standard.sampling(
            data=mixture_dat, iter=mcmc_setup['iterations'], chains=mcmc_setup['n_chains'], init=init_list)
        with open(path + '/fit_standard' + str(mcmc_setup['power']) + '.pickle', "wb") as f:
            pickle.dump({'model': sm_standard,
                         'fit': fit_standard}, f, protocol=-1)

        print('Finished Standard Fitting')
    if 'fit_correct' in model_names:
        sm_correct = pystan.StanModel(model_code=mixture_model_correct)
        fit_correct = sm_correct.sampling(
            data=mixture_dat, iter=mcmc_setup['iterations'], chains=mcmc_setup['n_chains'], init=init_list)
        with open(path + '/fit_correct' + str(mcmc_setup['power']) + '.pickle', "wb") as f:
            pickle.dump({'model': sm_correct,
                         'fit': fit_correct}, f, protocol=-1)

        print('Finished Correct Fitting')
    if 'fit_approx' in model_names:
        sm_approx = pystan.StanModel(model_code=mixture_model_approx)
        fit_approx = sm_approx.sampling(
            data=mixture_dat, iter=mcmc_setup['iterations'], chains=mcmc_setup['n_chains'], init=init_list)
        with open(path + '/fit_approx' + str(mcmc_setup['power']) + '.pickle', "wb") as f:
            pickle.dump({'model': sm_approx,
                         'fit': fit_approx}, f, protocol=-1)

        print('Finished Approxmiate Fitting')


def load_fitted_object(pickle_names, path):
    """
        Function to load pickle object,
        params: names, List of pickle names
        params: directory to which the those pickles are stored

    """
    List = []
    for sub_name in pickle_names:
        print(path + sub_name + ".pickle")
        with open(path + sub_name + ".pickle", "rb") as f:
            fitted_object = pickle.load(f)

        List.append(fitted_object)
    return List


if __name__ == "__main__":

    print(config_dict)
    path = sys.argv[2]
    print(path)
    pickle_names = ['fit_correct', 'fit_approx']
    model_run(pickle_names, path, config_dict)

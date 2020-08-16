# Gaussian_Mixture_Pystan
Research on robust bayeisan with perturbed gaussian mixture model


Mainly to replicate the experiment in section 5 of [Jeffrey Miller 2019](https://www.tandfonline.com/doi/abs/10.1080/01621459.2018.1469995#:~:text=The%20standard%20approach%20to%20Bayesian,outcome%20of%20a%20Bayesian%20procedure.)


Tried to investigate the MCMC efficiency between 

1. Exact Robust Sampling

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?P^{\xi}(X|\theta)=[\sum_{z=1}^q&space;P(X|Z_i...)\pi(Z_i)]^{\xi}" title="P^{\xi}(X|\theta)=[\sum_{z=1}^q P(X|Z_i...)\pi(Z_i)]^{\xi})


<img src="https://latex.codecogs.com/svg.latex?P^{\xi}(X|\theta)=[\sum_{z=1}^q&space;P(X|Z_i...)\pi(Z_i)]^{\xi}" title="P^{\xi}(X|\theta)=[\sum_{z=1}^q P(X|Z_i...)\pi(Z_i)]^{\xi}" />

2. Approximate Robust Sampling

<img src="https://latex.codecogs.com/svg.latex?P^{\xi}(X|\theta)&space;\approx&space;\sum_{z=1}^q&space;P^{\xi}(X|Z_i...)\pi(Z_i)" title="P^{\xi}(X|\theta) \approx \sum_{z=1}^q P^{\xi}(X|Z_i...)\pi(Z_i)" />


The conclusion is that the approximate sampling tends to favor more parsimony model compared to the exact sampling one. Jupyter Notebook can be found in [Exact Sampling](Calibration_Exact4000_Chg_Init.html) and [Approximate Sampling](https://github.com/LeonCaesa/Gaussian_Mixture_Pystan/blob/master/Calibration_Approxt4000_Chg_Init.html)


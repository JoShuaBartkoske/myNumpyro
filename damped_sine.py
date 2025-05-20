'''
This is a test for numpyro created by Joshua Thomas Bartkoske for testing a fit to a damped sinusoid.
It will start as a sinusoid and then become a damped sinusoid.
Date of Creation: May 20, 2025

We're going to try and do this without ChatGPT and then see what ChatGPT would do
'''

# imports from the numpyro bnn.py example
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# set the backend
backend= "Qt5Agg"
matplotlib.use(backend=backend)  

# define the sinusoid and the damped sinusoid
def sinusoid_model(x, omega, amplitude, phase):
    # x --> the x-values for creating the sinusoid
    # omega --> the angular frequency: omega = 2*pi*f
    # amplitude --> the amplitude of the sinusoid
    # phase --> the phase of the sinusoid
    # OUTPUT --> a custom sinusoidal curve over the range of x
    return jnp.sin(omega*x + phase)*amplitude

def damped_sinusoid(x, omega, amplitude, phase=0, decay=0):
    # x --> the x-values for creating the sinusoid
    # omega --> the angular frequency: omega = 2*pi*f
    # amplitude --> the amplitude of the sinusoid
    # phase --> the phase of the sinusoid
    # decay --> the exponent in the exponential for the decay
    # OUTPUT --> a custom decaying sinusoid over the range of x
    sinusoid = sinusoid_model(x, omega, amplitude, phase)
    return sinusoid*jnp.exp(-decay*x)

# create synthetic data
omega = 2*np.pi
amplitude = 2
phase = np.pi/2
decay = 0.5
x = np.linspace(0,10,1000)
y_true = damped_sinusoid(x=x, omega=omega, amplitude=amplitude, phase=phase, decay=decay)
noise = np.random.normal(loc=0,scale=0.05, size=len(y_true))
y = y_true + noise

fig, ax = plt.subplots(figsize=(8,6))
plt.grid(visible=True, which='both', axis='both')
ax.plot(x,y, label="damped sine")
if backend=="Qt5Agg":
    plt.show()
elif backend=="Agg":
    plt.savefig("damped_sinusoid.png", dpi=300)

# now for creating the numpyro model
def model(data, xvals):
    om = numpyro.sample("omega", dist.Uniform(0.0, 10.0))  
    amp = numpyro.sample("amplitude", dist.Uniform(0.0,10.0))
    ph = numpyro.sample("phase", dist.Uniform(0.0, 2*jnp.pi))
    d = numpyro.sample("decay", dist.Uniform(0.0, 2.0))

    model_Dampsine = damped_sinusoid(
        x=xvals,
        omega=om,
        amplitude=amp,
        phase=ph,
        decay=d
    )
    
    # Observation model
    sigma = numpyro.sample("obs_noise", dist.HalfNormal(0.05))
    numpyro.sample("obs", dist.Normal(model_Dampsine, sigma), obs=data)
    # numpyro.sample("obs", dist.Normal(model_Dampsine, 1e-6), obs=data)


nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, y, xvals=x)

posterior_samples = mcmc.get_samples()

import arviz as az

azdata = az.from_numpyro(mcmc)
az.plot_trace(azdata, compact=True, figsize=(15, 25))
plt.show()

# Step 1: Extract median and std from posterior samples
best_fit_params = {}
param_errors = {}
for param in ['omega', 'amplitude', 'phase', 'decay']:
    samples = posterior_samples[param]
    best_fit_params[param] = np.median(samples)
    param_errors[param] = np.std(samples)

print("Best-fit parameters (median ± std):")
for param in best_fit_params:
    median = best_fit_params[param]
    std = param_errors[param]
    print(f"{param}: {median:.5f} ± {std:.5f}")

# Step 2: compute model for best-fit parameters
best_fit_sinusoid = damped_sinusoid(x, best_fit_params['omega'], best_fit_params['amplitude'],
                                    best_fit_params['phase'], best_fit_params['decay'])

# Step 3: Plot observed vs best-fit model
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label='True', color='black', alpha=0.6)
plt.plot(x, y, label='Noisy', color='blue', alpha=0.4)
plt.plot(x, best_fit_sinusoid, label='Best-fit Model', color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("True vs Best-fit Model Damped Sinusoid")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

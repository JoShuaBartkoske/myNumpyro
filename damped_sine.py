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
y = damped_sinusoid(x=x, omega=omega, amplitude=amplitude, phase=phase, decay=decay)

fig, ax = plt.subplots(figsize=(8,6))
plt.grid(visible=True, which='both', axis='both')
ax.plot(x,y, label="damped sine")
if backend=="Qt5Agg":
    plt.show()
elif backend=="Agg":
    plt.savefig("damped_sinusoid.png", dpi=300)


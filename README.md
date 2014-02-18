Look Ahead Hamiltonian Monte Carlo
==================================

Implements Look Ahead Hamiltonian Monte Carlo (LAHMC) and standard Hamiltonian Monte Carlo (HMC) in both Python and MATLAB.

LAHMC is described in the paper:<br>
> Sohl-Dickstein, Jascha and Mudigonda, Mayur and DeWeese, Michael R.<br>
> Hamiltonian Monte Carlo Without Detailed Balance.<br>
> International Conference on Machine Learning. 2014<br>
> http://jmlr.org/proceedings/papers/v32/sohl-dickstein14.pdf

## Example Python Code

The following code draws samples from an isotropic Gaussian distribution using LAHMC.

```python
from LAHMC import LAHMC
import numpy as np

# Define the energy function and gradient
def E(X, sigma=1.):
    """ Energy function for isotropic Gaussian """
    return np.sum(X**2, axis=0).reshape((1,-1))/2./sigma**2
def dEdX(X, sigma=1.):
    """ Energy function gradient for isotropic Gaussian """
    return X/sigma**2

# Initialize the sample locations -- 2 dimensions, 100 particles
Xinit = np.random.randn(2,100)

# initialize the sampler.
sampler = LAHMC(Xinit, E, dEdX, epsilon=0.1, beta=0.1, kwargs={'sigma':0.1})
# perform 10 sampling steps for all 100 particles
X = sampler.sample(num_steps = 10)
# perform another 10 sampling steps
X = sampler.sample(num_steps = 10)
```

More detailed documentation, and additional options, can be found in **python/LAHMC.py**

## Example MATLAB Code

The following code draws samples from an isotropic Gaussian distribution using LAHMC.

```MATLAB
% opts holds all parameters which will be passed to the sampler
opts = [];
opts.epsilon = 0.1;
opts.beta = 0.1;
% number of sampling steps
opts.T = 10;
% energy function and gradient
opts.E = @E_gauss;
opts.dEdX = @dEdX_gauss;

% state will hold the particle positions and velocities between
% sampler calls, as well as counters for the number of transitions
% and function evaluations
state = []

% Initialize sample locations -- 2 dimensions, 100 particles
opts.Xinit = randn(2,100);
% Gaussian coupling matrix expected by E_gauss and dEdX_gauss
J = eye(2)*100;

% perform 10 sampling steps for all 100 particles
[X, state] = LAHMC(opts, state, J);
% perform another 10 sampling steps
[X, state] = LAHMC(opts, state, J);
```

More detailed documentation, and additional options, can be found in **matlab/LAHMC.m**.

## Reproduce Figure from the Paper

Code reproducing Figure 2 and Table 1 of the paper, and demonstrating usage of the sampler, can be found in **python/generate_figure_2.py** and **matlab/generate_figure_2.m**.  The exact plots appearing in the paper were generated using the MATLAB version of the code.

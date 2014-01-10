"""
Implements Look Ahead Hamiltonian Monte Carlo (LAHMC) and standard 
Hamiltonian Monte Carlo (HMC).  See the associated paper:
   Sohl-Dickstein, Jascha and Mudigonda, Mayur and DeWeese, Michael R.
   Hamiltonian Monte Carlo Without Detailed Balance.
   International Conference on Machine Learning. 2014

See generate_figure_2.py for an example usage.

Author: Jascha Sohl-Dickstein, Mayur Mudigonda (2014)
Web: http://redwood.berkeley.edu/mayur
Web: http://redwood.berkeley.edu/jascha
This software is made available under the Creative Commons
Attribution-Noncommercial License.
(http://creativecommons.org/licenses/by-nc/3.0/)
"""

import numpy as np
from collections import defaultdict

class HMC_state(object):
	""" Holds all the state variables for HMC particles."""

	def __init__(self, X, parent, V=None, EX=None, EV=None, dEdX=None):
		"""
		Initialize HMC particle states.  Called by LAHMC class.
		Not user facing.
		"""
		self.parent = parent
		self.X = X
		self.V = V
		nbatch = X.shape[1]
		self.active_idx = np.arange(nbatch)
		if V is None:
			N = self.X.shape[0]
			self.V = np.random.randn(N, nbatch)
		self.EX = EX
		if EX is None:
			self.EX = np.zeros((1,nbatch))
			self.update_EX()
		self.EV = EV
		if EV is None:
			self.EV = np.zeros((1,nbatch))
			self.update_EV()
		self.dEdX = dEdX
		if dEdX is None:
			self.dEdX = np.zeros(X.shape)
			self.update_dEdX()

	def update_EX(self):
		self.EX[:,self.active_idx] = self.parent.E(self.X[:,self.active_idx]).reshape((1,-1))
	def update_EV(self):
		self.EV[:,self.active_idx] = np.sum(self.V[:,self.active_idx]**2, axis=0).reshape((1,-1))/2.
	def update_dEdX(self):
		self.dEdX[:,self.active_idx] = self.parent.dEdX(self.X[:,self.active_idx])

	def copy(self):
		Z = HMC_state(self.X.copy(), self.parent, V=self.V.copy(), EX=self.EX.copy(), EV=self.EV.copy(), dEdX=self.dEdX.copy())
		Z.active_idx = self.active_idx.copy()
		return Z

	def update(self, idx, Z):
		""" replace batch elements idx with state from Z """
		if len(idx)==0:
			return
		self.X[:,idx] = Z.X[:,idx]
		self.V[:,idx] = Z.V[:,idx]
		self.EX[:,idx] = Z.EX[:,idx]
		self.EV[:,idx] = Z.EV[:,idx]
		self.dEdX[:,idx] = Z.dEdX[:,idx]


class LAHMC(object):
	"""
	Implements Look Ahead Hamiltonian Monte Carlo (LAHMC) and standard 
	Hamiltonian Monte Carlo (HMC).  See the associated paper:
	   Sohl-Dickstein, Jascha and Mudigonda, Mayur and DeWeese, Michael R.
	   Hamiltonian Monte Carlo Without Detailed Balance.
	   International Conference on Machine Learning. 2014

	See generate_figure_2.py for an example usage.
	"""

	def __init__(self, Xinit, E, dEdX, epsilon=0.1, alpha=0.2, beta=None, num_leapfrog_steps=10, num_look_ahead_steps=4, display=1):
		"""
		Implements Look Ahead Hamiltonian Monte Carlo (LAHMC) and standard 
		Hamiltonian Monte Carlo (HMC).  See the associated paper:
		   Sohl-Dickstein, Jascha and Mudigonda, Mayur and DeWeese, Michael R.
		   Hamiltonian Monte Carlo Without Detailed Balance.
		   International Conference on Machine Learning. 2014

		See generate_figure_2.py for an example usage.

		Required arguments:
		Xinit - The initial particle positions for sampling.  Should be an
		  array of size [number dimensions]x[batch size].
		E - Function that returns the energy.  E(X) should return
		  a 1x[batch size] array, with each entry corresponding to the energy
		  of the corresponding column of X.
		dEdX - Function that returns the gradient.  dEdX(X, varargin{:}) should
		  return an array of the same size as X,
		  [number dimensions]x[batch size].

		Keyword arguments:
		epsilon - Step length. (default 0.1)
		alpha - One way to specify momentum corruption rate.  See Eq. 22.
		  (default 0.2)
		beta - Alternate way to specify momentum corruption rate.  See Eq. 15.
		  (default value derived from alpha)
		num_leapfrog_steps - The number of Leapfrog steps per simulation run.
		  (default 10)
		num_look_ahead_steps - The maximum number of times to apply the L
		  operator. (default 4)
		display - How verbose to be with console logging.  Set to 0 to disable
		  logging. (default 1)
		"""
		self.M = num_leapfrog_steps
		self.K = num_look_ahead_steps
		self.N = Xinit.shape[0]
		self.nbatch = Xinit.shape[1]
		self.display = display
		self.E_external = E
		self.dEdX_external = dEdX

		self.state = HMC_state(Xinit.copy(), self)

		self.epsilon = epsilon

		self.beta = beta
		if beta is None:
			self.beta = alpha**(1./(self.epsilon*self.M))

		# keep track of how many of each kind of transition happens
		self.counter = defaultdict(int)
		self.counter_steps = 0

	def E(self, X):
		"""compute energy function at X"""
		# TODO use parameter flattening and unflattening
		# code from SFO
		E = self.E_external(X).reshape((1,-1))
		return E

	def dEdX(self, X):
		"""compute energy function gradient at X"""
		# TODO use parameter flattening and unflattening
		# code from SFO
		dEdX = self.dEdX_external(X)
		return dEdX

	def H(self, state):
		""" The Hamiltonian for the full system """
		H = state.EX + state.EV
		return H

	def leapfrog(self, state):
		""" A single leapfrog step for X and V """
		idx = state.active_idx
		state.V[:,idx] += -self.epsilon/2. * state.dEdX[:,idx]
		state.X[:,idx] += self.epsilon * state.V[:,idx]
		state.update_dEdX()
		state.V[:,idx] += -self.epsilon/2. * state.dEdX[:,idx]
		return state

	def L(self, state):
		""" Run the leapfrog operator for M leapfrog steps """
		for _ in range(self.M):
			state = self.leapfrog(state)
		state.update_EV()
		state.update_EX()
		return state

	def leap_prob(self, Z1, Z2):
		"""
		Metropolis-Hastings Probability of transitioning from state Z1 to
		state Z2.
		"""
		EZ1 = self.H(Z1)
		EZ2 = self.H(Z2)
		Ediff = EZ1 - EZ2
		p_acc = np.ones((1, Ediff.shape[1]))
		p_acc[Ediff<0] = np.exp(Ediff[Ediff<0])
		return p_acc

	def leap_prob_recurse(self, Z_chain, C, active_idx):
		"""
		Recursively compute to cumulative probability of transitioning from
		the beginning of the chain Z_chain to the end of the chain Z_chain.
		"""

		if np.isfinite(C[0,-1,0]):
			# we've already visited this leaf
			cumu = C[0,-1,:].reshape((1,-1))
			return cumu, C

		if len(Z_chain) == 2:
			# the two states are one apart
			p_acc = self.leap_prob(Z_chain[0], Z_chain[1])
			p_acc = p_acc[:,active_idx]
			#print C.shape, C[0,-1,:].shape, p_acc.shape, p_acc.ravel().shape
			C[0,-1,:] = p_acc.ravel()
			return p_acc, C

		cum_forward, Cl = self.leap_prob_recurse(Z_chain[:-1], C[:-1,:-1,:], active_idx)
		C[:-1,:-1,:] = Cl
		cum_reverse, Cl = self.leap_prob_recurse(Z_chain[:0:-1], C[:0:-1,:0:-1,:], active_idx)
		C[:0:-1,:0:-1,:] = Cl

		H0 = self.H(Z_chain[0])
		H1 = self.H(Z_chain[-1])
		Ediff = H0 - H1
		Ediff = Ediff[:,active_idx]
		start_state_ratio = np.exp(Ediff)
		prob = np.min(np.vstack((1. - cum_forward, start_state_ratio*(1. - cum_reverse))), axis=0).reshape((1,-1))
		cumu = cum_forward + prob
		C[0,-1,:] = cumu.ravel()
		return cumu, C

	def sampling_iteration(self):
		""" Perform a single sampling step. """

		# first do the HMC part of the step
		Z_chain = [self.state.copy(),]
		# use the same random number for comparison for the entire chain
		rand_comparison = np.random.rand(1, self.nbatch).ravel()
		# the current cumulative probability of acceptance
		p_cum = np.zeros((1, self.nbatch))
		# the cumulative probability matrix, so we only need to visit each leaf once when recursing
		C = np.ones((self.K+1, self.K+1, self.nbatch))*np.nan
		# the current set of indices for samples that have not yet been accepted for a transition
		active_idx = np.arange(self.nbatch, dtype=int)
		for kk in range(self.K):
			Z_chain.append(self.L(Z_chain[-1].copy()))
			# recursively calculate the cumulative probability of doing this many leaps
			p_cum, Cl = self.leap_prob_recurse(Z_chain, C[:kk+2, :kk+2, active_idx], active_idx)
			C[:kk+2, :kk+2, active_idx] = Cl
			# find all the samples that did this number of leaps, and update self.state with them
			accepted_idx = active_idx[p_cum.ravel() >= rand_comparison[active_idx]]
			self.counter['L%d'%(kk+1)] += len(accepted_idx)
			self.state.update(accepted_idx, Z_chain[-1])
			# update the set of active indices, so we don't do simulate trajectories for samples that are already assigned to a state
			active_idx = active_idx[p_cum.ravel() < rand_comparison[active_idx]]
			if len(active_idx) == 0:
				break
			Z_chain[-1].active_idx = active_idx
		# flip the momenutm for any samples that were unable to place elsewhere
		self.state.V[:,active_idx] = -self.state.V[:,active_idx]
		self.counter['F'] += len(active_idx)

		if self.display > 1:
			print "Transition counts ",
			for k in sorted(self.counter.keys()):
				print "%s:%d"%(k, self.counter[k]),

		# corrupt the momentum
		self.state.V = self.state.V*np.sqrt(1.-self.beta) + np.random.randn(self.N,self.nbatch)*np.sqrt(self.beta)
		self.state.update_EV()

	def sample(self, num_steps=100):
		"""
		Sample from the target distribution, for num_steps sampling steps.
		This is the function to call from external code.
		"""
		for si in range(num_steps):
			if self.display > 1:
				print "sampling step %d / %d,"%(si+1, num_steps),
			self.sampling_iteration()
			self.counter_steps += 1
			if self.display > 1:
				print 

		if self.display > 0:
			tot = 0
			for k in sorted(self.counter.keys()):
				tot += self.counter[k]
			print "Step %d, Transition fractions "%(self.counter_steps),
			for k in sorted(self.counter.keys()):
				print "%s:%g"%(k, self.counter[k]/float(tot)),
			print

		return self.state.X.copy()

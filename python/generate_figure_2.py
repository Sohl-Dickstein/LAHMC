from LAHMC import LAHMC
import matplotlib.pyplot as plt
import distributions
import numpy as np
import datetime

class counting_wrapper(object):
	def __init__(self, E, dEdX):
		"""
		A wrapper class for the energy function and gradient function, keeps
		track of how many times each one is called by the sampler, for later
		analysis.
		"""
		self.E_external = E
		self.dEdX_external = dEdX
		self.E_count = 0
		self.dEdX_count = 0
	def E(self, *args, **kwargs):
		self.E_count += args[0].shape[1]
		return self.E_external(*args, **kwargs)
	def dEdX(self, *args, **kwargs):
		self.dEdX_count += args[0].shape[1]
		return self.dEdX_external(*args, **kwargs)

def calc_corr(hist_single):
	"""
	Calculate autocorrelation given history.  Assumes 0 mean.
	"""

	T = len(hist_single)
	N = hist_single[0]['X'].shape[0]
	nbatch = hist_single[0]['X'].shape[1]

	X = np.zeros((N,nbatch,T))
	for tt in range(T):
		X[:,:,tt] = hist_single[tt]['X']

	c = np.zeros((T-1,))
	c[0] = np.mean(X**2)
	for t_gap in range(1, T-1):
		c[t_gap] = np.mean(X[:,:,:-t_gap]*X[:,:,t_gap:])

	return c/c[0]

def plot_all_distributions(history):
	"""
	Plot autocorrelation for all samplers and target distributions.
	"""

	for dist_name in history.keys():
		plt.figure()
		plt.suptitle(dist_name)
		for samp_name in history[dist_name].keys():
			hist_single = history[dist_name][samp_name]
			nsteps = len(hist_single)
			nbatch = hist_single[-1]['X'].shape[1]
			#nfunc = (hist_single[-1]['dEdX_count'] + hist_single[-1]['E_count'])/float(nbatch)
			nfunc = hist_single[-1]['dEdX_count']/float(nbatch)
			corr = calc_corr(hist_single)
			t_diff = np.linspace(0, nfunc, corr.shape[0])
			plt.plot(t_diff, corr, label=samp_name)
	plt.legend()
	plt.xlabel('Gradient Evaluations')
	plt.ylabel('Autocorrelation')
	plt.draw()
	plt.show()


def load_results(filename):
	"""
	Load saved sampling history.
	"""
	data = np.load(filename)
	history = data['history'][()]
	return history

def save_results(history, filename):
	"""
	Save the sampling history for all samplers and distributions to a .npz
	file.
	"""
	np.savez(filename, history=history)

def run_all(base_filename, num_steps=200, nbatch=100):
	"""
	Run all the samplers for all the target distributions, and save the
	resulting history.
	"""
	#np.random.seed(0) # make experiments repeatable

	filename = "%s_%s.npz"%(base_filename, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	history = dict()

	#np.random.seed(0) # make experiments repeatable
	distribution_list = [
		distributions.RoughWell(nbatch=nbatch),
		distributions.Gaussian(ndims=2, nbatch=nbatch, log_conditioning=6), 
		distributions.Gaussian(ndims=100, nbatch=nbatch, log_conditioning=6),
						]
	sampler_list = ('LAHMC', 'HMC',)

	for distribution in distribution_list:
		dist_name = distribution.description
		history[dist_name] = dict()
		for sampler_name in sampler_list:
			cw = counting_wrapper(distribution.E, distribution.dEdX)
			if sampler_name == 'LAHMC':
				sampler = LAHMC(distribution.Xinit, cw.E, cw.dEdX, epsilon=1., beta=0.1)
			elif sampler_name == 'HMC':
				# 1 look-ahead step corresponds to standard HMC
				sampler = LAHMC(distribution.Xinit, cw.E, cw.dEdX, epsilon=1., beta=0.1, num_look_ahead_steps=1)
			else:
				raise Exception("unknown sampler %s"%(sampler_name))
			# np.random.seed(0) # make experiments repeatable
			print "\n\nSampling from %s using %s"%(dist_name, sampler_name)
			history[dist_name][sampler_name] = []
			for ii in range(num_steps):
				X = sampler.sample(num_steps = 10)
				history[dist_name][sampler_name].append({'X':X.copy(), 'E_count':cw.E_count, 'dEdX_count':cw.dEdX_count})
			# save the current state of the history
			save_results(history, filename)

	return filename

if __name__ == '__main__':
	filename = run_all('LAHMC_compare')
	history = load_results(filename)
	plot_all_distributions(history)

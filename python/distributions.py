import numpy as np

class Gaussian(object):
	def __init__(self, ndims=2, nbatch=100, log_conditioning=6):
		"""
		Energy function, gradient, and hyperparameters for the "ill
		conditioned Gaussian" example from the LAHMC paper.
		"""

		conditioning = 10**np.linspace(-log_conditioning, 0, ndims)
		self.J = np.diag(conditioning)
		self.Xinit = (1./np.sqrt(conditioning).reshape((-1,1))) * np.random.randn(ndims,nbatch)
		self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, 10**log_conditioning)
	def E(self, X):
		return np.sum(X*np.dot(self.J,X), axis=0).reshape((1,-1))/2.
	def dEdX(self, X):
		return np.dot(self.J,X)/2. + np.dot(self.J.T,X)/2.

class RoughWell(object):
	def __init__(self, ndims=2, nbatch=100, scale1=100, scale2=4):
		"""
		Energy function, gradient, and hyperparameters for the "rough well"
		example from the LAHMC paper.
		"""

		self.scale1 = scale1
		self.scale2 = scale2
		self.Xinit = scale1 * np.random.randn(ndims,nbatch)
		self.description = '%dD Rough Well'%(ndims)
	def E(self, X):
		cosX = np.cos(X*2*np.pi/self.scale2)
		E = np.sum((X**2) / (2*self.scale1**2) + cosX, axis=0).reshape((1,-1))
		return E
	def dEdX(self, X):
		sinX = np.sin(X*2*np.pi/self.scale2)
		dEdX = X/self.scale1**2 + -sinX*2*np.pi/self.scale2
		return dEdX

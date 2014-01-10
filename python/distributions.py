import numpy as np

class Gaussian(object):
    def __init__(self, ndims=2, log_conditioning=6, nbatch=1):
        conditioning = 10**np.linspace(-log_conditioning, 0, ndims)
        self.J = np.diag(conditioning)
        self.Xinit = (1./np.sqrt(conditioning).reshape((-1,1))) * np.random.randn(ndims,nbatch)
        self.description = '%dD Anisotropic Gaussian, %g conditioning'%(ndims, 10**log_conditioning)
    def E(self, X):
        return np.sum(X*np.dot(self.J,X), axis=0).reshape((1,-1))/2.
    def dEdX(self, X):
        return np.dot(self.J,X)/2. + np.dot(self.J.T,X)/2.

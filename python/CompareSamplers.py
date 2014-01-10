from qNHMC import qNHMC
import matplotlib.pyplot as plt
import distributions
import numpy as np
import datetime

class counting_wrapper(object):
    def __init__(self, E, dEdX):
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

def calc_cov(hist_single):
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

    return c

def plot_all_distributions(history):
    for dist_name in history.keys():
        plt.figure()
        plt.suptitle(dist_name)
        for samp_name in history[dist_name].keys():
            hist_single = history[dist_name][samp_name]
            nsteps = len(hist_single)
            nbatch = hist_single[-1]['X'].shape[1]
            nfunc = (hist_single[-1]['dEdX_count'] + hist_single[-1]['E_count'])/nbatch
            cov = calc_cov(hist_single)
            t_diff = np.linspace(0, nfunc, cov.shape[0])
            plt.plot(t_diff, cov, label=samp_name)
    plt.draw()


def load_results(filename):
    data = np.load(filename)
    history = data['history'][()]
    return history

def save_results(history, filename):
    """
    Save the function trace for different optimizers for a 
    given model to a .npz file.
    """
    np.savez(filename, history=history)

def run_all(base_filename, num_steps=1000, nbatch=100):
    np.random.seed(0) # make experiments repeatable

    filename = "%s_%s.npz"%(base_filename, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    history = dict()

    distribution_list = [distributions.Gaussian(ndims=2, nbatch=nbatch, log_conditioning=6), ]
    sampler_list = ['LAHMC', 'HMC',]

    for distribution in distribution_list:
        cw = counting_wrapper(distribution.E, distribution.dEdX)
        dist_name = distribution.description
        history[dist_name] = dict()
        for sampler_name in sampler_list:
            if sampler_name == 'LAHMC':
                sampler = sampler_class(distribution.Xinit, cw.E, cw.dEdX)
            elif sampler_name == 'HMC':
            else:
                raise Exception("unknown sampler %s"%(sampler_name))
            np.random.seed(0) # make experiments repeatable
            history[dist_name][sampler_name] = []
            for ii in range(num_steps):
                X = sampler.sample(num_steps = 10)
                history[dist_name][sampler_name].append({'X':X, 'E_count':cw.E_count, 'dEdX_count':cw.dEdX_count})
            # save the current state of the history
            save_results(history, filename)
    return filename

if __name__ == '__main__':
    filename = run_all('qNHMC_compare', 100)
    history = load_results(filename)
    plot_all_distributions(history)

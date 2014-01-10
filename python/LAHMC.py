import numpy as np

class HMC_state(object):
    """ Holds all the state variables for qNHMC. """

    def __init__(self, X, parent, V=None, EX=None, EV=None, dEdX=None):
        self.parent = parent
        self.X = X
        self.V = V
        if V is None:
            N = self.X.shape[0]
            nbatch = self.X.shape[1]
            self.V = np.random.randn(N, nbatch)
        self.EX = EX
        if EX is None:
            self.update_EX()
        self.EV = EV
        if EV is None:
            self.EV = self.update_EV()
        self.dEdX = dEdX
        if dEdX is None:
            self.update_dEdX()
        self.active_idx = False

    def update_EX():
        if self.active_idx:
            self.EX[:,self.active_idx] = parent.E(self.X[:,self.active_idx])
        else:
            self.EX = parent.E(self.X)
    def update_EV():
        if self.active_idx:
            self.EV[:,self.active_idx] = np.sum(self.V[:,self.active_idx]**2, axis=0).reshape((1,-1))/2.
        else:
            self.EV = np.sum(self.V**2, axis=0).reshape((1,-1))/2.
    def update_dEdX():
        if self.active_idx:
            self.dEdX[:,self.active_idx] = parent.dEdX(self.X[:,self.active_idx])
        else:
            self.dEdX = parent.dEdX(self.X)

    def copy(self):
        Z = HMC_state(self.X.copy(), self.parent, V=self.V.copy(), EX=self.EX.copy(), EV=self.EV.copy(), dEdX=self.dEdX.copy())
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
    quasiNewton Hamiltonian Monte Carlo sampler
    """

    def __init__(self, Xinit, E, dEdX, num_leapfrog_steps=10, num_look_ahead_steps=5, epsilon=0.1, alpha=0.2, beta=None, display=10):

        self.M = num_leapfrog_steps
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

    def E(self, X):
        # TODO store recent history, to avoid recalculating recent

        # TODO use parameter flattening and unflattening
        # code from SFO
        E = self.E_external(X).reshape((1,-1))
        return E

    def dEdX(self, X):
        # TODO use parameter flattening and unflattening
        # code from SFO
        dEdX = self.dEdX_external(X)
        return dEdX

    def H(self, state):
        """ The Hamiltonian for the full system """
        if state.active_idx:
            idx = state.active_idx
            H = state.EX[:,idx] + state.EV[:,idx]
        else:
            H = state.EX + state.EV
        return H

    def leapfrog(self, state):
        """ A single leapfrog step for X and V """
        if state.active_idx:
            idx = state.active_idx
            state.V[:,idx] += -self.epsilon/2. * state.dEdX[:,idx]
            state.X[:,idx] += self.epsilon * state.V[:,idx]
            state.update_dEdX()
            state.V[:,idx] += -self.epsilon/2. * state.dEdX[:,idx]
        else:
            state.V += -self.epsilon/2. * state.dEdX
            state.X += self.epsilon * state.V
            state.update_dEdX()
            state.V += -self.epsilon/2. * state.dEdX
        return state

    def L(self, state):
        """ Run the leapfrog operator for M leapfrog steps """
        for _ in range(self.M):
            state = self.leapfrog(state)
        state.update_EV()
        state.update_EX()
        return state

    def F(self, state):
        """ Flip the momentum """
        state.V = -state.V
        return state

    def leap_prob(self, Z1, Z2):
        """
        Metropolis-Hastings Probability of transitioning from state Z1 to
        state Z2.
        """
        EZ1 = self.H(Z1)
        EZ2 = self.H(Z2)
        Ediff = EZ1 - EZ2
        p_acc = np.ones(1, Ediff.shape[1])
        p_acc[Ediff<0] = np.exp(Ediff[Ediff<0])
        return p_acc

    def leap_prob_recurse(self, Z_chain, C):
        if np.isfinite(C[0,-1,0]):
            # we've already visited this leaf
            cumu = C[0,-1,:]
            return cumu, C

        if len(Z_chain) == 2:
            # the two states are one apart
            p_acc = self.leap_prob(Z_chain[0], Z_chain[1])
            C[0,-1,:] = p_acc
            return p_acc, C

        cum_forward, Cl = self.leap_prob_recurse(Z_chain[:-1], C[:-1,:-1,:])
        C[:-1,:-1,:] = Cl
        cum_reverse, Cl = self.leap_prob_recurse(Z_chain[:0:-1], C[:0:-1,:0:-1,:])
        C[:0:-1,:0:-1,:] = Cl

        H0 = self.H(Z_chain[0])
        H1 = self.H(Z_chain[1])
        start_state_ratio = np.exp(H1 - H2)
        prob = np.min(np.vstack((1. - cum_forward, start_state_ratio*(1. - cum_reverse))), 0)
        cumu = cum_forward + prob
        C[0,-1,:] = cumu
        return cumu, C

    def sampling_iteration(self):
        """ Perform a single sampling step. """

        # first do the HMC part of the step
        Z_chain = (self.state.copy(),)
        # use the same random number for comparison for the entire chain
        rand_comparison = np.random.rand(1, self.nbatch)
        # the current cumulative probability of acceptance
        p_cum = np.zeros(1, self.nbatch)
        # the cumulative probability matrix, so we only need to visit each leaf once when recursing
        C = np.ones((self.K+1, self.K+1, nbatch))*np.nan
        # the current set of indices for samples that have not yet been accepted for a transition
        active_idx = np.arange(self.nbatch, dtype=int)
        for kk in range(self.K):
            Z_chain.append(self.L(Z_chain[-1].copy()))
            # recursively calculate the cumulative probability of doing this many leaps
            p_cum, Cl = self.leap_prob_recurse(Z_chain, C[:kk, :kk, active_idx])
            # find all the samples that did this number of leaps, and update self.state with them
            accepted_idx = active_idx[p_cum >= rand_comparison[active_idx]]
            self.counter['L%d'%(kk+1)] += len(accepted_idx)
            self.state.update(accepted_idx, Z_chain[-1])
            # update the set of active indices, so we don't do simulate trajectories for samples that are already assigned to a state
            active_idx = np.flatnonzero(p_cum < rand_comparison)
            Z_chain[-1].active_idx = active_idx
        # flip the momenutm for any samples that were unable to 
        self.state.V[:,active_idx] = -self.state.V[:,active_idx]
        self.counter['F'] += len(active_idx)
        # update the current state
        self.state = SZ
        if self.display > 1:
            print "Transition counts ",
            for k in self.counter.keys():
                print "%s:%d"%(k, self.counter[k]),
            print

        # corrupt the momentum
        self.state.V = self.state.V*np.sqrt(1.-self.beta) + np.random.randn(self.N,self.nbatch)*np.sqrt(self.beta)

    def sample(self, num_steps=100):
        """
        Sample from the target distribution, for num_steps sampling steps.
        This is the function to call from external code.
        """
        for si in range(num_steps):
            if self.display > 0:
                print "sampling step %d / %d,"%(si+1, num_steps),
            self.sampling_iteration()
            if self.display > 0:
                print 
        return self.state.X

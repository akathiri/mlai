import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    """
    This class represent a bandit (slot machine as an example)
    """
    def __init__(self, m):
        self.m = m # true mean
        self.mean = 0 # mean instance estimate
        self.N = 0 # samples
    
    def pull(self):
        """ 
        Pulling bandits arm which
        """
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 -1.0/self.N) + self.mean + 1.0/N * x

    def run_experiment(m, eps, N):
        """
        Run the experiment
        m: list of true mean for different bandits
        eps: choice of epsilon
        N: number of samples
        """
        bandits = [Bandit(_m) for _m in m]
        data = np.empty(N) # for plotting

        for i in range(N):
            # epsilon greedy
            p = np.random.randn()
            if p < eps:
                # explore
                j = np.random.choice(len(m))
            else:
                # exploit (pick the bandit with largest mean)
                j = np.argmax([b.mean for b in bandits])
            x = bandits[j].pull() 
            bandits[j].update(x)

        data[i] = x
    cummulative_avg = np.cumsum(data) / (np.arange(N) + 1)
    return cummulative_avg


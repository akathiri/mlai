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
        self.mean = (1 -1.0/self.N)*self.mean + 1.0/self.N * x

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

    # plotting
    plt.plot(cummulative_avg)
    for _m in m:
        plt.plot(np.ones(N)*_m)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print('True mean ' + str(b.m))
        print('Estimated mean ' + str(b.mean))
        print('-----------')

    return cummulative_avg

if __name__ == '__main__':
    c_1 =  run_experiment([1.0, 2.0, 3.0], 0.1,  100000)
    c_05 = run_experiment([1.0, 2.0, 3.0], 0.05, 100000)
    c_01 = run_experiment([1.0, 2.0, 3.0], 0.01, 100000)

    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()

    # linear plot
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()
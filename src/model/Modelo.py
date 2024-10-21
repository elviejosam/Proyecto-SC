import numpy as np

class BatAlgorithm:
    def __init__(self, n_bats=20, max_iter=100, A=0.5, r=0.5, alpha=0.9, gamma=0.9):
        self.n_bats = n_bats
        self.max_iter = max_iter
        self.A = A
        self.r = r
        self.alpha = alpha
        self.gamma = gamma
        self.bats = None
        self.velocities = None
        self.best_bat = None

    def fitness(self, Rape):
        return np.sum(Rape)

    def initialize(self, data_shape):
        self.bats = np.random.uniform(low=0, high=1, size=(self.n_bats, data_shape))
        self.velocities = np.zeros((self.n_bats, data_shape))
        self.best_bat = self.bats[np.argmin([self.fitness(bat) for bat in self.bats])]

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.n_bats):
                freq = np.random.uniform(0, 1)
                self.velocities[i] += (self.bats[i] - self.best_bat) * freq
                self.bats[i] += self.velocities[i]

                if np.random.rand() > self.r:
                    self.bats[i] = self.best_bat + self.A * np.random.normal(size=self.bats.shape[1])

                if self.fitness(self.bats[i]) < self.fitness(self.best_bat):
                    self.best_bat = self.bats[i]

            self.A *= self.alpha
            self.r *= (1 - np.exp(-self.gamma * t))

        return self.best_bat

import numpy as np

from pymoo.core.crossover import Crossover


class LinearCrossover(Crossover):

    def __init__(self, **kwargs):
        '''
        Linear Recombination Crossover
        ----------------------------
        A crossover operator that performs linear recombination between two parents to produce two offspring. For each parent, a random intersection point is generated.
        '''
        super().__init__(2, 2, **kwargs)

    def _do(self, _, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape


        #random mask
        mask1 = np.random.rand(n_var)
        mask2 = 1 - mask1

        _X = np.copy(X)
        _X[0] = X[1] * mask1 + X[0] * mask2
        _X[1] = X[0] * mask1 + X[1] * mask2

        return _X


class LX(LinearCrossover):
    pass
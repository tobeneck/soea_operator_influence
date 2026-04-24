#vibe coded by copilot
import numpy as np
from pymoo.core.mutation import Mutation


class UniformMutation(Mutation):
    """
    Uniform Mutation Operator for pymoo
    
    This operator randomly changes gene values using uniform random sampling.
    Each gene has a probability of mutation (typically 1/n_var like in Polynomial Mutation).
    When a gene is selected for mutation, it's replaced with a random value from the variable bounds.
    
    Parameters
    ----------
    prob : float, optional
        Mutation probability per gene. If None, defaults to 1/n_var (like PM)
    
    prob_var : float, optional  
        Probability of mutating each variable. If provided, overrides prob calculation
        
    Examples
    --------
    >>> from operators.uniform_mutation import UniformMutation
    >>> mutation = UniformMutation()
    >>> # or with custom probability
    >>> mutation = UniformMutation(prob=0.1)
    """
    
    def __init__(self, prob, prob_var=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.prob_var = prob_var
        
    def _do(self, problem, X, **kwargs):
        """
        Apply uniform mutation to the population
        
        Parameters
        ----------
        problem : Problem
            The optimization problem (contains variable bounds)
        X : np.ndarray
            Population array of shape (n_individuals, n_vars)
            
        Returns
        -------
        np.ndarray
            Mutated population
        """
        # Get problem dimensions and bounds
        n_individuals, n_vars = X.shape
        xl, xu = problem.bounds()
            
        # Create copy for mutation
        Y = X.copy()
        
        # Generate random probabilities for each gene
        do_mutation = np.random.random((n_individuals, n_vars)) < self.prob
        
        # For genes selected for mutation, generate uniform random values within bounds
        for i in range(n_vars):
            # Find individuals where variable i should be mutated
            mutate_mask = do_mutation[:, i]
            n_mutate = np.sum(mutate_mask)
            
            if n_mutate > 0:
                # Generate uniform random values within bounds for this variable
                random_values = np.random.uniform(
                    low=xl[i], 
                    high=xu[i], 
                    size=n_mutate
                )
                # Apply mutation
                Y[mutate_mask, i] = random_values
                
        return Y


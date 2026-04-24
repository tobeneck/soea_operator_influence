import numpy as np

from benchmark_helper import run_test_combinations

from pymoo_problems.soo.rastrigin import Rastrigin

from operators.linear_recombination_crossover import LinearCrossover
from operators.uniform_mutation import UniformMutation

from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.gauss import GaussianMutation

from tea_pymoo.tracing.t_sampling import TracingTypes
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.tracing.t_mutation import T_Mutation


random_populations = {
    "population 1" : np.loadtxt("../data/initial_pops/worst.csv", delimiter=";", dtype=float),
    "population 2" : np.loadtxt("../data/initial_pops/median.csv", delimiter=";", dtype=float),
    "population 3" : np.loadtxt("../data/initial_pops/best.csv", delimiter=";", dtype=float),
}

problems = {
    "Rastrigin" : Rastrigin(n_var=10),
}


crossovers = {
    "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR), 
    "LX" : T_Crossover(crossover=LinearCrossover(), tracing_type=TracingTypes.TRACE_VECTOR),
}

mutations = {
    "RM" : T_Mutation(mutation=UniformMutation(prob=0.1), tracing_type=TracingTypes.TRACE_VECTOR, value_dependent=False), # 0.1 as 1 % mutation chance for genome size of 10
    "GM" : T_Mutation(mutation=GaussianMutation(prob=0.1), tracing_type=TracingTypes.TRACE_VECTOR, value_dependent=True), # 0.1 as 1 % mutation chance for genome size of 10
}

run_test_combinations(
    problems=problems,
    output_folder="../data/test_results",
    random_populations=random_populations,
    algorithms=["GA"],
    n_gen=100,
    mutations=mutations,
    crossovers=crossovers,
)
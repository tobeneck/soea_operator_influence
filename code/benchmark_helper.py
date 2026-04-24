from tea_pymoo.tracing.t_mutation import T_Mutation
from tea_pymoo.tracing.t_crossover import T_Crossover
from tea_pymoo.callbacks.general.counting_impact_pop_callback import Counting_Impact_Pop_Callback
from tea_pymoo.callbacks.general.fitness_impact_pop_callback import Fitness_Impact_Pop_Callback
from tea_pymoo.callbacks.general.entropy_impact_pop_callback import Entropy_Impact_Pop_Callback
from tea_pymoo.callbacks.general.fitness_entropy_impact_pop_callback import Fitness_Entropy_Impact_Pop_Callback
from tea_pymoo.callbacks.general.counting_impact_inds_callback import Counting_Impact_Inds_Callback
from tea_pymoo.callbacks.general.genome_callback import Genome_Callback
from tea_pymoo.callbacks.general.entropy_impact_inds_callback import Entropy_Impact_Inds_Callback


from tea_pymoo.callbacks.soo.performance_callback import Performance_Callback
from tea_pymoo.callbacks.soo.fitness_callback import Fitness_Callback

from tea_pymoo.callbacks.accumulated_callback import AccumulateCallbacks
from tea_pymoo.tracing.tracing_types import TracingTypes
from tea_pymoo.tracing.t_sampling import T_Sampling

from pymoo.optimize import minimize
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover

from pymoo.algorithms.soo.nonconvex.ga import GA

from datetime import datetime

import numpy as np
    
def run_test_combinations(
        problems:dict,
        output_folder:str,
        random_populations:dict,
        crossovers:dict= {
            "UX" : T_Crossover(crossover=UniformCrossover(), tracing_type=TracingTypes.TRACE_VECTOR), #TODO: change operators!
            #"SBX" : T_Crossover(crossover=SimulatedBinaryCrossover(prob=1.0, eta=20), tracing_type=TracingTypes.TRACE_VECTOR),
        },
        mutations:dict= {
            "PM" : T_Mutation(mutation=PolynomialMutation(prob=1.0, eta=20), tracing_type=TracingTypes.TRACE_VECTOR), # TODO: change
        },
        algorithms=["GA"],
        n_gen:int=20,
        pop_size:int=20,
        tracing_type:TracingTypes=TracingTypes.TRACE_VECTOR,
        ):
    start_time=datetime.now()
    t_sampling = T_Sampling(FloatRandomSampling(), tracing_type=tracing_type)

    #run the tests:
    for problem_name in problems:#iterate over the problems
        problem= problems[problem_name]
        for crossover_name in crossovers:#iterate over the crossovers
            for mutation_name in mutations:#iterate over the crossovers
                for algorithm_name in algorithms:#iterate over the algorithms
                    for pop_name in random_populations:#iterate over the different initial populations
                    

                        elapsed_time = datetime.now()-start_time
                        print("processing problem", problem_name, "with crossover", crossover_name, "and initial pop", pop_name, "using the algorithm", algorithm_name, "and current runtime of", elapsed_time.days,"d", elapsed_time.seconds // 3600, "h", elapsed_time.seconds // 60 % 60, "m",  elapsed_time.seconds % 60, "s")

                        pop = t_sampling.do(problem, pop_size, seeds=random_populations[pop_name])

                        algorithm = GA(
                            pop_size=pop_size,
                            sampling=pop, 
                            crossover=crossovers[crossover_name],
                            mutation=mutations[mutation_name],
                            eliminate_duplicates=True
                        )

                        for i in range(31):#31 re-runs as usual
                            #set up callbacks:
                            additional_run_info = {
                                "run_number": i,
                                "crossover": crossover_name,
                                "mutation": mutation_name,
                                #"problem_name": problem_name,
                                "initial_population": pop_name,
                                }
                            callbacks = [
                            #pop callbacks:
                            Counting_Impact_Pop_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="counting_impact_pop"),
                            Fitness_Impact_Pop_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="fitness_impact_pop"),
                            Entropy_Impact_Pop_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="entropy_impact_pop"),
                            Fitness_Entropy_Impact_Pop_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="fitness_entropy_impact_pop"),
                            
                            # #ind callbacks:
                            # Counting_Impact_Inds_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="counting_impact_inds"),
                            # Entropy_Impact_Inds_Callback(additional_run_info = additional_run_info, initial_popsize = pop_size, tracing_type=tracing_type, optimal_inds_only=False, filename="entropy_impact_inds"),

                            #general callbacks:
                            Performance_Callback(additional_run_info=additional_run_info, filename="performance"),
                            Fitness_Callback(additional_run_info=additional_run_info, filename="fitness"),
                            Genome_Callback(additional_run_info=additional_run_info, n_var=problem.n_var, filename="genome"),
                            ]
                            callback = AccumulateCallbacks(collectors=callbacks)

                            #run the test
                            res = minimize(problem,
                                        algorithm,
                                        ('n_gen', n_gen),
                                        seed=i,#seed is the run number!
                                        verbose=False,
                                        callback=callback
                                        )
                            
                            #print output:
                            callback.finalize(output_folder)
    end_time=datetime.now()
    elapsed = end_time - start_time
    print("Took",elapsed.days, "d", elapsed.seconds // 3600, "h", elapsed.seconds // 60 % 60, "m", elapsed.seconds % 60, "s")
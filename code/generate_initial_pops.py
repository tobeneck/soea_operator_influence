import numpy as np
import random
import matplotlib.pyplot as plt
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo_problems.soo.rastrigin import Rastrigin
import pickle
import os

random.seed(42)
np.random.seed(42)

pop_size = 20
n_var = 10
n_populations = 1000

# ==================================================================================================
# generate 1000 populations
# ==================================================================================================



problem = Rastrigin(n_var=n_var)
sampling = FloatRandomSampling()

populations = []
best_fitness = []
avg_fitness = []
median_fitness = []
worst_fitness = []

print(f"Generating {n_populations} populations with {pop_size} individuals each...")

for i in range(n_populations):

    # generate pop
    pop = sampling(problem, pop_size)
    
    # evaluate pop
    pop.set("F", problem.evaluate(pop.get("X")))

    # calculate statistics
    fitness_values = pop.get("F").flatten()
    best_fit = np.min(fitness_values)
    avg_fit = np.mean(fitness_values)
    med_fit = np.median(fitness_values)
    worst_fit = np.max(fitness_values)

    #save the data
    populations.append(pop)
    best_fitness.append(best_fit)
    avg_fitness.append(avg_fit)
    median_fitness.append(med_fit)
    worst_fitness.append(worst_fit)

worst_pop_index = np.argmax(avg_fitness)
median_pop_index = np.argsort(avg_fitness)[len(avg_fitness)//2]
best_pop_index = np.argmin(avg_fitness)


print(f"best pop worst fitness {worst_fitness[best_pop_index]}, median fitness: {median_fitness[best_pop_index]}, and best fitness: {best_fitness[best_pop_index]}")
print(f"median pop worst fitness {worst_fitness[median_pop_index]}, median fitness: {median_fitness[median_pop_index]}, and best fitness: {best_fitness[median_pop_index]}")
print(f"worst pop worst fitness {worst_fitness[worst_pop_index]}, median fitness: {median_fitness[worst_pop_index]}, and best fitness: {best_fitness[worst_pop_index]}")

np.savetxt("../data/initial_pops/worst.csv", populations[worst_pop_index].get("X"), delimiter=";")
np.savetxt("../data/initial_pops/median.csv", populations[median_pop_index].get("X"), delimiter=";")
np.savetxt("../data/initial_pops/best.csv", populations[best_pop_index].get("X"), delimiter=";")















def generate_and_evaluate_populations(n_populations=1000, pop_size=20, n_var=10):
    """
    Generiert n_populations Populationen und evaluiert sie mit der Rastrigin-Funktion
    
    Parameters:
    -----------
    n_populations : int
        Anzahl der zu generierenden Populationen
    pop_size : int
        Größe jeder Population
    n_var : int
        Anzahl der Variablen für die Rastrigin-Funktion
        
    Returns:
    --------
    populations : list
        Liste aller generierten Populationen
    avg_fitness : list
        Liste der durchschnittlichen Fitness-Werte für jede Population
    """
    
    # Initialisiere Problem und Sampling
    problem = Rastrigin(n_var=n_var)
    sampling = FloatRandomSampling()
    
    populations = []
    avg_fitness = []
    median_fitness = []
    worst_fitness = []
    
    print(f"Generiere {n_populations} Populationen mit je {pop_size} Individuen...")
    
    for i in range(n_populations):
        if (i + 1) % 100 == 0:
            print(f"Fortschritt: {i + 1}/{n_populations}")
        
        # Generiere Population
        pop = sampling(problem, pop_size)
        
        # Evaluiere Population
        pop.set("F", problem.evaluate(pop.get("X")))
        
        # Berechne Fitness-Statistiken für diese Population
        fitness_values = pop.get("F").flatten()
        avg_fit = np.mean(fitness_values)
        med_fit = np.median(fitness_values)
        worst_fit = np.max(fitness_values)
        
        populations.append(pop)
        avg_fitness.append(avg_fit)
        median_fitness.append(med_fit)
        worst_fitness.append(worst_fit)
    
    return populations, avg_fitness, median_fitness, worst_fitness

def select_representative_populations(populations, avg_fitness):
    """
    Wählt die beste, median und schlechteste Population basierend auf durchschnittlicher Fitness aus
    
    Parameters:
    -----------
    populations : list
        Liste aller Populationen
    avg_fitness : list
        Liste der durchschnittlichen Fitness-Werte
        
    Returns:
    --------
    dict mit den ausgewählten Populationen und ihren Indices
    """
    
    avg_fitness = np.array(avg_fitness)
    
    # Finde Indices für beste, median und schlechteste Population
    best_idx = np.argmin(avg_fitness)  # Niedrigste Fitness ist beste bei Rastrigin
    worst_idx = np.argmax(avg_fitness)  # Höchste Fitness ist schlechteste
    
    # Sortiere für Median
    sorted_indices = np.argsort(avg_fitness)
    median_idx = sorted_indices[len(sorted_indices) // 2]
    
    return {
        'best': {
            'population': populations[best_idx],
            'avg_fitness': avg_fitness[best_idx],
            'index': best_idx
        },
        'median': {
            'population': populations[median_idx],
            'avg_fitness': avg_fitness[median_idx],
            'index': median_idx
        },
        'worst': {
            'population': populations[worst_idx],
            'avg_fitness': avg_fitness[worst_idx],
            'index': worst_idx
        }
    }

def save_results(populations, fitness_stats, selected_pops, output_dir="results"):
    """
    Speichert die Ergebnisse in Dateien
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichere alle Populationen
    with open(os.path.join(output_dir, "all_populations.pkl"), "wb") as f:
        pickle.dump(populations, f)
    
    # Speichere Fitness-Statistiken
    np.save(os.path.join(output_dir, "fitness_stats.npy"), fitness_stats)
    
    # Speichere ausgewählte Populationen
    with open(os.path.join(output_dir, "selected_populations.pkl"), "wb") as f:
        pickle.dump(selected_pops, f)
    
    print(f"Ergebnisse gespeichert in: {output_dir}/")


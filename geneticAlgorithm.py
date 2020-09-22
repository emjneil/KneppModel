# ---- Genetic Algorithm ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string
from geneticalgorithm import geneticalgorithm as ga


# # # # # --------- MINIMIZATION ----------- # # # # # # # 

species = ['roeDeer','rabbits','fox','songbirdsWaterfowl','raptors','reptilesAmphibians','arableGrass','woodland','thornyScrub','wetland']

def ecoNetwork(t, x, interaction_strength_chunk, rowContents_growth):
    # define new array to return
    output_array = []
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * x[outer_index]
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount += interaction_strength_chunk[outer_species][inner_species] * x[outer_index] * x[inner_index]
            if amount.item() >= 1e5:
                amount = None
                break
        # append values to output_array
        output_array.append(amount)
    # return array
    return output_array


def objectiveFunction(x): 
    # insert growth rate of 0 (minimizer isn't guessing)
    x = np.insert(x,19,0)
    # insert interaction matrices of 0
    x = np.insert(x,20,0)
    x = np.insert(x,21,0)
    x = np.insert(x,22,0)
    x = np.insert(x,23,0)
    x = np.insert(x,24,0)
    x = np.insert(x,25,0)
    x = np.insert(x,29,0)
    x = np.insert(x,30,0)
    x = np.insert(x,31,0)
    x = np.insert(x,33,0)
    x = np.insert(x,35,0)
    x = np.insert(x,39,0)
    x = np.insert(x,40,0)
    x = np.insert(x,42,0)
    x = np.insert(x,44,0)
    x = np.insert(x,46,0)
    x = np.insert(x,47,0)
    x = np.insert(x,48,0)
    x = np.insert(x,49,0)
    x = np.insert(x,50,0)
    x = np.insert(x,51,0)
    x = np.insert(x,53,0)
    x = np.insert(x,55,0)
    x = np.insert(x,56,0)
    x = np.insert(x,57,0)
    x = np.insert(x,58,0)
    x = np.insert(x,59,0)
    x = np.insert(x,60,0)
    x = np.insert(x,62,0)
    x = np.insert(x,64,0)
    x = np.insert(x,66,0)
    x = np.insert(x,67,0)
    x = np.insert(x,68,0)
    x = np.insert(x,69,0)
    x = np.insert(x,70,0)
    x = np.insert(x,71,0)
    x = np.insert(x,73,0)
    x = np.insert(x,75,0)
    x = np.insert(x,76,0)
    x = np.insert(x,77,0)
    x = np.insert(x,78,0)
    x = np.insert(x,79,0)
    x = np.insert(x,82,0)
    x = np.insert(x,83,0)
    x = np.insert(x,84,0)
    x = np.insert(x,85,0)
    x = np.insert(x,86,0)
    x = np.insert(x,87,0)
    x = np.insert(x,88,0)
    x = np.insert(x,89,0)
    x = np.insert(x,92,0)
    x = np.insert(x,95,0)
    x = np.insert(x,97,0)
    x = np.insert(x,99,0)
    x = np.insert(x,102,0)
    x = np.insert(x,104,0)
    x = np.insert(x,108,0)
    x = np.insert(x,109,0)
    x = np.insert(x,111,0)
    x = np.insert(x,112,0)
    x = np.insert(x,119,0)
    # define X0, growthRate, interactionMatrix
    X0 = x[0:10] 
    growthRate = x[10:20]
    growthRate = np.asarray(growthRate)
    rowContents_growth = pd.DataFrame(data=growthRate.reshape(1,10),columns=species)
    interaction_strength = x[20:120]
    interaction_strength_chunk = pd.DataFrame(data=interaction_strength.reshape(10,10),index = species, columns=species)
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    print(y[49:50:,]*200)
    # roe deer, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland
    result = (((y[49:50:, 6]-(3.6/200))**2) + ((y[49:50:, 2]-(8.4/200))**2) + ((y[49:50:, 1]-(296.6/200))**2) + ((y[49:50:, 4]-(0.91/200))**2) + ((y[49:50:, 5]-(20.4/200))**2) + ((y[49:50:, 0]-(247.1/200))**2)  + ((y[49:50:, 3]-(13.62/200))**2) + ((y[49:50:, 8]-(0.47/200))**2) + ((y[49:50:, 7]-(0.76/200))**2))
    print(result)
    return (result)


bds = np.array([[0.0675,0.51],[0.00135,2.97],[0.0055,0.079],[0.000075,0.135],[0.00055,0.0085],[0.0017,0.202],[0.01,0.02],[0.00225,0.00315],[0.00025,0.00225],[0.00008,0.0001],
    # growth
    [0,5],[0,5],[0,1],[0,5],[0,1],[0,5],[0,5],[0,5],[0,5],
    # interaction bds
    [-1,0],[-1,0],[-1,0],
    [0,1],[0,1],[-1,0],[-1,0],[-1,0],
    [-1,0],[-1,0],[-1,0],
    [0,1],[0,1],
    [-1,0],[-1,0],[-1,0],
    [0,1],[0,1],
    [0,1],[0,1],
    [0,1],[0,1],[0,1],[0,1],[-1,0],[-1,0],
    [0,1],[0,1],[0,1],[0,1],[-1,0],[0,1],
    [0,1],[0,1],[0,1],[0,1],[-1,0],[-1,0],[-1,0]])


algorithm_param = {'max_num_iteration': 1000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


optimization =  ga(function = objectiveFunction, dimension = 58, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=120)
optimization.run()
print(optimization)


# species = ['roeDeer','rabbits','fox','songbirdsWaterfowl','raptors','reptilesAmphibians','arableGrass','woodland','thornyScrub','wetland']

# order of outputs 
# ['arableGrass',    'fox',         'rabbits',      'raptors',          'reptiles',       'roeDeer',     'songbirdsWaterfowl', 'thornyScrub',            'wetland',       'woodland'])
#   2.9-4.2 (3.6)   1.1-15.7 (8.4)    0.27-593 (297)  0.11-1.7(0.91)     0.3-40(20.4)    4.9-449 (247)        0.15-27 (13.62)    0.049-0.445 (0.47)           0.018       0.45-0.63 (0.76)

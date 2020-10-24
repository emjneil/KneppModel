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

species = ['arableGrass','organicCarbon','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    return X * (r + np.matmul(A, X))

def objectiveFunction(x): 
    # insert interaction matrices of 0
    x = np.insert(x,6,0)
    x = np.insert(x,16,0)
    x = np.insert(x,20,0)
    x = np.insert(x,21,0)
    x = np.insert(x,25,0)
    x = np.insert(x,26,0)
    # define X0, growthRate, interactionMatrix
    X0 = [1] * len(species)
    # growth rates
    growthRate = x[0:5]
    r = np.asarray(growthRate)
    interaction_strength = x[5:30]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(5,5),index = species, columns=species)
    A = interaction_strength.to_numpy()
    t = np.linspace(0, 5, 50)
    results = solve_ivp(ecoNetwork, (0, 5), X0,  t_eval = t, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    print(y[49:50:,])
    result = (((y[49:50, 0]-0.86)**2) +  ((y[49:50, 1]-1.4)**2) + ((y[49:50, 2]-2.2)**2) + ((y[49:50, 3]-7)**2) + ((y[49:50, 4]-0.91)**2))
    print(result)
    return (result)

# order of outputs   
# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.86            1.4        2.2              7               0.91


bds = np.array([
    # growth
    [0,1],[0,1],[0,1],[0,1],[0,1],
    # interaction bds
    [-1,0],[-1,0],[-1,0],[-1,0],
    [0,1],[-1,0],[0,1],[0,1],[0,1],
    [0,1],[-1,0],[0,1],[0,1],
    [-1,0],[-1,0],[-1,0],
    [-1,0],[0,1],[-1,0]])


algorithm_param = {'max_num_iteration': 500,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


optimization =  ga(function = objectiveFunction, dimension = 24, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=20)
optimization.run()
print(optimization)

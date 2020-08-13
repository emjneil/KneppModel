# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# download packages
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
import networkx as nx
import timeit
import seaborn as sns


# # # # # --------- MINIMIZATION ----------- # # # # # # # 

interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
# store species in a list
species = list(interactionMatrix_csv.columns.values)

def ecoNetwork(t, X, interaction_strength_chunk, rowContents_growth):
    # define new array to return
    output_array = []
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * X[outer_index]
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount += interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index]
        # append values to output_array
        output_array.append(amount)
    # return array
    return output_array
# Filter justifications
    # roe deer start out 2.25 to 112.46; filter is 2.25 to 449.8. Average = 226.0
    # arableGrass start out 338 to 401; filter is to 267 to 401; average = 334
    # woodland start out 22.3 to 62.3; filter is to 53.4 to 97.9; average = 75.7
    # thornyScrub start out 4.5 to 44.5; filter is 4.9 to 89; average = 46.95
all_iterations_outputs = []
allIterationsResults = []

def objectiveFunction(x):
    X0 = x[0:13] 
    growthRate = x[13:26]
    rowContents_growth = pd.DataFrame(data=growthRate.reshape(1,13),columns=species)
    interaction_strength = x[26:195]
    interaction_strength_chunk = pd.DataFrame(data=interaction_strength.reshape(13,13),index = species, columns=species)
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    outputs = (y[49:50:,])
    print(y[49:50:,])
    result = (((y[49:50:, 1]-(226/200))**2) + ((y[49:50:, 5]-(49.885/200))**2) + ((y[49:50:, 7]-(4.32/200))**2) + ((y[49:50:, 9]-(334/200))**2) + ((y[49:50:, 10]-(75.7/200))**2) + ((y[49:50:, 11]-(46.95/200))**2))
    all_iterations_outputs.append(outputs)
    allIterationsResults.append(result)
    print(result)
    return result

# here are my guesses to start at
X0guess_init = [0,42,0,0,208,52,12,4,20,346,48,36,17]
# normalize X0
X0guess = [x / 200 for x in X0guess_init]
growthGuess = [0,0.38,0,0,0.47,0.65,0.1,0.1,0.1,0.94,0.10,0.38,0]
interactionGuess = [0,0,0,0,0,0,0,0,0,-0.056,-0.056,-0.056,0,
                    0,0,0,0,0,0.305,0,0,0,-0.18,-0.055,-0.055,0,
                    0,0,0,0,0,0,0,0,-0.055,-0.34,-0.056,-0.056,0,
                    0,0,0,0,0,0,0,0,0,-0.57,-0.38,0,0.94,
                    0,0,0,0,0,0.056,0,0.94,0,-0.23,-0.056,-0.13,0,
                    0,-0.055,0,0,-0.58,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0.944,0,0,0,0,0,
                    0,0,0,0,-0.056,0,-0.056,0,-0.056,0,0,0,0,
                    0,0,0.93,0,0,0,0,0.93,0,0,0,0,0,
                    0.93,0.93,0.93,0.93,0.93,0,0,0,0,0,0,0,0,
                    0.93,0.93,0.93,0.93,0.93,0,0,0,0,-0.84,0,-0.53,0,
                    0.93,0.93,0.93,0,0.38,0,0.93,0,0.93,-0.95,0.866,0,0,
                    0.93,0.93,0.93,0.93,0,0,0.93,0.93,0.93,-0.056,-0.056,-0.056,0
]

combined = X0guess + growthGuess + interactionGuess
guess = np.array(combined)

# define bounds for each variable; make sure X0 bounds are normalized
X0bds = ((0,0),(2.25/200,112.46/200),(0,0),(0,0),(2.7/200,452/200),(1.12/200,97.53/200),(0.15/200,27/200),(0.11/200,8.42/200),(0.311/200,40.4/200),(338/200,401/200),(22.3/200,62.3/200),(4.5/200,44.5/200),(15/200,24/200))
growthbds = ((0,0),(0,1),(0,0),(0,0),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,0))
interactionbds = ((0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
                    (0,0),(0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
                    (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(-1,0),(0,0),
                    (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0), (0,0),(0,1),
                    (0,0),(0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
                    (0,0),(-1,0),(0,0),(0,0),(-1,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,0),(0,0),(0,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(0,0),(0,0),(0,0),
                    (0,0),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,1),(0,1),(0,1),(0,1),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,1),(0,1),(0,1),(0,1),(0,1),(0,0),(0,0),(0,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),
                    (0,1),(0,1),(0,1),(0,0),(0,1),(0,0),(0,1),(0,0),(0,1),(-1,0),(0,1),(0,0),(0,0),
                    (0,1),(0,1),(0,1),(0,1),(0,0),(0,0),(0,1),(0,1),(0,1),(-1,0),(-1,0),(-1,0),(0,0))


# combine them into one dataframe
bds = X0bds + growthbds + interactionbds

optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'Powell', options ={'maxiter': 100}, tol=0.1)
# optimization = optimize.differential_evolution(objectiveFunction, bounds = bds, maxiter = 1, tol=10000)
print(optimization)
# print(all_iterations_outputs)
# print(allIterationsResults)

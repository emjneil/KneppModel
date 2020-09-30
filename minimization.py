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
from hyperopt import hp, fmin, tpe, space_eval


# from GPyOpt.methods import BayesianOptimization
from skopt import gp_minimize


# # # # # --------- MINIMIZATION ----------- # # # # # # # 

species = ['roeDeer','rabbits','fox','songbirdsWaterfowl','raptors','reptilesAmphibians','arableGrass','woodland','thornyScrub','wetland']

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
    # with pd.option_context('display.max_columns', None):
    #     print(X0, rowContents_growth, interaction_strength_chunk)

    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    print(y[49:50:,]*200)
# roe deer, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland
    result = (((y[49:50:, 6]-(3.6/200))**2) + ((y[49:50:, 2]-(8.4/200))**2) + ((y[49:50:, 1]-(296.6/200))**2) + ((y[49:50:, 4]-(0.91/200))**2) + ((y[49:50:, 5]-(20.4/200))**2) + ((y[49:50:, 0]-(247.1/200))**2)  + ((y[49:50:, 3]-(13.62/200))**2) + ((y[49:50:, 8]-(0.47/200))**2) + ((y[49:50:, 7]-(0.76/200))**2))
    # multiply it by how many filters were passed
    # all_filters = [4.9 <= (y[49:50:, 0]*200) <= 449.8, 0.27 <= (y[49:50:, 1]*200) <= 593, 0.34 <= (y[49:50:, 5]*200) <= 40.4, 1.1 <= (y[49:50:, 2]*200) <= 15.7, 0.24 <= (y[49:50:, 3]*200) <= 27.0,  0.11 <= (y[49:50:, 4]*200) <= 1.7, 2.9 <= (y[49:50:, 6]*200) <= 4.2, 0.45 <= (y[49:50:, 7]*200) <= 0.63, 0.049 <= (y[49:50:, 8]*200) <= 0.45]
    # result2 = sum(all_filters)
    # # return total number of filters minus filters passed
    # result3 = 9-result2
    # result = result3 * result1
    print(result)
    return (result)

# here are my guesses to start at (X0 normalized / 200)
X0guess = [0.43, 0.22, 0.07, 0.069, 0.0063, 0.2, 0.017, 0.0024, 0.0017, 0.000089]
growthGuess = [3.5, 3.2, 0.36, 0.03, 0.80, 4.43, 1.44, 3.4, 4.84]
interactionGuess = [
                    -0.38,-0.74,-0.41,
                    0.84,0.41,-0.36,-0.28,-0.35,
                    -0.93,-0.97,-0.39,
                    0.95,0.090,
                    -0.91,-0.11,-0.96,
                    0.2,0.85,
                    0.93,0.32,
                    0.26,0.23,0.59,0.86,-0.17,-0.039,
                    0.89,0.28,0.095,0.60,-0.99,0.11,
                    0.38,0.050,0.62,0.63,-0.24,-0.79,-0.58
]

combined = X0guess + growthGuess + interactionGuess
guess = np.array(combined)

# roe deer, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland

X0bds = ((13.5/200,101.1/200),(0.27/200,593/200),(1.1/200,15.7/200),(0.15/200,27/200),(0.11/200,1.7/200),(0.34/200,40.4/200),(2.05/200,4/200),(0.45/200,0.63/200),(0.05/200,0.45/200),(0.016/200,0.02/200))
growthbds = ((0,5),(0,5),(0,1),(0,5),(0,1),(0,5),(0,5),(0,5),(0,5))
interactionbds = (
                    (-1,0),(-1,0),(-1,0),
                    (0,1),(0,1),(-1,0),(-1,0),(-1,0),
                    (-1,0),(-1,0),(-1,0),
                    (0,1),(0,1),
                    (-1,0),(-1,0),(-1,0),
                    (0,1),(0,1),
                    (0,1),(0,1),
                    (0,1),(0,1),(0,1),(0,1),(-1,0),(-1,0),
                    (0,1),(0,1),(0,1),(0,1),(-1,0),(0,1),
                    (0,1),(0,1),(0,1),(0,1),(-1,0),(-1,0),(-1,0))

# combine them into one dataframe
bds = X0bds + growthbds + interactionbds

# optimization = gp_minimize(objectiveFunction, dimensions = bds, x0 = guess2)

#L-BFGS-B, Powell, TNC, SLSQP, can have bounds
optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-6)
print(optimization)

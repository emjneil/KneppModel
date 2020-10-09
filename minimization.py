# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

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

species = ['arableGrass','fox','rabbits','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, interaction_strength_chunk, rowContents_growth):
    # make X zero if it falls under some threshold
    for n, i in enumerate(X):
        if i < 1e-8:
            X[n] = 0
    # define new array to return
    output_array = []
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * X[outer_index]
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount2 = amount + (interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index])
        # append values to output_array
        output_array.append(amount2)
    # return array
    return output_array


def objectiveFunction(x): 
    # insert interaction matrices of 0
    x = np.insert(x,13,0)
    x = np.insert(x,16,0)
    x = np.insert(x,17,0)
    x = np.insert(x,18,0)
    x = np.insert(x,21,0)
    x = np.insert(x,22,0)
    x = np.insert(x,23,0)
    x = np.insert(x,27,0)
    x = np.insert(x,31,0)
    x = np.insert(x,32,0)
    x = np.insert(x,37,0)
    x = np.insert(x,43,0)
    # define X0, growthRate, interactionMatrix
    X0 = x[0:6] 
    growthRate = x[6:12]
    growthRate = np.asarray(growthRate)
    growthRate_table = pd.DataFrame(data=growthRate.reshape(1,6),columns=species)
    rowContents_growth = growthRate_table.squeeze()
    interaction_strength = x[12:48]
    interaction_strength_chunk = pd.DataFrame(data=interaction_strength.reshape(6,6),index = species, columns=species)
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    print(y[49:50:,])
    result = (((y[49:50:, 0]-3.6)**2) +  ((y[49:50:, 1]-1.2)**2) + ((y[49:50:, 2]-1130.5)**2) + ((y[49:50:, 3]-6.7)**2)  + ((y[49:50:, 4]-.5)**2) + ((y[49:50:, 5]-.8)**2))
    print(result)
    return (result)

# species = ['arableGrass','fox','rabbits','roeDeer','thornyScrub','woodland']

# here are my guesses to start at
X0guess = [2.8,0.8,829.5,3.1,0.4,0.6]
growthGuess = [0.1,0.07,0.02,0.07,0.05,0.1]
interactionGuess = [
                    -0.21,0.71,0.84,
                    -0.53,-0.19,
                    -0.49,0.31,-0.28,-0.49,-0.11,
                    -0.44,-0.5,-0.39,-0.36,
                    -0.51,0.92,0.5,-0.19,0.38,
                    -0.1,0.04,0.09,-0.36,-0.90
]

combined = X0guess + growthGuess + interactionGuess
guess = np.array(combined)


X0bds = ((2.1,4),(0.2,2.2),(1,2260),(1.3,5.3),(0.1,0.5),(0.5,0.6))
growthbds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
interactionbds = (
                    (-1,0),(0,1),(0,1),
                    (-1,0),(-1,0),
                    (-1,0),(0,1),(-1,0),(-1,0),(-1,0),
                    (-1,0),(-1,0),(-1,0),(-1,0),
                    (-1,0),(0,1),(0,1),(-1,0),(0,1),
                    (-1,0),(0,1),(0,1),(-1,0),(-1,0))

# combine them into one dataframe
bds = X0bds + growthbds + interactionbds

# optimization = gp_minimize(objectiveFunction, dimensions = bds, x0 = guess2)

#L-BFGS-B, Powell, TNC, SLSQP, can have bounds
optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-6)
print(optimization)



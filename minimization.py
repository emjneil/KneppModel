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

species = ['arableGrass','fox','organicCarbon', 'rabbits','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    # return array
    return X * (r + np.matmul(A, X))


def objectiveFunction(x): 
    # insert interaction matrices of 0
    x = np.insert(x,15,0)
    x = np.insert(x,16,0)
    x = np.insert(x,21,0)
    x = np.insert(x,23,0)
    x = np.insert(x,25,0)
    x = np.insert(x,26,0)
    x = np.insert(x,27,0)
    x = np.insert(x,37,0)
    x = np.insert(x,39,0)
    x = np.insert(x,43,0)
    x = np.insert(x,44,0)
    x = np.insert(x,45,0)
    x = np.insert(x,49,0)
    x = np.insert(x,50,0)
    x = np.insert(x,51,0)
    x = np.insert(x,56,0)
    x = np.insert(x,57,0)
    x = np.insert(x,58,0)
    # define X0, growthRate, interactionMatrix
    X0 = x[0:7] 
    # growth rates
    growthRate = x[7:14]
    r = np.asarray(growthRate)
    interaction_strength = x[14:65]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(7,7),index = species, columns=species)
    A = interaction_strength.to_numpy()
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    print(y[49:50:,]/0.0001)
    result = (((y[49:50, 0]-0.00036)**2) +  ((y[49:50, 1]-0.00071)**2) + ((y[49:50, 2]-0.00015)**2) + ((y[49:50, 3]-0.1131)**2) + ((y[49:50, 4]-0.00067)**2)  + ((y[49:50, 5]-0.00006)**2) + ((y[49:50, 6]-0.00005)**2))
    print(result)
    return (result)

# order of outputs   
# ['arableGrass',    'fox'                organicCarb     'rabbits'         'roeDeer',     'thornyScrub',            'woodland'])
#   3.3-4 (3.6)    0.22 to 13.9 (7.1)     1-2(1.5)           1-2260 (1130.5)      4.9-9(6.7)     0.05 to 0.89 (0.6)      0.2-0.8 (0.5)

# here are my guesses to start at
X0guess = [0.00038, 0.000096, 0.0001, 0.18, 0.00034, 0.000022, 0.000067]
growthGuess = [0.037, 0.17, 0.002, 0.07, 0.05, 0.16, 0.04]
interactionGuess = [
                    -0.18,-0.39,-0.95,-0.96,-0.19,
                    -0.09,0.19,
                    0.96,0.9,-0.18,0.15,0.52,0.2,0.56,
                    0.03,-0.06,-0.87,0.8,0.23,
                    0.6,-0.17,0.63,0.44,
                    -0.6,-0.32,-0.09,-0.87,
                    -0.32,-0.78,0.9,-0.67
]

combined = X0guess + growthGuess + interactionGuess
guess = np.array(combined)


X0bds = ((3.6/10000,4.2/10000),(0.2/10000,2.2/10000),(1/10000,1.1/10000),(1/10000,2260/10000),(1.3/10000,5.3/10000),(0.1/10000,0.5/10000),(0.2/10000,0.7/10000))
growthbds = ((0,1),(0.5/7.85,3.49/7.85),(0,1),(0,1),(0.15/7.85,0.45/7.85),(0,1),(0,1))
interactionbds = (
                    (-1,0),(-1,0),(-1,0),(-1,0),(-1,0),
                    (-1,0),(0,1),
                    (0,1),(0,1),(-1,0),(0,1),(0,1),(0,1),(0,1),
                    (0,1),(-1,0),(-1,0),(0,1),(0,1),
                    (0,1),(-1,0),(0,1),(0,1),
                    (-1,0),(-1,0),(-1,0),(-1,0),
                    (-1,0),(-1,0),(0,1),(-1,0),
                    )
# combine them into one dataframe
bds = X0bds + growthbds + interactionbds

# optimization = gp_minimize(objectiveFunction, dimensions = bds, x0 = guess2)

#L-BFGS-B, Powell, TNC, SLSQP, can have bounds
optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-6)
print(optimization)



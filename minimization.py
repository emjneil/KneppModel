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

species = ['arableGrass','organicCarbon','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-6] = 0
    X[X>100] = 100
    # return array
    return X * (r + np.matmul(A, X))


def objectiveFunction(x): 
    # insert interaction matrices of 0
    x = np.insert(x,6,0)
    x = np.insert(x,16,0)
    x = np.insert(x,21,0)
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
    result = (((y[49:50, 0]-0.86)**2) +  ((y[49:50, 1]-1.4)**2) + ((y[49:50, 2]-2.2)**2) + ((y[49:50, 3]-11.1)**2) + ((y[49:50, 4]-0.91)**2))
    print(result)
    return (result)


# order of outputs   
# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.86            1.4        2.2              11.1              0.91


growthbds = ((0.1,1),(0,0.1),(0.1,0.5),(0.1,1),(0.1,1))
# arable grass is mostly impacted by thorny scrub & woodland (not self-impacts or roe)
interactionbds = (
                    (-1,0),(0,1),(-1,0),(-1,0),
                    (0.1,0.25),(-0.8,-0.6),(0.1,0.2),(0,0.1),(0.1,0.2),
                    (0,0.25),(-1,0),(0,0.25),(0,0.25),
                    (-0.25,0),(-0.1,0),(-0.1,0),(-1,0),
                    (-0.25,0),(-0.1,0),(0,1),(-1,0),
                    )
# combine them into one dataframe
bds = growthbds + interactionbds

#L-BFGS-B, Powell, TNC, SLSQP, can have bounds
# optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-6)
optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 1000)
print(optimization)



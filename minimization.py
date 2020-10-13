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
    # define new array to return
    output_array = []
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * X[outer_index]
        # second loop
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount2 = amount + (interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index]) 
        # append values to output_array
        output_array.append(amount2)
    # stop things from going to zero
    for i, (a,b) in enumerate(zip(X, output_array)):
        if a + b < 1e-8:
            X[i] = 0
            output_array[i] = 0
    # rescale habitat data
    totalHabitat = ((output_array[0] + X[0]) + (output_array[4] + X[4]) + (output_array[5] + X[5]))
    # if habitats aren't all at zero, scale them to 4.5
    if totalHabitat != 0 or None:
        output_array[0] = (((output_array[0] + X[0])/totalHabitat) * 4.5/100) -X[0]
        output_array[4] = (((output_array[4] + X[4])/totalHabitat) * 4.5/100) -X[4]
        output_array[5] = (((output_array[5] + X[5])/totalHabitat) * 4.5/100) -X[5]
    # otherwise return zero
    else:
        output_array[0] = 0
        output_array[4] = 0
        output_array[5] = 0
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
    # keep habitat types scaled to the total size (4.5)
    sumHabitat = X0[0] +  X0[4] + X0[5]
    X0[0] = (X0[0]/sumHabitat) * 4.5/100
    X0[4] = (X0[4]/sumHabitat) * 4.5/100
    X0[5] = (X0[5]/sumHabitat) * 4.5/100
    # growth rates
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
    print(y[49:50:,]*100)
    result = (((y[49:50:, 0]-3.4/100)**2) +  ((y[49:50:, 1]-1.2/100)**2) + ((y[49:50:, 2]-1130.5/100)**2) + ((y[49:50:, 3]-6.7/100)**2)  + ((y[49:50:, 4]-.6/100)**2) + ((y[49:50:, 5]-.5/100)**2))
    print(result)
    return (result)

# order of outputs   
# ['arableGrass',    'fox'                  'rabbits'         'roeDeer',     'thornyScrub',            'woodland'])
#   3.3-3.7 (3.4)    0.22 to 2.2 (1.2)  1-2260 (1130.5)      4.9-9(6.7)     0.05 to 0.89 (0.6)      0.2-0.8 (0.5)

# here are my guesses to start at
X0guess = [0.039, 0.012, 9.8, 0.03, 0.0019, 0.0046]
growthGuess = [2.1, 0.003, 0.004, 0.003, 2.7, 2.8]
interactionGuess = [
                    -0.76,0.57,0.5,
                    -0.55,-0.25,
                    -0.78,0.89,-0.26,-0.98,-0.37,
                    -0.55,-0.43,-0.09,-0.7,
                    -0.05,0.04,0.93,-0.85,0.9,
                    -0.55,0.89,0.87,-0.31,-0.15
]

combined = X0guess + growthGuess + interactionGuess
guess = np.array(combined)


X0bds = ((3.6/100,4.2/100),(0.2/100,2.2/100),(1/100,2260/100),(1.3/100,5.3/100),(0.1/100,0.5/100),(0.2/100,0.7/100))
growthbds = ((0,10),(0,10),(0,10),(0,10),(0,10),(0,10))
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
optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-12)
print(optimization)



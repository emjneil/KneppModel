# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# # download packages
# from scipy import integrate
# from scipy.integrate import solve_ivp
# from scipy import optimize
# from scipy.optimize import differential_evolution
# import pylab as p
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import itertools as IT
# import string
# from hyperopt import hp, fmin, tpe, space_eval


# # from GPyOpt.methods import BayesianOptimization
# from skopt import gp_minimize


# # # # # # --------- MINIMIZATION ----------- # # # # # # # 

# # interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
# # # store species in a list
# # species = list(interactionMatrix_csv.columns.values)

# species = ['roeDeer','rabbits','fox','songbirdsWaterfowl','raptors','reptilesAmphibians','arableGrass','woodland','thornyScrub','wetland']

# def ecoNetwork(t, X, interaction_strength_chunk, rowContents_growth):
#     # define new array to return
#     output_array = []
#     # loop through all the species, apply generalized Lotka-Volterra equation
#     for outer_index, outer_species in enumerate(species): 
#         # grab one set of growth rates at a time
#         amount = rowContents_growth[outer_species] * X[outer_index]
#         for inner_index, inner_species in enumerate(species):
#             # grab one set of interaction matrices at a time
#             amount += interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index]
#         # append values to output_array
#         output_array.append(amount)
#     # return array
#     return output_array
# # Filter justifications
#     # roe deer start out 2.25 to 112.46; filter is 2.25 to 449.8. Average = 226.0
#     # arableGrass start out 338 to 401; filter is to 267 to 401; average = 334
#     # woodland start out 22.3 to 62.3; filter is to 53.4 to 97.9; average = 75.7
#     # thornyScrub start out 4.5 to 44.5; filter is 4.9 to 89; average = 46.95

# def objectiveFunction(x): 
#     X0 = x[0:13] 
#     growthRate = x[13:26]
#     growthRate = np.asarray(growthRate)
#     rowContents_growth = pd.DataFrame(data=growthRate.reshape(1,13),columns=species)
#     interaction_strength = x[26:195]
#     interaction_strength_chunk = pd.DataFrame(data=interaction_strength.reshape(13,13),index = species, columns=species)
#     t = np.linspace(0, 10, 50)
#     results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
#     # reshape the outputs
#     y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
#     # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
#     print(y[49:50:,]*200)
#     result = (((y[49:50:, 1]-(226/200))**2) + ((y[49:50:, 5]-(49.885/200))**2) + ((y[49:50:, 4]-(227.35/200))**2) + ((y[49:50:, 7]-(4.32/200))**2) + ((y[49:50:, 9]-(334/200))**2) + ((y[49:50:, 10]-(75.7/200))**2) + ((y[49:50:, 11]-(46.95/200))**2))
#     print(result)
#     return (result)


# # large herb, roe deer, tamworth, beaver, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland
# # here are my guesses to start at
# X0guess = [0,0.19,0,0,1.6,0.31,0.07,0.0006,0.0017,1.82,0.12,0.06,0.11]
# # normalize X01
# # X0guess = [x / 200 for x in X0guess_init]
# growthGuess = [0,0.06,0,0,0.02,0.04,0.14,0.06,0.65,0.58,0.69,0.69,0]
# interactionGuess = [0,0,0,0,0,0,0,0,0,-0.33,-0.6,-0.05,0,
#                     0,0,0,0,0,0,0,0,0,-0.17,-0.84,-0.5,0,
#                     0,0,0,0,0,0,0,0,-0.82,-0.07,-0.45,-0.2,0,
#                     0,0,0,0,0,0,0,0,0,-0.86,-0.1,0,0.68,
#                     0,0,0,0,0,0.08,0,0.74,0,-0.66,-0.89,-0.33,0, 
#                     0,0,0,0,-0.56,0,-0.82,0,-0.90,0,0,0,0,
#                     0,0,0,0,0,0.87,0,0.82,0,0,0,0,0,
#                     0,0,0,0,-0.58,0,-0.57,0,-0.6,0,0,0,0,
#                     0,0,0.25,0,0,0.89,0,0.1,0,0,0,0,0,
#                     0.19,0.00005,0.39,0.13,0.07,0,0,0,0,0,0,0,0,
#                     0.23,0.42,0.98,0.76,0.15,0,0,0,0,-0.08,0,-0.52,0,
#                     0.42,0.62,0.22,0,0.54,0,0.05,0,0.66,-0.89,0.13,0,0,
#                     0.99,0.69,0.63,0.04,0,0,0.24,0.15,0.08,-0.02,-0.26,-0.8,0
# ]

# combined = X0guess + growthGuess + interactionGuess
# guess = np.array(combined)

# # large herb, roe deer, tamworth, beaver, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland

# X0bds = ((0,0),(2.25/200,112.46/200),(0,0),(0,0),(2.7/200,452/200),(1.12/200,97.53/200),(0.15/200,27/200),(0.11/200,8.42/200),(0.311/200,40.4/200),(338/200,401/200),(22.3/200,62.3/200),(4.5/200,44.5/200),(15/200,24/200))
# growthbds = ((0,0),(0,1),(0,0),(0,0),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,0))
# interactionbds = ((0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
#                     (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
#                     (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(-1,0),(0,0),
#                     (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0), (0,0),(0,1),
#                     (0,0),(0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
#                     (0,0),(0,0),(0,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(0,0),(0,0),(0,0),
#                     (0,0),(0,0),(0,0),(0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),
#                     (0,0),(0,0),(0,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(0,0),(0,0),(0,0),
#                     (0,0),(0,0),(0,1),(0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),
#                     (0,1),(0,1),(0,1),(0,1),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),
#                     (0,1),(0,1),(0,1),(0,1),(0,1),(0,0),(0,0),(0,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),
#                     (0,1),(0,1),(0,1),(0,0),(0,1),(0,0),(0,1),(0,0),(0,1),(-1,0),(0,1),(0,0),(0,0),
#                     (0,1),(0,1),(0,1),(0,1),(0,0),(0,0),(0,1),(0,1),(0,1),(-1,0),(-1,0),(-1,0),(0,0))


# # combine them into one dataframe
# bds = X0bds + growthbds + interactionbds

# # optimization = gp_minimize(objectiveFunction, dimensions = bds, x0 = guess2)

# optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'Powell', options ={'maxiter': 10000}, tol=1e-3)
# print(optimization)







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

species = ['roeDeer','rabbits','fox','songbirdsWaterfowl','arableGrass','woodland','thornyScrub']
# species = ['roeDeer','rabbits','fox','songbirdsWaterfowl','raptors','reptilesAmphibians','arableGrass','woodland','thornyScrub','wetland']

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
    X0 = x[0:10] 
    growthRate = x[10:20]
    growthRate = np.asarray(growthRate)
    rowContents_growth = pd.DataFrame(data=growthRate.reshape(1,10),columns=species)
    interaction_strength = x[20:130]
    interaction_strength_chunk = pd.DataFrame(data=interaction_strength.reshape(10,10),index = species, columns=species)
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    print(y[49:50:,]*200)
    result = (
        ((y[49:50:, 0]-(226/200))**2) + ((y[49:50:, 1]-(227/200))**2) + ((y[49:50:, 2]-(37.1/200))**2) + ((y[49:50:, 3]-(13.63/200))**2) + ((y[49:50:, 4]-(4.32/200))**2) +  ((y[49:50:, 5]-(20.4/200))**2) + ((y[49:50:, 6]-(334/200))**2) + ((y[49:50:, 7]-(75.7/200))**2) + ((y[49:50:, 8]-(46.95/200))**2))
    # # multiply it by how many filters were passed
    # all_filters = [2.25 <= (y[49:50:, 0]*200) <= 449.8, 2.7 <= (y[49:50:, 1]*200) <= 452, 0.34 <= (y[49:50:, 5]*200) <= 40.4, 4.7 <= (y[49:50:, 2]*200) <= 69.5, 0.26 <= (y[49:50:, 3]*200) <= 27.0,  0.11 <= (y[49:50:, 4]*200) <= 8.42, 267 <= (y[49:50:, 6]*200) <= 401, 53.4 <= (y[49:50:, 7]*200) <= 97.9, 4.9 <= (y[49:50:, 8]*200) <= 89]
    # result2 = sum(all_filters)
    # # return total number of filters minus filters passed
    # result3 = 9-result2
    # result = result3 * result1
    print(result)
    return (result)

   # arableGrass,    fox,    rabbits,          roeDeer,      songbirdsWaterfowl, thornyScrub, woodland
# 267-401      4.7-69.5    2.7-452        2.25-449.8      0.26-27               4.9-89     53.4-97.9


# arableGrass,    fox,    rabbits,   raptors,          reptilesAmphibians, roeDeer,      songbirdsWaterfowl, thornyScrub, wetland, woodland
# 267-401      4.7-69.5    2.7-452   0.11 to 8.42          4.32-40.4        2.25-449.8      0.26-27               4.9-89            53.4-97.9


# roe deer, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland 2 5 6 7
# here are my guesses to start at
X0guess = [0.12,0.3,0.06,0.03,0.004,0.02,1.99,0.26,0.20,0.08]
# normalize X01
# X0guess = [x / 200 for x in X0guess_init]
growthGuess = [0.001,0.15,0.02,0.05,0.23,0.09,0.68,0.25,0.02,0]
interactionGuess = [
                    0,0,0,0,0,0,-0.78,-0.82,-0.47,0,
                    0,0,0.11,0,0.18,0,-0.12,-0.1,-0.6,0, 
                    0,-0.85,0,-0.08,0,-0.03,0,0,0,0,
                    0,0,0.77,0,0.11,0,0,0,0,0,
                    0,-0.85,0,-0.08,0,-0.97,0,0,0,0,
                    0,0,0.55,0,0.78,0,0,0,0,0,
                    0.002,0.02,0,0,0,0,0,0,0,0,
                    0.9,0.2,0,0.08,0.02,0,-0.76,0,-0.7,0,
                    0.06,0.66,0,0.38,0,0.07,-0.98,0.48,0,0,
                    0.33,0,0,0.49,0.95,0.79,-0.17,-0.42,-0.29,0
]

combined = X0guess + growthGuess + interactionGuess
guess = np.array(combined)

# roe deer, rabbit, fox, songbird, raptor, reptile, arable, wood, thorny, wetland

X0bds = ((2.25/200,112.46/200),(2.7/200,452/200),(1.12/200,97.53/200),(0.15/200,27/200),(0.11/200,8.42/200),(0.34/200,40.4/200),(338/200,401/200),(22.3/200,62.3/200),(4.5/200,44.5/200),(15/200,24/200))
growthbds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,0))
interactionbds = (
                    (0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
                    (0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(-1,0),(-1,0),(-1,0),(0,0),
                    (0,0),(-1,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(0,0),(0,0),(0,0),
                    (0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,0),(-1,0),(0,0),(-1,0),(0,0),(-1,0),(0,0),(0,0),(0,0),(0,0),
                    (0,0),(0,0),(0,1),(0,0),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,1),(0,1),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),
                    (0,1),(0,1),(0,0),(0,1),(0,1),(0,0),(-1,0),(0,0),(-1,0),(0,0),
                    (0,1),(0,1),(0,0),(0,1),(0,0),(0,1),(-1,0),(0,1),(0,0),(0,0),
                    (0,1),(0,0),(0,0),(0,1),(0,1),(0,1),(-1,0),(-1,0),(-1,0),(0,0))

# combine them into one dataframe
bds = X0bds + growthbds + interactionbds

# optimization = gp_minimize(objectiveFunction, dimensions = bds, x0 = guess2)

#L-BFGS-B, Powell, TNC, SLSQP, can have bounds
optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'SLSQP', options ={'maxiter': 10000}, tol=1e-3)
print(optimization)

# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
# import pylab as p
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import itertools as IT
# import string
from skopt import gp_minimize
import numpy.matlib


# # # # # --------- MINIMIZATION ----------- # # # # # # # 

species = ['arableGrass','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    X[X>1e5] = 1e5
    return X * (r + np.matmul(A, X))


def calcJacobian(A, r, n):
    # make an empty array to fill (with diagonals = 1, zeros elsewhere since we want eigenalue)
    i_matrix = np.eye(len(n))
    # put n into an array to multiply by A
    n_array = np.matlib.repmat(n, 1, len(n))
    n_array = np.reshape (n_array, (7,7))
    # calculate
    J = i_matrix * r + A * n_array + i_matrix * np.matmul(A, n)
    return J


def calcStability(A, r, n):
    J = calcJacobian(A, r, n)
    ev = np.real(np.linalg.eig(J)[0])
    max_eig = np.max(ev)
    if max_eig < 0:
        return True
    else:
        return False


def objectiveFunction(x): 
    r =  x[0:7]
    # insert interaction matrices of 0
    x = np.insert(x,9,0)
    x = np.insert(x,16,0)
    x = np.insert(x,17,0)
    x = np.insert(x,18,0)
    x = np.insert(x,29,0)
    x = np.insert(x,30,0)
    x = np.insert(x,32,0)
    x = np.insert(x,36,0)
    x = np.insert(x,37,0)
    x = np.insert(x,38,0)
    # x = np.insert(x,42,0)
    x = np.insert(x,44,0)
    # x = np.insert(x,49,0)
    x = np.insert(x,51,0)
    # define X0, growthRate, interactionMatrix
    X0 = [1,0,1,1,0,1,1]
    # growth rates
    interaction_strength = x[7:56]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(7,7),index = species, columns=species)
    all_times = []
    A = interaction_strength.to_numpy()
    # use p inverse (instead of inv) to avoid "singular matrix" errors
    ia = np.linalg.pinv(A)
    # check viability of the parameter set. n is the equilibrium state; calc as inverse of -A*r
    n = -np.matmul(ia, r)
    # if all the values of n are above zero at equilibrium, check the stability 
    if np.all(n > 0):
        isStable = calcStability(A, r, n)
        # if the parameter set is viable (stable & all n > 0 at equilibrium); do the calculation
        if isStable == True:
            # ODE1
            t_init = np.linspace(0, 4, 20)
            results = solve_ivp(ecoNetwork, (0, 4), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
            # reshape the outputs
            y = (np.vstack(np.hsplit(results.y.reshape(len(species), 20).transpose(),1)))
            y = pd.DataFrame(data=y, columns=species)
            all_times = np.append(all_times, results.t)
            y['time'] = all_times
            # ODE 2
            last_results = y.loc[y['time'] == 4]
            last_results = last_results.drop('time', axis=1)
            last_results = last_results.values.flatten()
            # set large herbivore numbers
            last_results[1] = 1
            last_results[4] = 1
            # next ODE
            t = np.linspace(4, 5, 5)
            second_ABC = solve_ivp(ecoNetwork, (4,5), last_results,  t_eval = t, args=(A, r), method = 'RK23')   
            # take those values and re-run for another year, adding forcings
            starting_2010 = second_ABC.y[0:7, 4:5]
            starting_values_2010 = starting_2010.flatten()
            starting_values_2010[1] = 2.0
            starting_values_2010[4] = 0.5
            t_1 = np.linspace(5, 6, 5)
            # run the model for another year 2010-2011
            third_ABC = solve_ivp(ecoNetwork, (5,6), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2011 = third_ABC.y[0:7, 4:5]
            starting_values_2011 = starting_2011.flatten()
            starting_values_2011[1] = 1.1
            starting_values_2011[4] = 1.3
            t_2 = np.linspace(6, 7, 5)
            # run the model for 2011-2012
            fourth_ABC = solve_ivp(ecoNetwork, (6,7), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2012 = fourth_ABC.y[0:7, 4:5]
            starting_values_2012 = starting_2012.flatten()
            starting_values_2012[1] = 1.1
            starting_values_2012[4] = 1.5
            t_3 = np.linspace(7, 8, 5)
            # run the model for 2012-2013
            fifth_ABC = solve_ivp(ecoNetwork, (7,8), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2013 = fifth_ABC.y[0:7, 4:5]
            starting_values_2013 = starting_2013.flatten()
            starting_values_2013[1] = 1.8
            starting_values_2013[4] = 0.18
            t_4 = np.linspace(8, 9, 5)
            # run the model for 2011-2012
            sixth_ABC = solve_ivp(ecoNetwork, (8,9), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2014 = sixth_ABC.y[0:7, 4:5]
            starting_values_2014 = starting_2014.flatten()
            starting_values_2014[1] = 0.6
            starting_values_2014[4] = 3
            t_5 = np.linspace(9,10, 5)
            # run the model for 2011-2012
            seventh_ABC = solve_ivp(ecoNetwork, (9,10), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2015 = seventh_ABC.y[0:7, 4:5]
            starting_values_2015 = starting_2015.flatten()
            starting_values_2015[1] = 1.2
            starting_values_2015[4] = 0.5
            t_6 = np.linspace(10,11, 5)
            # run the model for 2011-2012
            eighth_ABC = solve_ivp(ecoNetwork, (10,11), starting_values_2015,  t_eval = t_6, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2016 = eighth_ABC.y[0:7, 4:5]
            starting_values_2016 = starting_2016.flatten()
            starting_values_2016[1] = 1.21
            starting_values_2016[4] = 0.5
            t_7 = np.linspace(11,12, 5)
            # run the model for 2011-2012
            ninth_ABC = solve_ivp(ecoNetwork, (11,12), starting_values_2016,  t_eval = t_7, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2017 = ninth_ABC.y[0:7, 4:5]
            starting_values_2017 = starting_2017.flatten()
            starting_values_2017[1] = np.random.uniform(low=0.56,high=2.0)
            starting_values_2017[4] = np.random.uniform(low=0.18,high=3)
            t_8 = np.linspace(12,13, 5)
            # run the model for 2011-2012
            tenth_ABC = solve_ivp(ecoNetwork, (12,13), starting_values_2017,  t_eval = t_8, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2018 = tenth_ABC.y[0:7, 4:5]
            starting_values_2018 = starting_2018.flatten()
            starting_values_2018[1] = np.random.uniform(low=0.56,high=2.0)
            starting_values_2018[4] = np.random.uniform(low=0.18,high=3)
            t_9 = np.linspace(13,14, 5)
            # run the model for 2011-2012
            eleventh_ABC = solve_ivp(ecoNetwork, (13,14), starting_values_2018,  t_eval = t_9, args=(A, r), method = 'RK23')
            # concatenate & append all the runs
            combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y))
            combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, eighth_ABC.t, ninth_ABC.t, tenth_ABC.t, eleventh_ABC.t))
            # reshape the outputs
            y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 50).transpose(),1)))
            y_2 = pd.DataFrame(data=y_2, columns=species)
            y_2['time'] = combined_times
            # choose the final year (we want to compare the final year to the middle of the filters)
            last_year_1 = y.loc[y['time'] == 4]
            last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
            last_year_2 = y_2.loc[y_2['time'] == 14]
            last_year_2 = last_year_2.drop('time', axis=1).values.flatten()
            # print the outputs
            print(last_year_2)
            result = (((last_year_1[0]-0.86)**2) +  ((last_year_1[2]-1.4)**2) + ((last_year_1[3]-2.2)**2) + ((last_year_1[5]-10)**2) + ((last_year_1[6]-0.91)**2) + ((last_year_2[0]-0.72)**2) +  ((last_year_2[2]-2)**2) + ((last_year_2[3]-4.1)**2) + ((last_year_2[5]-28.8)**2) + ((last_year_2[6]-0.91)**2))
            print("r",result)  
            return (result)
        # otherwise return some high number (to stop minimizer errors)
        else:
            return 1e5
    # otherwise return some high number
    else:
        return 1e5

# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.86            1.4        2.2              10              0.91

# ['arableGrass',  largeHerb, orgCarb  'roeDeer',tamworthPig,  'thornyScrub','woodland'])
#   0.72                       2          4.1                     28.8          0.91

growth_bds = ((0,1),(0.14,0.28),(0,1),(0.28,0.34),(0.01,0.3),(0.9,1),(0,0.014))
# roe deer = 0.31 growth rate (+-50%)
# carbon = forced to be low (0.01 max), otherwise minimizer makes it high & model predicts it'll be HIGHER without herbivore reintroductions
# wild horses = 0.18-0.25; red deer = 0.15 (+-10% min/max)
# feral hog  = 0.18-0.21; boar in England = 0.016-0.27 (+-10%min/max)
# woodland = lots of studies (see code sheet); min to max (0.004-0.014); made it 0.1
interactionbds = (
                    (-1,0),(-0.0001,0),(-0.0001,0),(-0.0001,0),(-0.01,0),(-0.01,0),
                    (0.3,0.4),(-0.8,-0.7),(0.01,0.03),(0.4,0.45),
                    (0,0.5),(0,0.5),(-1,0),(0,0.5),(0,0.5),(0,0.5),(0,0.5),
                    (0.5,0.6),(-0.8,-0.7),(0.01,0.1),(0.4,0.5),
                    (0.5,0.6),(-0.5,-0.4),(0.01,0.1),(0.4,0.5),
                    (-0.01,0),(-0.1,0),(-0.1,0),(-0.1,0),(-0.01,0),(-0.3,-0.2),
                    (-0.001,0),(-0.001,0),(-0.001,0),(-0.001,0),(0,0.01),(-1,0)
)

# combine them into one dataframe
bds =  growth_bds + interactionbds

# growthGuess = [0.94, 0.28, 0.61, 0.39, 0.17, 0.83, 0.018]
# interactionGuess = [
#                     -0.85,-0.07,-0.02,-0.04,-0.003,-0.05,
#                     0.38,-0.99,0.03,0.43,
#                     0.37,0.15,-0.85,0.05,0.04,0.004,0.32,
#                     0.59,-0.73,0.08,0.16,
#                     0.62,-0.55,0.04,0.21,
#                     -0.09,-0.06,-0.02,-0.003,-0.02,-0.08,
#                     -0.005,-0.23,-0.03,-0.36,0.06,-0.33
# ]

# combined = growthGuess + interactionGuess
# guess = np.array(combined)

# optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-6)
optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 1000)
# optimization = optimize.fmin_l_bfgs_b(objectiveFunction, x0 = guess, bounds = bds, approx_grad = True, epsilon = 1e-04)

# save to csv
print(optimization, file=open("final_optimizationOutput.txt", "w"))
print(optimization)

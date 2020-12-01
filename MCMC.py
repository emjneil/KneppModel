# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
from scipy import integrate
from scipy.integrate import solve_ivp
import pylab as p
import pandas as pd
import numpy as np
import itertools as IT
import string
from hyperopt import hp, fmin, tpe, space_eval
from skopt import gp_minimize
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.stats as st
import pyabc
from pyabc import (ABCSMC,
                   RV, Distribution,
                   MedianEpsilon,
                   LocalTransition)
from pyabc.visualization import plot_kde_2d, plot_data_callback



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


# ['arableGrass',  largeHerb, orgCarb  'roeDeer',tamworthPig,  'thornyScrub','woodland'])
#   0.86                       1.4        2.2                     10             0.91
#   0.72                       2          4.1                     28.8           0.91


with pm.Model() as ecosystemModel:
    # create the parameters
    # grow_grass = pm.Bound(pm.Uniform, lower= 0, upper=1.0)('grow_grass')
    grow_grass = pm.Uniform('grow_grass', lower= 0, upper=1.0)
    grow_largeHerb = pm.Uniform('grow_largeHerb', lower=0.08, upper=0.38)
    grow_orgCarb = pm.Uniform('grow_orgCarb', lower=0, upper=1.0)
    grow_roe = pm.Uniform('grow_roe', lower=0.16, upper=0.47)
    grow_tamPig = pm.Uniform('grow_tamPig', lower=0.006, upper=0.41)
    grow_thornyScrub = pm.Uniform('grow_thornyScrub', lower=0, upper=1.0)
    grow_wood = pm.Uniform('grow_wood',lower=0.002, upper=0.02)
    growth_bds =  grow_grass + grow_largeHerb + grow_orgCarb + grow_roe + grow_tamPig + grow_thornyScrub + grow_wood
    print(grow_grass, growth_bds)
    # interaction parameters
    x1 = pm.Uniform('x1', lower=-1, upper=0)
    x2 = pm.Uniform("x2", lower=-1, upper=0)
    x3 = pm.Uniform("x3", lower=-1, upper=0)
    x4 = pm.Uniform("x4",lower=-1, upper=0)
    x5 = pm.Uniform("x5", lower=-1, upper=0)
    x6 = pm.Uniform("x6",lower=-1, upper=0)
    x7 = pm.Uniform("x7", lower=0, upper=1)
    x8 = pm.Uniform("x8", lower=-1, upper=0)
    x9 = pm.Uniform("x9", lower=0, upper=1)
    x10 = pm.Uniform("x10", lower=0, upper=1)
    x11 = pm.Uniform("x11", lower=0, upper=1)
    x12 = pm.Uniform("x12", lower=0, upper=1)
    x13 = pm.Uniform("x13", lower=-1, upper=0)
    x14 = pm.Uniform("x14", lower=0, upper=1)
    x15 = pm.Uniform("x15", lower=0, upper=1)
    x16 = pm.Uniform("x16", lower=0, upper=1)
    x17 = pm.Uniform("x17", lower=0, upper=1)
    x18 = pm.Uniform("x18", lower=0, upper=1)
    x19 = pm.Uniform("x19", lower=-1, upper=0)
    x20 = pm.Uniform("x20", lower=0, upper=1)
    x21 = pm.Uniform("x21",lower=0, upper=1)
    x22 = pm.Uniform("x22", lower=0, upper=1)
    x23 = pm.Uniform("x23", lower=-1, upper=1)
    x24 = pm.Uniform("x24", lower=0, upper=1)
    x25 = pm.Uniform("x25", lower=0, upper=1)
    x26 = pm.Uniform("x26", lower=-1, upper=1)
    x27 = pm.Uniform("x27", lower=-1, upper=0)
    x28 = pm.Uniform("x28", lower=-1, upper=0)
    x29 = pm.Uniform("x29", lower=-1, upper=0)
    x30 = pm.Uniform('x30', lower=-1, upper=0)
    x31 = pm.Uniform("x31", lower=-1, upper=0)
    x32 = pm.Uniform("x32", lower=-1, upper=1)
    x33 = pm.Uniform("x33", lower=-1, upper=0)
    x34 = pm.Uniform("x34", lower=-1, upper=0)
    x35 = pm.Uniform("x35", lower=-1, upper=0)
    x36 = pm.Uniform("x36", lower=-1, upper=0)
    x37 = pm.Uniform("x37", lower=-1, upper=0)
    interactionbds = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20+x21+x22+x23+x24+x25+x26+x27+x28+x29+x30+x31+x32+x33+x34+x35+x36+x37
    # combine dataframes
    x = growth_bds + interactionbds
    print(x)

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
    x = np.insert(x,44,0)
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
            t_init = np.linspace(0, 4, 40)
            results = solve_ivp(ecoNetwork, (0, 4), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
            # reshape the outputs
            y = (np.vstack(np.hsplit(results.y.reshape(len(species), 40).transpose(),1)))
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
            result = last_year_1 + last_year_2
            print("r",result) 
            observed = pm.Normal("observed", mu = result, value=np.array([0.86,0,1.4,2.2,0,10,0.91,0.72,1.28,2,4.1,4.1,28.8,0.91])) 
            
    # M = pm.Model(objectiveFunction)
    samples = pm.sample(10000)
    pm.traceplot(samples, ['growth_bds', 'interactionbds'])
    pm.traceplot(samples)
    print(samples)

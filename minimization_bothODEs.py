4.# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
from skopt import gp_minimize
import numpy.matlib


# # # # # --------- MINIMIZATION ----------- # # # # # # #

species = ['arableGrass','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
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
    # only primary producers should have intrinsic growth rate
    x = np.insert(x,1,0)
    x = np.insert(x,2,0)
    x = np.insert(x,3,0)
    x = np.insert(x,4,0)
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
    x = np.insert(x,42,0)
    x = np.insert(x,44,0)
    x = np.insert(x,49,0)
    x = np.insert(x,51,0)
    # define X0
    X0 = [1,0,1,1,0,1,1]
    # define interaction strength
    interaction_strength = x[7:56]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(7,7),index = species, columns=species)
    A = interaction_strength.to_numpy()
    # make a list to hold times in
    all_times = []
    # check stability 
    ia = np.linalg.pinv(A)
    # n is the equilibrium state; calc as inverse of -A*r
    n = -np.matmul(ia, r)

    # if all the values of n are above zero at equilibrium, check the stability
    if np.all(n > 0):
        isStable = calcStability(A, r, n)
        # if the parameter set is viable (stable & all n > 0 at equilibrium); do the 2005-2009 ODE
        if isStable == True:
            t_init = np.linspace(0, 4, 12)
            results = solve_ivp(ecoNetwork, (0, 4), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
            # reshape the outputs
            y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
            y = pd.DataFrame(data=y, columns=species)
            all_times = np.append(all_times, results.t)
            y['time'] = all_times
            # Calculate 2009-2010
            last_results = y.loc[y['time'] == 4]
            last_results = last_results.drop('time', axis=1)
            last_results = last_results.values.flatten()
            last_results[1] = 1
            last_results[4] = 1
            t = np.linspace(4, 5, 10)
            second_ABC = solve_ivp(ecoNetwork, (4,5), last_results,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2010-2011) adding forcings
            starting_2010 = second_ABC.y[0:7, 9:10].flatten()
            starting_values_2010 = starting_2010
            starting_values_2010[1] = 1.9
            starting_values_2010[4] = 0.9
            t_1 = np.linspace(5, 6, 10)
            third_ABC = solve_ivp(ecoNetwork, (5,6), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2011-2012), adding forcings
            starting_2011 = third_ABC.y[0:7, 9:10].flatten()
            starting_values_2011 = starting_2011
            starting_values_2011[1] = 2.8
            starting_values_2011[4] = 1.1
            t_2 = np.linspace(6, 7, 5)
            fourth_ABC = solve_ivp(ecoNetwork, (6,7), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2012-2013), adding forcings
            starting_2012 = fourth_ABC.y[0:7, 4:5].flatten()
            starting_values_2012 = starting_2012
            starting_values_2012[1] = 3.2
            starting_values_2012[4] = 1.7
            t_3 = np.linspace(7, 8, 5)
            fifth_ABC = solve_ivp(ecoNetwork, (7,8), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2013-2014), adding forcings
            starting_2013 = fifth_ABC.y[0:7, 4:5].flatten()
            starting_values_2013 = starting_2013
            starting_values_2013[1] = 5.1
            starting_values_2013[4] = 0.3
            t_4 = np.linspace(8, 9, 5)
            sixth_ABC = solve_ivp(ecoNetwork, (8,9), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2014-2015), adding forcings
            starting_2014 = sixth_ABC.y[0:7, 4:5].flatten()
            starting_values_2014 = starting_2014
            starting_values_2014[1] = 3.1
            starting_values_2014[4] = 0.9
            t_5 = np.linspace(9,10, 5)
            seventh_ABC = solve_ivp(ecoNetwork, (9,10), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23')

            # take those values and re-run for another year (2015-2016), adding forcings
            starting_2015 = seventh_ABC.y[0:7, 4:5].flatten()
            starting_values_2015 = starting_2015
            starting_values_2015[1] = 2.4
            starting_values_2015[4] = 0.4
            t_6 = np.linspace(10,11, 5)
            eighth_ABC = solve_ivp(ecoNetwork, (10,11), starting_values_2015,  t_eval = t_6, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2016-2017), adding forcings
            starting_2016 = eighth_ABC.y[0:7, 4:5].flatten()
            starting_values_2016 = starting_2016
            starting_values_2016[1] = 3.1
            starting_values_2016[4] = 0.4
            t_7 = np.linspace(11,12, 5)
            ninth_ABC = solve_ivp(ecoNetwork, (11,12), starting_values_2016,  t_eval = t_7, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2017-2018), adding forcings
            starting_2017 = ninth_ABC.y[0:7, 4:5].flatten()
            starting_values_2017 = starting_2017
            starting_values_2017[1] = 2.6
            starting_values_2017[4] = 0.6
            t_8 = np.linspace(12,13, 5)
            tenth_ABC = solve_ivp(ecoNetwork, (12,13), starting_values_2017,  t_eval = t_8, args=(A, r), method = 'RK23')
            # take those values and re-run for another year (2018-2019), adding forcings
            starting_2018 = tenth_ABC.y[0:7, 4:5].flatten()
            starting_values_2018 = starting_2018
            starting_values_2018[1] = 4.2
            starting_values_2018[4] = 0.5
            t_9 = np.linspace(13,14, 5)
            eleventh_ABC = solve_ivp(ecoNetwork, (13,14), starting_values_2018,  t_eval = t_9, args=(A, r), method = 'RK23')
            # run for 2019-2020
            starting_2019 = eleventh_ABC.y[0:7, 4:5].flatten()
            starting_values_2019 = starting_2019
            starting_values_2019[1] = 4.1
            starting_values_2019[4] = 0.4
            t_10 = np.linspace(14,15,5)
            twelfth_ABC = solve_ivp(ecoNetwork, (14,15), starting_values_2019,  t_eval = t_10, args=(A, r), method = 'RK23')
            # run 2020-2021
            starting_2020 = twelfth_ABC.y[0:7, 4:5].flatten()
            # concatenate & append all the runs
            combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y, twelfth_ABC.y))
            combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, eighth_ABC.t, ninth_ABC.t, tenth_ABC.t, eleventh_ABC.t, twelfth_ABC.t))
            # reshape the outputs
            y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 65).transpose(),1)))
            y_2 = pd.DataFrame(data=y_2, columns=species)
            y_2['time'] = combined_times
            # choose the final year (we want to compare the final year to the middle of the filters)
            last_year_1 = y.loc[y['time'] == 4]
            last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
            last_year_2 = y_2.loc[y_2['time'] == 15]
            last_year_2 = last_year_2.drop('time', axis=1).values.flatten()
            # print the outputs
            print(last_year_2)
            result = ( 

                # ((starting_2010[1]-1.2)**2) + ((starting_2010[4]-1.8)**2) +

                # don't have middle of filters for 2011-2014 due to 1) lack of data; 2) species were supplemented (e.g. fallow & red deer added)
                # take the average of the calving years data for post-reintroductions, or the datapoint (where only 1 is available, i.e. 2010 and 2020)
                ((starting_2015[1]-3.4)**2) + ((starting_2015[4]-1)**2) +
                ((starting_2016[1]-3.8)**2) + ((starting_2016[4]-0.7)**2) +
                ((starting_2017[1]-4.1)**2) + ((starting_2017[4]-0.7)**2) +

                # took out 2018 bc their numbers (fallow deer) appear to have been supplemented
                # ((starting_2018[1]-5.4)**2) + ((starting_2018[4]-1)**2) +

                ((starting_2019[1]-5.5)**2) + ((starting_2019[4]-0.5)**2) +
                ((starting_2020[1]-5.0)**2) + ((starting_2020[4]-1)**2) +

                ((last_year_1[0]-0.97)**2) +  ((last_year_1[2]-1.4)**2) + ((last_year_1[3]-2.2)**2) + 
                ((last_year_1[5]-10)**2) + ((last_year_1[6]-1.2)**2) + ((last_year_2[0]-0.73)**2) +  
                ((last_year_2[2]-2)**2) + ((last_year_2[3]-4.2)**2) + ((last_year_2[5]-26.6)**2) + 
                ((last_year_2[6]-1.36)**2))

            print("r",result)
            return (result)
    # otherwise return some high number (to stop minimizer errors)
        else:
            return 1e4
    else:
        return 1e5

# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.97            1.4        2.2              10              1.2

# ['arableGrass',  largeHerb, orgCarb  'roeDeer',tamworthPig,  'thornyScrub','woodland'])
#   0.73                       2          4.2                    26.6          1.36

growth_bds = ((0.85,0.95),(0.64,0.67),(0.027,0.035))


interactionbds = (
                    (-0.78,-0.74),(-0.004,-0.002),(-0.002,-0.0005),(-0.008,-0.005),(-0.013,-0.01),(-0.016,-0.012),
                    (0,1),(-0.5,0),(0,0.25),(0,1),
                    (0.05,0.095),(0.0035,0.01),(-0.1,-0.09),(0.0015,0.004),(0.0025,0.007),(0.00015,0.00025),(0.05,0.095),
                    (0,1),(-0.5,0),(0,0.25),(0,1),
                    (0,1),(-0.5,0),(0,0.25),(0,1),
                    (-0.07,-0.03),(-0.04,-0.01),(-0.05,-0.02),(-0.0019,-0.0017),(-0.018,-0.01),
                    (-0.0005,-0.0003),(-0.0003,-0.0001),(-0.0004,-0.0002),(0.0001,0.00035),(-0.009,-0.007)
)

# combine them into one dataframe
bds =  growth_bds + interactionbds

optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 50)
# print(optimization, file=open("final_optimizationOutput.txt", "w"))
print(optimization)

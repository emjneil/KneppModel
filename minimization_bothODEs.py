# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
from skopt import gp_minimize
import numpy.matlib


# # # # # --------- MINIMIZATION ----------- # # # # # # #

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','organicCarbon','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']


def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))

# check for stability
def calcJacobian(A, r, n):
    # make an empty array to fill (with diagonals = 1, zeros elsewhere since we want eigenalue)
    i_matrix = np.eye(len(n))
    # put n into an array to multiply by A
    n_array = np.matlib.repmat(n, 1, len(n))
    n_array = np.reshape (n_array, (10,10))
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
    x = np.insert(x,0,0)
    x = np.insert(x,1,0)
    x = np.insert(x,3,0)
    x = np.insert(x,4,0)
    x = np.insert(x,5,0)
    x = np.insert(x,6,0)
    x = np.insert(x,7,0)
    r =  x[0:10]

    # insert interaction matrices of 0
    x = np.insert(x,10,-0.001)
    x = np.insert(x,11,0)
    x = np.insert(x,12,0)
    x = np.insert(x,13,0)
    x = np.insert(x,14,0)
    x = np.insert(x,15,0)
    x = np.insert(x,16,0)
    x = np.insert(x,17,0)
    x = np.insert(x,18,0)
    x = np.insert(x,19,0)
    x = np.insert(x,20,0)
    x = np.insert(x,23,0)
    x = np.insert(x,24,0)
    x = np.insert(x,25,0)
    x = np.insert(x,26,0)
    x = np.insert(x,27,0)
    x = np.insert(x,34,0)
    x = np.insert(x,40,0)
    x = np.insert(x,41,0)
    x = np.insert(x,44,0)
    x = np.insert(x,45,0)
    x = np.insert(x,46,0)
    x = np.insert(x,47,0)
    x = np.insert(x,60,0)
    x = np.insert(x,61,0)
    x = np.insert(x,63,0)
    x = np.insert(x,64,0)
    x = np.insert(x,66,0)
    x = np.insert(x,67,0)
    x = np.insert(x,70,0)
    x = np.insert(x,71,0)
    x = np.insert(x,73,0)
    x = np.insert(x,74,0)
    x = np.insert(x,75,0)
    x = np.insert(x,77,0)
    x = np.insert(x,80,0)
    x = np.insert(x,81,0)
    x = np.insert(x,83,0)
    x = np.insert(x,84,0)
    x = np.insert(x,85,0)
    x = np.insert(x,86,0)
    x = np.insert(x,92,0)
    x = np.insert(x,94,0)
    x = np.insert(x,102,0)
    x = np.insert(x,104,0)

    # define X0
    X0 = [0,0,1,0,1,0,1,0,1,1]
    # define interaction strength
    interaction_strength = x[10:110]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(10,10),index = species, columns=species)
    A = interaction_strength.to_numpy()
    # check viability of the parameter set (is it stable?)
    # ia = np.linalg.pinv(A)
    # # n is the equilibrium state; calc as inverse of -A*r
    # n = -np.matmul(ia, r)
    # isStable = calcStability(A, r, n)
    # if all the values of n are above zero at equilibrium, & if the parameter set is viable (stable & all n > 0 at equilibrium); do the calculation
    # if np.all(n > 0) &  isStable == True:
    # run 2005, 2006, 2007, 2008, 3 points per year
    all_ecoCheck = []
    t_eco = np.linspace(0, 1, 2)
    for i in range(len(species)):
        X0_ecoCheck = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        X0_ecoCheck[i] = 1
        ecoCheck_ABC = solve_ivp(ecoNetwork, (0, 1), X0_ecoCheck,  t_eval = t_eco, args=(A, r), method = 'RK23') 
        all_ecoCheck = np.append(all_ecoCheck, ecoCheck_ABC.y)
    all_ecoCheck_results = (np.vstack(np.hsplit(all_ecoCheck.reshape(len(species), 20).transpose(),1)))
    all_ecoCheck_results = pd.DataFrame(data=all_ecoCheck_results, columns=species)
    # ecological reality check: primary producers should not decline with no herbivores present
    if (all_ecoCheck_results.loc[5,'grasslandParkland'] >= 1) & (all_ecoCheck_results.loc[17,'thornyScrub'] >= 1) & (all_ecoCheck_results.loc[19,'woodland'] >= 1):
        
        all_times = []
        t_init = np.linspace(0, 3.95, 12)
        results = solve_ivp(ecoNetwork, (0, 3.95), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
        # reshape the outputs
        y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
        y = pd.DataFrame(data=y, columns=species)
        all_times = np.append(all_times, results.t)
        y['time'] = all_times
        last_results = y.loc[y['time'] == 3.95]
        last_results = last_results.drop('time', axis=1)
        last_results = last_results.values.flatten()
        # ponies, longhorn cattle, and tamworth pigs reintroduced in 2009
        starting_2009 = last_results.copy()
        starting_2009[0] = 1
        starting_2009[3] = 1
        starting_2009[7] = 1
        t_01 = np.linspace(4, 4.95, 3)
        second_ABC = solve_ivp(ecoNetwork, (4,4.95), starting_2009,  t_eval = t_01, args=(A, r), method = 'RK23')
        # 2010
        last_values_2009 = second_ABC.y[0:10, 2:3].flatten()
        starting_values_2010 = last_values_2009.copy()
        starting_values_2010[0] = 0.57
        starting_values_2010[1] = 1
        starting_values_2010[3] = 1.45
        starting_values_2010[7] = 0.85
        t_1 = np.linspace(5, 5.95, 3)
        third_ABC = solve_ivp(ecoNetwork, (5,5.95), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
        # 2011
        last_values_2010 = third_ABC.y[0:10, 2:3].flatten()
        starting_values_2011 = last_values_2010.copy()
        starting_values_2011[0] = 0.65
        starting_values_2011[1] = 1.93
        starting_values_2011[3] = 1.74
        starting_values_2011[7] = 1.1
        t_2 = np.linspace(6, 6.95, 3)
        fourth_ABC = solve_ivp(ecoNetwork, (6,6.95), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
        # 2012
        last_values_2011 = fourth_ABC.y[0:10, 2:3].flatten()
        starting_values_2012 = last_values_2011.copy()
        starting_values_2012[0] = 0.74
        starting_values_2012[1] = 2.38
        starting_values_2012[3] = 2.19
        starting_values_2012[7] = 1.65
        t_3 = np.linspace(7, 7.95, 3)
        fifth_ABC = solve_ivp(ecoNetwork, (7,7.95), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
        # 2013
        last_values_2012 = fifth_ABC.y[0:10, 2:3].flatten()
        starting_values_2013 = last_values_2012.copy()
        starting_values_2013[0] = 0.43
        starting_values_2013[1] = 2.38
        starting_values_2013[3] = 2.43
        # red deer reintroduced
        starting_values_2013[5] = 1
        starting_values_2013[7] = 0.3
        t_4 = np.linspace(8, 8.95, 3)
        sixth_ABC = solve_ivp(ecoNetwork, (8,8.95), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
        # 2014
        last_values_2013 = sixth_ABC.y[0:10, 2:3].flatten()
        starting_values_2014 = last_values_2013.copy()
        starting_values_2014[0] = 0.43
        starting_values_2014[1] = 2.38
        starting_values_2014[3] = 4.98
        starting_values_2014[5] = 1
        starting_values_2014[7] = 0.9
        t_5 = np.linspace(9, 9.95, 3)
        seventh_ABC = solve_ivp(ecoNetwork, (9,9.95), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23') 
        # 2015
        last_values_2014 = seventh_ABC.y[0:10, 2:3].flatten()
        starting_values_2015 = last_values_2014
        starting_values_2015[0] = 0.43
        starting_values_2015[1] = 2.38
        starting_values_2015[3] = 2.02
        starting_values_2015[5] = 1
        starting_values_2015[7] = 0.9
        t_2015 = np.linspace(10, 10.95, 3)
        ABC_2015 = solve_ivp(ecoNetwork, (10,10.95), starting_values_2015,  t_eval = t_2015, args=(A, r), method = 'RK23')
        # 2016
        last_values_2015 = ABC_2015.y[0:10, 2:3].flatten()
        starting_values_2016 = last_values_2015.copy()
        starting_values_2016[0] = 0.48
        starting_values_2016[1] = 3.33
        starting_values_2016[3] = 1.62
        starting_values_2016[5] = 2
        starting_values_2016[7] = 0.4
        t_2016 = np.linspace(11, 11.95, 3)
        ABC_2016 = solve_ivp(ecoNetwork, (11,11.95), starting_values_2016,  t_eval = t_2016, args=(A, r), method = 'RK23')       
        # 2017
        last_values_2016 = ABC_2016.y[0:10, 2:3].flatten()
        starting_values_2017 = last_values_2016.copy()
        starting_values_2017[0] = 0.43
        starting_values_2017[1] = 3.93
        starting_values_2017[3] = 1.49
        starting_values_2017[5] = 1.08
        starting_values_2017[7] = 0.35
        t_2017 = np.linspace(12, 12.95, 3)
        ABC_2017 = solve_ivp(ecoNetwork, (12,12.95), starting_values_2017,  t_eval = t_2017, args=(A, r), method = 'RK23')     
        # 2018
        last_values_2017 = ABC_2017.y[0:10, 2:3].flatten()
        starting_values_2018 = last_values_2017.copy()
        starting_values_2018[0] = 0.39
        starting_values_2018[1] = 5.98
        starting_values_2018[3] = 1.66
        starting_values_2018[5] = 1.85
        starting_values_2018[7] = 0.8
        t_2018 = np.linspace(13, 13.95, 3)
        ABC_2018 = solve_ivp(ecoNetwork, (13,13.95), starting_values_2018,  t_eval = t_2018, args=(A, r), method = 'RK23')     
        # 2019
        last_values_2018 = ABC_2018.y[0:10, 2:3].flatten()
        starting_values_2019 = last_values_2018.copy()
        starting_values_2019[0] = 0
        starting_values_2019[1] = 6.62
        starting_values_2019[3] = 1.64
        starting_values_2019[5] = 2.85
        starting_values_2019[7] = 0.45
        t_2019 = np.linspace(14, 14.95, 3)
        ABC_2019 = solve_ivp(ecoNetwork, (14,14.95), starting_values_2019,  t_eval = t_2019, args=(A, r), method = 'RK23')
        # 2020
        last_values_2019 = ABC_2019.y[0:10, 2:3].flatten()
        starting_values_2020 = last_values_2019.copy()
        starting_values_2020[0] = 0.65
        starting_values_2020[1] = 5.88
        starting_values_2020[3] = 1.53
        starting_values_2020[5] = 2.7
        starting_values_2020[7] = 0.35
        t_2020 = np.linspace(15, 16, 3)
        ABC_2020 = solve_ivp(ecoNetwork, (15,16), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')  
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # reshape the outputs
        y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 36).transpose(),1)))
        y_2 = pd.DataFrame(data=y_2, columns=species)
        y_2['time'] = combined_times
        
        # choose the final year (we want to compare the final year to the middle of the filters)
        last_year_1 = y.loc[y['time'] == 3.95]
        last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
        last_year_2 = y_2.loc[y_2['time'] == 16]
        last_year_2 = last_year_2.drop('time', axis=1).values.flatten()

        result = ( 
            # 2005 filtering conditions for all nodes
            (((last_year_1[2]-0.87)**2)/0.87) +  
            (((last_year_1[4]-1.4)**2)/1.4) + 
            (((last_year_1[6]-2.2)**2)/2.2) + 
            (((last_year_1[8]-10)**2)/10) + 
            (((last_year_1[9]-1.2)**2)/1.2) + 

            # in 2015, ponies = the same, fallow deer started at 3.33 next year but 0.88 were culled; longhorn got to maximum 2.45; pigs got to maximum 1.1
            # (((last_values_2015[0]-0.43)**2)/0.43) + 
            (((last_values_2015[1]-4.21)**2)/4.21) + 
            (((last_values_2015[3]-2.53)**2)/2.53) + 
            (((last_values_2015[7]-1.15)**2)/1.15) +

            # # in 2016, ponies = the same, fallow deer = 3.9 but 25 were culled; longhorn got to maximum 2.03; pig got to maximum 0.85
            # (((last_values_2016[0]-0.48)**2)/0.48) + 
            (((last_values_2016[1]-4.52)**2)/4.52) + 
            (((last_values_2016[3]-2.19)**2)/2.19) + 
            (((last_values_2016[7]-0.95)**2)/0.95)  +

            # # in 2017, ponies = the same; fallow = 7.34 + 1.36 culled, this was forced instead of filtered bc it seems too high for natural growth rate (maybe supplemented?), cows got to max 2.06; red deer got to 1.85 + 2 culled; pig got to 1.1
            # (((last_values_2017[0]-0.43)**2)/0.43) + 
            # # (((last_values_2017[1]-8.05)**2)/8.05) +
            (((last_values_2017[3]-2.11)**2)/2.11) + 
            (((last_values_2017[5]-2.0)**2)/2.0) +
            # (((last_values_2017[7]-1.45)**2)/1.45) +

            # # in 2018, ponies = same, fallow = 6.62 + 57 culled; cows got to max 2.21; reds got to 2.85 + 3 culled; pigs got to max 1.15
            # (((last_values_2018[0]-0.39)**2)/0.39) + 
            (((last_values_2018[1]-7.98)**2)/7.98) + 
            (((last_values_2018[3]-2.3)**2)/2.3) + 
            (((last_values_2018[5]-3.08)**2)/3.08) + 
            (((last_values_2018[7]-1.15)**2)/1.15) +

            # # in 2019, ponies = 0, fallow = 6.62 + 1.36 culled; longhorn maximum took off filtering condition for tamworth pigs. it got up to 1.8 in one month but was culled back that same month; minimizer isn't dealing well with this
            (((last_values_2019[1]-7.55)**2)/7.55) + 
            (((last_values_2019[3]-2.21)**2)/2.21) + 
            (((last_values_2019[5]-3.39)**2)/3.39) + 
            # (((last_values_2019[7]-1.85)**2)/1.85) +

            # 2020 filtering conditions for all nodes 
            # (((last_year_2[0]-0.65)**2)/0.65) + 
            (((last_year_2[2]-0.74)**2)/0.74) + 
            (((last_year_2[4]-2)**2)/2) + 
            (((last_year_2[6]-4.2)**2)/4.2) + 
            (((last_year_2[7]-0.95)**2)/0.95) +
            (((last_year_2[8]-25.4)**2)/25.4) + (((last_year_2[9]-1.33)**2)/1.33))
            
        if result < 10:
            print("2009", last_year_1, 
                '\n' "2015", last_values_2015,  '\n' "2016", last_values_2016, '\n' "2017", last_values_2017, '\n'"2018",last_values_2018, '\n'"2019", last_values_2019,
                    '\n'"2020", last_year_2, '\n' "r",result)
        return (result)
    else:
        return 1e5



# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.87            1.4        2.2              10              1.2

# [exmoor pony    fallow    'arableGrass',  longhorn, orgCarb    red deer   'roeDeer',tamworthPig,  'thornyScrub','woodland'])
#    0.65         > 5.88         0.74       1.53        2        > 2.7         4.2      > 0.95           25.4          1.33

growth_bds = ((0.6,1),(0.5,0.8),(0.03,0.08))

# VSC leaning on data
# Terminal leaning on methods

interactionbds = (
    # # # exmoor pony; these have no growth (no stallions) - special case
    # (-0.01,-0.001),

    # fallow deer
    # (-7.6,-7.5),(21.5,22),(0.5,0.7),(19.5,20), # leaning on methods
    (-0.09,-0.07),(0.4,0.7),(0,0.1),(0,0.15), # leaning on data
    # -1 to 0: -0.083, 0.63, 0.000066, 0.24, (0.19, passed all filters)
    # -0.6 to -0.1: -0.1, 0.39, 0.002, 0.6 (0.17, passed all filters)
    # -0.4 to -0.6: -0.4, 0.0092, 0.0052, 1.68 (1.03, failed all filters)

    # grassland parkland
    (-0.001,-0.0001),(-0.001,-0.0001),(-1,0),(-0.001,-0.0001),(-0.001,-0.0001),(-0.001,-0.0001),(-0.001,-0.0005),(-0.03,-0.01),(-0.03,-0.01),
    
    # longhorn cattle 
    # (8.5,11),(-6.5,-6),(0,0.1),(5,6), # leaning on methods
    (0.8,0.9),(-0.65,-0.5),(0.01,0.05),(0.1,0.15), # leaning on data
    # 4.12, -7.63, 0.15, 8.1 got to 0.09 but first filter was too low (2.25)
    
    # organic carbon
    (0.001,0.005),(0.001,0.005),(0.07,0.09),(0.001,0.005),(-0.1,-0.096),(0.002,0.005),(0.001,0.005),(0.001,0.005),(0.001,0.0015),(0.07,0.09),
    
    # red deer
    (0.4,0.6),(-0.35,-0.2),(0,0.1),(0.3,0.4), # leaning on data
    # (3,3.5),(-5.6,-5.3),(0.3,0.5),(2.6,3), # leaning on methods
    # -1 to 0: 0.75, -0.28, 0.018, 0.12 - (0.093, passed all filters)
    # -0.6 to -0.4: 0.19, -0.41, 0.046, 0.13 - (0.079, passed all filters
    
    # roe deer
    (1.54,3),(-7,-6),(0.1,0.9),(1.95,3),
    # tamworth pig
    (3,3.5),(-7.3,-7),(0.06,0.09),(2,2.6),
    # thorny scrubland
    (-0.1,0),(-0.1,0),(-0.1,0),(-0.1,0),(-0.1,0),(-0.1,0),(-0.005,-0.001),(-0.1,-0.01),
    # woodland
    (-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(0.0001,0.001),(-0.009,-0.007))


# combine them into one dataframe
bds =  growth_bds + interactionbds

optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 150)
# print(optimization, file=open("final_optimizationOutput.txt", "w"))
print(optimization)
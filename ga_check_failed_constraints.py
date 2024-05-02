# ---- Genetic Algorithm ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import numpy.matlib
from geneticalgorithm import geneticalgorithm as ga
import seaborn as sns
import math
import random


# # # # # --------- MINIMIZATION ----------- # # # # # # #

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']



def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    
    # consumers have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0

    return X * (r + np.matmul(A, X))


def ecoNetwork_nor(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    
    return X * (r + np.matmul(A, X))








def run_model(X0, A, r):
    
    all_times = []
    t_init = np.linspace(0, 3.75, 12)
    results = solve_ivp(ecoNetwork, (0, 3.75), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
    y = pd.DataFrame(data=y, columns=species)
    all_times = np.append(all_times, results.t)
    y['time'] = all_times
    last_results = y.loc[y['time'] == 3.75]
    last_results = last_results.drop('time', axis=1)
    last_results = last_results.values.flatten()
    # ponies, longhorn cattle, and tamworth pigs reintroduced in 2009
    starting_2009 = last_results.copy()
    starting_2009[0] = 1
    starting_2009[3] = 1
    starting_2009[6] = 1
    t_01 = np.linspace(4, 4.75, 3)
    second_ABC = solve_ivp(ecoNetwork, (4,4.75), starting_2009,  t_eval = t_01, args=(A, r), method = 'RK23')
    # 2010
    last_values_2009 = second_ABC.y[0:11, 2:3].flatten()
    starting_values_2010 = last_values_2009.copy()
    starting_values_2010[0] = 0.57
    # fallow deer reintroduced
    starting_values_2010[1] = 1
    starting_values_2010[3] = 1.5
    starting_values_2010[6] = 0.85
    t_1 = np.linspace(5, 5.75, 3)
    third_ABC = solve_ivp(ecoNetwork, (5,5.75), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
    # 2011
    last_values_2010 = third_ABC.y[0:11, 2:3].flatten()
    starting_values_2011 = last_values_2010.copy()
    starting_values_2011[0] = 0.65
    starting_values_2011[1] = 1.9
    starting_values_2011[3] = 1.7
    starting_values_2011[6] = 1.1
    t_2 = np.linspace(6, 6.75, 3)
    fourth_ABC = solve_ivp(ecoNetwork, (6,6.75), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
    # 2012
    last_values_2011 = fourth_ABC.y[0:11, 2:3].flatten()
    starting_values_2012 = last_values_2011.copy()
    starting_values_2012[0] = 0.74
    starting_values_2012[1] = 2.4
    starting_values_2012[3] = 2.2
    starting_values_2012[6] = 1.7
    t_3 = np.linspace(7, 7.75, 3)
    fifth_ABC = solve_ivp(ecoNetwork, (7,7.75), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
    # 2013
    last_values_2012 = fifth_ABC.y[0:11, 2:3].flatten()
    starting_values_2013 = last_values_2012.copy()
    starting_values_2013[0] = 0.43
    starting_values_2013[1] = 2.4
    starting_values_2013[3] = 2.4
    # red deer reintroduced
    starting_values_2013[4] = 1
    starting_values_2013[6] = 0.3
    t_4 = np.linspace(8, 8.75, 3)
    sixth_ABC = solve_ivp(ecoNetwork, (8,8.75), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
    # 2014
    last_values_2013 = sixth_ABC.y[0:11, 2:3].flatten()
    starting_values_2014 = last_values_2013.copy()
    starting_values_2014[0] = 0.44
    starting_values_2014[1] = 2.4
    starting_values_2014[3] = 5
    starting_values_2014[4] = 1
    starting_values_2014[6] = 0.9
    t_5 = np.linspace(9, 9.75, 3)
    seventh_ABC = solve_ivp(ecoNetwork, (9,9.75), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23') 
    # 2015
    last_values_2014 = seventh_ABC.y[0:11, 2:3].flatten()
    starting_values_2015 = last_values_2014
    starting_values_2015[0] = 0.43
    starting_values_2015[1] = 2.4
    starting_values_2015[3] = 2
    starting_values_2015[4] = 1
    starting_values_2015[6] = 0.71
    t_2015 = np.linspace(10, 10.75, 3)
    ABC_2015 = solve_ivp(ecoNetwork, (10,10.75), starting_values_2015,  t_eval = t_2015, args=(A, r), method = 'RK23')
    # 2016
    last_values_2015 = ABC_2015.y[0:11, 2:3].flatten()
    starting_values_2016 = last_values_2015.copy()
    starting_values_2016[0] = 0.48
    starting_values_2016[1] = 3.3
    starting_values_2016[3] = 1.7
    starting_values_2016[4] = 2
    starting_values_2016[6] = 0.7
    t_2016 = np.linspace(11, 11.75, 3)
    ABC_2016 = solve_ivp(ecoNetwork, (11,11.75), starting_values_2016,  t_eval = t_2016, args=(A, r), method = 'RK23')       
    # 2017
    last_values_2016 = ABC_2016.y[0:11, 2:3].flatten()
    starting_values_2017 = last_values_2016.copy()
    starting_values_2017[0] = 0.43
    starting_values_2017[1] = 3.9
    starting_values_2017[3] = 1.7
    starting_values_2017[4] = 1.1
    starting_values_2017[6] = 0.95
    t_2017 = np.linspace(12, 12.75, 3)
    ABC_2017 = solve_ivp(ecoNetwork, (12,12.75), starting_values_2017,  t_eval = t_2017, args=(A, r), method = 'RK23')     
    # 2018
    last_values_2017 = ABC_2017.y[0:11, 2:3].flatten()
    starting_values_2018 = last_values_2017.copy()
    # pretend bison were reintroduced (to estimate growth rate / interaction values)
    starting_values_2018[0] = 0
    starting_values_2018[1] = 6.0
    starting_values_2018[3] = 1.9
    starting_values_2018[4] = 1.9
    starting_values_2018[6] = 0.84
    t_2018 = np.linspace(13, 13.75, 3)
    ABC_2018 = solve_ivp(ecoNetwork, (13,13.75), starting_values_2018,  t_eval = t_2018, args=(A, r), method = 'RK23')     
    # 2019
    last_values_2018 = ABC_2018.y[0:11, 2:3].flatten()
    starting_values_2019 = last_values_2018.copy()
    starting_values_2019[0] = 0
    starting_values_2019[1] = 6.6
    starting_values_2019[3] = 1.7
    starting_values_2019[4] = 2.9
    starting_values_2019[6] = 0.44
    t_2019 = np.linspace(14, 14.75, 3)
    ABC_2019 = solve_ivp(ecoNetwork, (14,14.75), starting_values_2019,  t_eval = t_2019, args=(A, r), method = 'RK23')
    # 2020
    last_values_2019 = ABC_2019.y[0:11, 2:3].flatten()
    starting_values_2020 = last_values_2019.copy()
    starting_values_2020[0] = 0.65
    starting_values_2020[1] = 5.9
    starting_values_2020[3] = 1.5
    starting_values_2020[4] = 2.7
    starting_values_2020[6] = 0.55
    t_2020 = np.linspace(15, 16, 3)
    ABC_2020 = solve_ivp(ecoNetwork, (15,16), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')     
    # concatenate & append all the runs
    combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
    combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
    # reshape the outputs
    y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
    y_2 = pd.DataFrame(data=y_2, columns=species)
    y_2['time'] = combined_times
    # choose the final year (we want to compare the final year to the middle of the filters)
    last_year_1 = y.loc[y['time'] == 3.75]
    last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
    last_year_2 = y_2.loc[y_2['time'] == 16]
    last_year_2 = last_year_2.drop('time', axis=1).values.flatten()
    return last_year_1, last_year_2, last_values_2015, last_values_2016, last_values_2017, last_values_2018, last_values_2019, y_2






def reality_1(A, r): # scrub without consumers or woodland
    t = np.linspace(2005, 2055, 10)
    X0 = [0, 0, 1, 0, 0, 0, 0, 1, 0] # no herbivores
    r[8] = 0
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2055), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_1 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 10).transpose(),1)))
    realityCheck_1 = pd.DataFrame(data=realityCheck_1, columns=species)
    realityCheck_1['time'] = realityCheck_ABC.t
    return realityCheck_1




def reality_2(A, r): # grassland with no scrub or woodland
    t = np.linspace(2005, 2055, 10)
    X0 = [0, 0, 1, 0, 0, 0, 0, 0, 0]
    r[7] = 0
    r[8] = 0
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2055), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_2 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 10).transpose(),1)))
    realityCheck_2 = pd.DataFrame(data=realityCheck_2, columns=species)
    realityCheck_2['time'] = realityCheck_ABC.t
    return realityCheck_2



def reality_3(A, r): # no herbivores
    t = np.linspace(2005, 2055, 10)
    X0 = [0, 0, 1, 0, 0, 0, 0, 1, 1] # no herbivores
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2055), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_3 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 10).transpose(),1)))
    realityCheck_3 = pd.DataFrame(data=realityCheck_3, columns=species)
    realityCheck_3['time'] = realityCheck_ABC.t
    return realityCheck_3




def objectiveFunction(x):
    # define X0
    X0 = [0,0,1,0,0,1,0,1,1]

    x = np.insert(x,0,0)
    x = np.insert(x,1,0)
    x = np.insert(x,3,0)
    x =np.insert(x,4,0)
    x =np.insert(x,5,0)
    x = np.insert(x,6,0)

    r =  x[0:9]

    # insert interaction matrices of 0
    # pony
    x = np.insert(x,9,0)
    x = np.insert(x,10,0)
    x = np.insert(x,12,0)
    x = np.insert(x,13,0)
    x = np.insert(x,14,0)
    x = np.insert(x,15,0)
    # fallow
    x = np.insert(x,18,0)
    x = np.insert(x,19,0)
    x = np.insert(x,21,0)
    x = np.insert(x,22,0)
    x = np.insert(x,23,0)
    x = np.insert(x,24,0)
    # cattle
    x = np.insert(x,36,0)
    x = np.insert(x,37,0)
    x = np.insert(x,39,0)
    x = np.insert(x,40,0)
    x = np.insert(x,41,0)
    x = np.insert(x,42,0)
    # red deer
    x = np.insert(x,45,0)
    x = np.insert(x,46,0)
    x = np.insert(x,48,0)
    x = np.insert(x,49,0)
    x = np.insert(x,50,0)
    x = np.insert(x,51,0)
    # roe
    x = np.insert(x,54,0)
    x = np.insert(x,55,0)
    x = np.insert(x,57,0) 
    x = np.insert(x,58,0)
    x = np.insert(x,59,0)
    x = np.insert(x,60,0)
    # pig
    x = np.insert(x,63,0)
    x = np.insert(x,64,0)
    x = np.insert(x,66,0)
    x = np.insert(x,67,0)
    x = np.insert(x,68,0)
    x = np.insert(x,69,0)
    # scrub
    x = np.insert(x,74,0)
    # wood
    x = np.insert(x,83,0)

    # define interaction strength
    interaction_strength = x[9:90]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(9,9),index = species, columns=species)
    A = interaction_strength.to_numpy()

    # run the model
    last_year_1, last_year_2, last_values_2015, last_values_2016, last_values_2017, last_values_2018, last_values_2019, y_2 = run_model(X0, A, r)
    # run the reality checks - this one's for scrub
    realityCheck_1 = reality_1(A,r)
    realityCheck_1_last_year = realityCheck_1.loc[realityCheck_1['time'] == 2055]
    # grassland without competition
    realityCheck_2 = reality_2(A,r)
    realityCheck_2_last_year = realityCheck_2.loc[realityCheck_2['time'] == 2055]
    # reality check 3
    realityCheck_3 = reality_3(A,r)
    realityCheck_3_fifty_years = realityCheck_3.loc[realityCheck_3['time'] == 2055]


    # find runs with outputs closest to the middle of the filtering conditions
    result = ( 
        # 2009 filtering conditions
        (((last_year_1[2]-0.6)/0.6)**2) +  
        (((last_year_1[5]-2.2)/2.2)**2) + 
        (((last_year_1[7]-3)/3)**2) + 
        (((last_year_1[8]-2)/2)**2) + 
        # 2015
        (((last_values_2015[0]-0.44)/0.44)**2) + 
        (((last_values_2015[1]-3.3)/3.3)**2) + 
        (((last_values_2015[3]-2.5)/2.5)**2) +
        (((last_values_2015[4]-1)/1)**2) +
        (((last_values_2015[6]-1.1)/1.1)**2) +
        # # 2016
        (((last_values_2016[0]-0.47)/0.47)**2) + 
        (((last_values_2016[1]-3.9)/3.9)**2) + 
        (((last_values_2016[3]-2.1)/2.1)**2) +
        (((last_values_2016[4]-2)/2)**2) +
        (((last_values_2016[6]-0.88)/0.88)**2) +
        # # 2017
        (((last_values_2017[0]-0.44)/0.44)**2) + 
        (((last_values_2017[1]-6)/6)**2) +
        (((last_values_2017[3]-2.1)/2.1)**2) +
        (((last_values_2017[4]-1.1)/1.1)**2) +
        (((last_values_2017[6]-1.1)/1.1)**2) +
        # # # 2018
        (((last_values_2018[1]-7.5)/7.5)**2) +
        (((last_values_2018[3]-2.2)/2.2)**2) + 
        (((last_values_2018[4]-1.9)/1.9)**2) + 
        (((last_values_2018[6]-1.2)/1.2)**2) +
        # # # 2019
        (((last_values_2019[1]-8.3)/8.3)**2) + 
        (((last_values_2019[3]-2.1)/2.1)**2) + 
        (((last_values_2019[4]-2.9)/2.9)**2) + 
        (((last_values_2019[6]-0.5)/0.5)**2) + 
        # 2020
        (((last_year_2[0]-0.7)/0.7)**2) + 
        (((last_year_2[2]-0.3)/0.3)**2) +
        (((last_year_2[5]-4.2)/4.2)**2) + 
        (((last_year_2[6]-1)/1)**2) + 
        (((last_year_2[7]-12.1)/12.1)**2) +
        (((last_year_2[8]-3.7)/3.7)**2) +
        
        # constraint - scrubland should be 23.3
        (((realityCheck_1_last_year.iloc[0]['thornyScrub']-23.3)/23.3)**2) + 
        # constraint
        (((realityCheck_2_last_year.iloc[0]['grasslandParkland']-1.1)/1.1)**2) + 
        # constraint
        (((realityCheck_3_fifty_years.iloc[0]['woodland']-17.2)/17.2)**2)
    )

    if result < 5:
        print(result)
    return (result)



def run_optimizer():

    bds = np.array([
        # growth
        [0.5,1],[0,0.5],[0,0.15],
        # exmoor pony
        [2.5,5],[0,0.5],[0,0.5],    
        # fallow deer
        [2.5,5],[0,0.5],[0,0.5],  
        # grassland parkland 
        [-0.05,0],[-0.05,0],[-1,-0.4],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],
        # longhorn cattle
        [2,5],[0,0.5],[0,0.5],    
        # red deer
        [2.5,5],[0,0.5],[0,0.5],   
        # roe deer
        [2.5,5],[0,0.5],[0,0.5],    
        # tamworth pig 
        [2.5,5],[0,0.5],[0,0.5],    
        # thorny scrubland
        [-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.025,0],[-0.05,0],
        # woodland
        [-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[-0.05,0],[0,0.01],[-0.025,0],
    ])


    # bds = np.array([
    #     # growth
    #     [0.91,0.92],[0.34,0.35],[0.1,0.12],
    #     # exmoor pony
    #     [2.7,2.75],[0.16,0.17],[0.4,0.42],    
    #     # fallow deer
    #     [3.8,4],[0.36,0.37],[0.48,0.5],  
    #     # grassland parkland 
    #     [-0.0025,-0.001],[-0.0035,-0.003],[-0.83,-0.82],[-0.015,-0.0093],[-0.003,-0.0025],[-0.001,-0.0009],[-0.009,-0.005],[-0.046,-0.044],[-0.049,-0.048],
    #     # longhorn cattle
    #     [4.95,5],[0.21,0.22],[0.39,0.4],    
    #     # red deer
    #     [2.8,2.9],[0.29,0.31],[0.42,0.43],   
    #     # roe deer
    #     [4.8,4.9],[0.1,0.3],[0.1,0.4],    
    #     # tamworth pig 
    #     [3.65,3.7],[0.22,0.23],[0.4,0.41],    
    #     # thorny scrubland
    #     [-0.003,-0.001],[-0.01,-0.005],[-0.006,-0.005],[-0.005,-0.0032],[-0.0025,-0.0021],[-0.002,-0.0015],[-0.016,-0.015],[-0.023,-0.022],
    #     # woodland
    #     [-0.0042,-0.004],[-0.0059,-0.0058],[-0.0083,-0.0082],[-0.004,-0.0039],[-0.0032,-0.0031],[-0.0037,-0.0035],[0.005,0.008],[-0.0063,-0.0061],
    # ])

    algorithm_param = {'max_num_iteration':25,\
                    'population_size': 1500,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':10}

    optimization =  ga(function = objectiveFunction, dimension = 46, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=30)
    optimization.run()
    return optimization.output_dict








def graph_results():

    output_parameters = run_optimizer()

    output_df = pd.DataFrame(output_parameters)

    # define X0
    X0 = [0,0,1,0,0,1,0,1,1]
    # fill in the zeroes
    output_parameters["variable"] = np.insert(output_parameters["variable"],0,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],1,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],3,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],4,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],5,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],6,0)
    # pony
    output_parameters["variable"] = np.insert(output_parameters["variable"],9,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],10,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],12,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],13,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],14,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],15,0)
    # fallow
    output_parameters["variable"] = np.insert(output_parameters["variable"],18,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],19,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],21,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],22,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],23,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],24,0)
    # cattle
    output_parameters["variable"] = np.insert(output_parameters["variable"],36,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],37,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],39,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],40,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],41,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],42,0)
    # red deer
    output_parameters["variable"] = np.insert(output_parameters["variable"],45,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],46,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],48,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],49,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],50,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],51,0)
    # roe
    output_parameters["variable"] = np.insert(output_parameters["variable"],54,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],55,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],57,0) 
    output_parameters["variable"] = np.insert(output_parameters["variable"],58,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],59,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],60,0)
    # pig
    output_parameters["variable"] = np.insert(output_parameters["variable"],63,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],64,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],66,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],67,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],68,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],69,0)

    # scrub
    output_parameters["variable"] = np.insert(output_parameters["variable"],74,0)
    # wood
    output_parameters["variable"] = np.insert(output_parameters["variable"],83,0)
    # define parameters
    r = output_parameters["variable"][0:9]

    interaction_strength = output_parameters["variable"][9:90]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(9,9),index = species, columns=species)
    A = interaction_strength.to_numpy()


    # run it
    last_year_1, last_year_2, last_values_2015, last_values_2016, last_values_2017, last_values_2018, last_values_2019, y_2 = run_model(X0, A, r)
    # put it in a dataframe
    y_values = y_2[["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland"]].values.flatten()
    species_list = np.tile(["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland"],48) 
    indices = np.repeat(y_2['time'], 9)
    final_df_exp1 = pd.DataFrame(
        {'Abundance %': y_values, 'Ecosystem Element': species_list, 'Time': indices})
    # graph it
    f = sns.FacetGrid(final_df_exp1, col="Ecosystem Element", col_wrap=4, sharey = False)
    f.map(sns.lineplot, 'Time', 'Abundance %')
    # add filter lines
    axes = f.axes.flatten()
    # 2009
    f.axes[2].vlines(x=3.75,ymin=0.18,ymax=1, color='r')
    f.axes[5].vlines(x=3.75,ymin=1.7,ymax=3.3, color='r')
    f.axes[7].vlines(x=3.75,ymin=1,ymax=4.9, color='r')
    f.axes[8].vlines(x=3.75,ymin=1,ymax=2.9, color='r')
    f.axes[0].plot(4, 1, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(4, 1, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(4, 1, 'go',markersize=2.5) # and forcings
    # 2010
    f.axes[1].plot(5, 1, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(5, 1.5, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(5, 0.85, 'go',markersize=2.5) # and forcings
    # 2011
    f.axes[0].plot(6, 0.65, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(6, 1.9, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(6, 1.7, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(6, 1.1, 'go',markersize=2.5) # and forcings
    # 2012
    f.axes[0].plot(7, 0.74, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(7, 2.4, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(7, 2.2, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(7, 1.7, 'go',markersize=2.5) # and forcings
    # 2013
    f.axes[0].plot(8, 0.43, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(8, 2.4, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(8, 2.4, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(8, 1, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(8, 0.3, 'go',markersize=2.5) # and forcings
    # 2014
    f.axes[0].plot(9, 0.44, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(9, 2.4, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(9, 5, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(9, 1, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(9, 0.9, 'go',markersize=2.5) # and forcings
    # 2015
    f.axes[0].vlines(x=10.75,ymin=0.39,ymax=0.49, color='r')
    f.axes[1].vlines(x=10.75,ymin=2.7,ymax=3.9, color='r')
    f.axes[3].vlines(x=10.75,ymin=2,ymax=2.9, color='r')
    f.axes[4].vlines(x=10.75,ymin=0.6,ymax=1.4, color='r')
    f.axes[6].vlines(x=10.75,ymin=0.6,ymax=1.6, color='r')
    f.axes[0].plot(10, 0.43, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(10, 2.4, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(10, 2, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(10, 1, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(10, 0.71, 'go',markersize=2.5) # and forcings
    # 2016
    f.axes[0].vlines(x=11.75,ymin=0.43,ymax=0.5, color='r')
    f.axes[1].vlines(x=11.75,ymin=3.3,ymax=4.5, color='r')
    f.axes[3].vlines(x=11.75,ymin=1.6,ymax=2.5, color='r')
    f.axes[4].vlines(x=11.75,ymin=1.6,ymax=2.4, color='r')
    f.axes[6].vlines(x=11.75,ymin=0.35,ymax=1.4, color='r')
    f.axes[0].plot(11, 0.48, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(11, 3.3, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(11, 1.7, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(11, 2, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(11, 0.7, 'go',markersize=2.5) # and forcings
    # 2017
    f.axes[0].vlines(x=12.75,ymin=0.39,ymax=0.48, color='r')
    f.axes[1].vlines(x=12.75,ymin=5.4,ymax=6.6, color='r')
    f.axes[3].vlines(x=12.75,ymin=1.6,ymax=2.5, color='r')
    f.axes[4].vlines(x=12.75,ymin=0.7,ymax=1.5, color='r')
    f.axes[6].vlines(x=12.75,ymin=0.6,ymax=1.6, color='r')
    f.axes[0].plot(12, 0.43, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(12, 3.9, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(12, 1.7, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(12, 1.1, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(12, 0.95, 'go',markersize=2.5) # and forcings
    # 2018
    f.axes[1].vlines(x=13.75,ymin=6.9,ymax=8.1, color='r')
    f.axes[3].vlines(x=13.75,ymin=1.7,ymax=2.7, color='r')
    f.axes[4].vlines(x=13.75,ymin=1.5,ymax=2.2, color='r')
    f.axes[6].vlines(x=13.75,ymin=0.7,ymax=1.7, color='r')
    f.axes[0].plot(13, 0, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(13, 6.0, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(13, 1.9, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(13, 1.9, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(13, 0.84, 'go',markersize=2.5) # and forcings
    # 2019
    f.axes[1].vlines(x=14.75,ymin=7.7,ymax=8.9, color='r')
    f.axes[3].vlines(x=14.75,ymin=1.6,ymax=2.5, color='r')
    f.axes[4].vlines(x=14.75,ymin=2.5,ymax=3.2, color='r')
    f.axes[6].vlines(x=14.75,ymin=0,ymax=1, color='r')
    f.axes[0].plot(14, 0, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(14, 6.6, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(14, 1.7, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(14, 2.9, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(14, 0.44, 'go',markersize=2.5) # and forcings
    # 2020
    f.axes[0].vlines(x=15.75,ymin=0.61,ymax=0.7, color='r')
    f.axes[6].vlines(x=15.75,ymin=0.5,ymax=1.5, color='r')
    f.axes[2].vlines(x=15.75,ymin=0.18,ymax=0.41, color='r')
    f.axes[5].vlines(x=15.75,ymin=1.7,ymax=6.7, color='r')
    f.axes[7].vlines(x=15.75,ymin=9.7,ymax=14.4, color='r')
    f.axes[8].vlines(x=15.75,ymin=2.8,ymax=4.6, color='r')
    f.axes[0].plot(15, 0.65, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(15, 5.9, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(15, 1.5, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(15, 2.7, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(15, 0.55, 'go',markersize=2.5) # and forcings
 
    # show plot
    f.fig.suptitle('Optimiser outputs: second parameter set')
    plt.tight_layout()
    # plt.savefig('optimiser_outputs.png')
    plt.show()







    # now reality check 1
    t = np.linspace(2015, 2016, 2)
    X0 = [1, 1, 0, 1, 1, 1, 1, 0, 0] # no primary producers, only consumers

    realityCheck_ABC = solve_ivp(ecoNetwork, (2015, 2016), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_1 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 2).transpose(),1)))
    realityCheck_1 = pd.DataFrame(data=realityCheck_1, columns=species)
    realityCheck_1['time'] = realityCheck_ABC.t
    # extract the node values from all dataframes
    final_runs1 = realityCheck_1.drop(['time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (2))
    # time 
    firstODEyears = np.repeat(realityCheck_1['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})

    # at time 1, all herbivores should be < 0.01
    last_time = final_df.loc[final_df["Time"] == 2016]
    # print(last_time)

    if last_time.loc[last_time['Ecosystem Element'] == 'exmoorPony', 'Abundance %'].item() < 0.1 and last_time.loc[last_time['Ecosystem Element'] == 'fallowDeer', 'Abundance %'].item() < 0.1 and last_time.loc[last_time['Ecosystem Element'] == 'longhornCattle', 'Abundance %'].item() < 0.1 and last_time.loc[last_time['Ecosystem Element'] == 'redDeer', 'Abundance %'].item() < 0.1 and last_time.loc[last_time['Ecosystem Element'] == 'roeDeer', 'Abundance %'].item() < 0.1 and last_time.loc[last_time['Ecosystem Element'] == 'tamworthPig', 'Abundance %'].item() < 0.1:
        output_df["Constraint 1"] = 1
    else:
        output_df["Constraint 1"] = 0

    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Abundance %')
    g.fig.suptitle('Reality check: No primary producers')
    plt.tight_layout()
    plt.show()





    # now reality check 2 - woodland should level out; scrub should increase at onset
    t = np.linspace(2005, 2105, 100)
    X0 = [0, 0, 1, 0, 0, 0, 0, 1, 1] # no herbivores
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_2 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 100).transpose(),1)))
    realityCheck_2 = pd.DataFrame(data=realityCheck_2, columns=species)
    realityCheck_2['time'] = realityCheck_ABC.t
    # extract the node values from all dataframes
    final_runs1 = realityCheck_2.drop(['time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (100))
    # time 
    firstODEyears = np.repeat(realityCheck_2['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})

    last_time = final_df.loc[final_df["Time"] == 2105]

    # if thorny scrub is not > woodland and grass at approx. 15 years and woodland not > scrub and grass at 100 years, fail
    max_scrub_time = final_df.loc[final_df["Time"] > 2020]
    all_scrub_time = max_scrub_time.loc[max_scrub_time["Time"] < 2021]
    # pick a year to look at 
    time_options = random.choice(list(all_scrub_time.Time.unique()))
    scrub_time = final_df.loc[final_df["Time"] == time_options]

    # remember it's scaled to 1
    if (scrub_time.loc[scrub_time['Ecosystem Element'] == 'thornyScrub', 'Abundance %'].item() * 0.043 > scrub_time.loc[scrub_time['Ecosystem Element'] == 'woodland', 'Abundance %'].item() *0.058 and scrub_time.loc[scrub_time['Ecosystem Element'] == 'thornyScrub', 'Abundance %'].item() * 0.043 > scrub_time.loc[scrub_time['Ecosystem Element'] == 'grasslandParkland', 'Abundance %'].item() * 0.899) and (last_time.loc[last_time['Ecosystem Element'] == 'woodland', 'Abundance %'].item() *0.058 > last_time.loc[last_time['Ecosystem Element'] == 'thornyScrub', 'Abundance %'].item() *0.043 and last_time.loc[last_time['Ecosystem Element'] == 'woodland', 'Abundance %'].item() * 0.058 > last_time.loc[last_time['Ecosystem Element'] == 'grasslandParkland', 'Abundance %'].item() * 0.899):
        output_df["Constraint 2"] = 1
    else:
        output_df["Constraint 2"] = 0
    
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Abundance %')
    g.fig.suptitle('Reality check: No consumers')
    plt.tight_layout()
    plt.show()

    # if woodland hasn't levelled out at 17.2, fail
    if last_time.loc[last_time['Ecosystem Element'] == 'woodland', 'Abundance %'].item() > 15.48 and last_time.loc[last_time['Ecosystem Element'] == 'woodland', 'Abundance %'].item() < 18.92:
        output_df["Constraint 6"] = 1
    else:
        output_df["Constraint 6"] = 0

     





    # now reality check 3: overloaded consumers, no negative r
    t = np.linspace(2005, 2105, 100)
    X0 = [2, 2, 1, 2, 2, 2, 2, 1, 1]
    A[[0],2] = 1
    A[[1],2] = 1
    A[[3],2] = 1
    A[[4],2] = 1
    A[[5],2] = 1
    A[[6],2] = 1

    realityCheck_ABC = solve_ivp(ecoNetwork_nor, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_3 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 100).transpose(),1)))
    realityCheck_3 = pd.DataFrame(data=realityCheck_3, columns=species)
    realityCheck_3['time'] = realityCheck_ABC.t
    # extract the node values from all dataframes
    final_runs1 = realityCheck_3.drop(['time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (100))
    # time 
    firstODEyears = np.repeat(realityCheck_3['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
    
    # if all habitats aren't < 0.01 after 100 years, fail
    last_time = final_df.loc[final_df["Time"] == 2105]

    if last_time.loc[last_time['Ecosystem Element'] == 'grasslandParkland', 'Abundance %'].item() > 0.01 or last_time.loc[last_time['Ecosystem Element'] == 'woodland', 'Abundance %'].item() > 0.01 or last_time.loc[last_time['Ecosystem Element'] == 'thornyScrub', 'Abundance %'].item() > 0.01:
        output_df["Constraint 3"] = 0
    else:
        output_df["Constraint 3"] = 1
    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Abundance %')
    g.fig.suptitle('Reality check: Overloaded consumers')
    plt.tight_layout()
    plt.show()







    # now reality check 4 - scrubland should level out at 23.3
    t = np.linspace(2005, 2105, 100)
    X0 = [0, 0, 1, 0, 0, 0, 0, 1, 0] # no consumers or woodland
    r[8] = 0

    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_4 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 100).transpose(),1)))
    realityCheck_4 = pd.DataFrame(data=realityCheck_4, columns=species)
    realityCheck_4['time'] = realityCheck_ABC.t
    # extract the node values from all dataframes
    final_runs1 = realityCheck_4.drop(['time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (100))
    # time 
    firstODEyears = np.repeat(realityCheck_4['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
    last_time = final_df.loc[final_df["Time"] == 2105]

    # if scrubland hasn't levelled out around 23.3, fail
    if last_time.loc[last_time['Ecosystem Element'] == 'thornyScrub', 'Abundance %'].item() > 20.97 and last_time.loc[last_time['Ecosystem Element'] == 'thornyScrub', 'Abundance %'].item() < 25.63:
        output_df["Constraint 5"] = 1
    else:
        output_df["Constraint 5"] = 0
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Abundance %')
    g.fig.suptitle('Reality check: No woodland or consumers')
    plt.tight_layout()
    plt.show()





    # now reality check 5 - grassland should level out at 1.1
    t = np.linspace(2005, 2105, 100)
    X0 = [0, 0, 1, 0, 0, 0, 0, 0, 0] # nothing but grassland
    r[7]=0
    r[8]=0

    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_5 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 100).transpose(),1)))
    realityCheck_5 = pd.DataFrame(data=realityCheck_5, columns=species)
    realityCheck_5['time'] = realityCheck_ABC.t
    # extract the node values from all dataframes
    final_runs1 = realityCheck_5.drop(['time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (100))
    # time 
    firstODEyears = np.repeat(realityCheck_5['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})

    last_time = final_df.loc[final_df["Time"] == 2105]

    # if scrubland hasn't levelled out around 23.3, fail
    if last_time.loc[last_time['Ecosystem Element'] == 'grasslandParkland', 'Abundance %'].item() > 0.99 and last_time.loc[last_time['Ecosystem Element'] == 'grasslandParkland', 'Abundance %'].item() < 1.21:
        output_df["Constraint 4"] = 1
    else:
        output_df["Constraint 4"] = 0
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Abundance %')
    g.fig.suptitle('Reality check: Grassland only')
    plt.tight_layout()
    plt.show()



    return output_df





# keep track of everything
final_df = {}
variables_used = {}

for n in range(10):

    print(n)

    output_df = graph_results()

    print(output_df)

    if output_df["Constraint 1"][0].item() + output_df["Constraint 2"][0].item()+ output_df["Constraint 3"][0].item() + output_df["Constraint 4"][0].item() + output_df["Constraint 5"][0].item() + output_df["Constraint 6"][0].item() == 6:
        print("passed all")
        passed_all = 1
    else: 
        passed_all = 0



    # then append outputs
    final_df[n] = {"Run number": n, 
                            "Fit": output_df["function"][0].item(), 
                            "Passed Constraint 1": output_df["Constraint 1"][0].item(),
                            "Passed Constraint 2": output_df["Constraint 2"][0].item(),
                            "Passed Constraint 3": output_df["Constraint 3"][0].item(),
                            "Passed Constraint 4": output_df["Constraint 4"][0].item(),
                            "Passed Constraint 5": output_df["Constraint 5"][0].item(),
                            "Passed Constraint 6": output_df["Constraint 6"][0].item(),
                            "Passed All Constraints": passed_all,

    }

    variables_used[n] = output_df["variable"]


all_df = pd.DataFrame.from_dict(final_df, "index")
all_variables = pd.DataFrame.from_dict(variables_used, "index")

print(all_df)

# and save to csv
all_df.to_csv("all_df_ga_manualadjustments_10.csv")
all_variables.to_csv("all_variables_ga_manualadjustments_10.csv")

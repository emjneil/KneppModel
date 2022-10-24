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


# # # # # --------- MINIMIZATION ----------- # # # # # # #

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']


def ecoNetwork(t, X, A, r):
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


def reality_1(A, r): # remove primary producers, consumers should decline to zero 
    t = np.linspace(2015, 2016, 10)
    X0 = [1, 1, 0, 1, 1, 1, 1, 0, 0] # no primary producers, only consumers
    realityCheck_ABC = solve_ivp(ecoNetwork, (2015, 2016), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_1 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 10).transpose(),1)))
    realityCheck_1 = pd.DataFrame(data=realityCheck_1, columns=species)
    realityCheck_1['time'] = realityCheck_ABC.t
    return realityCheck_1


def reality_2(A, r): # thorny scrub should increase to 50% with no herbivory
    t = np.linspace(2005, 2020, 10)
    X0 = [0, 0, 1, 0, 0, 0, 0, 1, 1] # no herbivores
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2020), X0,  t_eval = t, args=(A, r), method = 'RK23') 
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

def reality_4(A, r): # overloaded herbivores
    t = np.linspace(2005, 2105, 100)
    X0 = [2, 2, 1, 2, 2, 2, 2, 1, 1] # overloaded consumers
    # consumers are invincible (could represent supplementary feeding)
    A[[0],0] = 0
    A[[1],1] = 0
    A[[3],3] = 0
    A[[4],4] = 0
    A[[5],5] = 0
    A[[6],6] = 0
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_4 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 100).transpose(),1)))
    realityCheck_4 = pd.DataFrame(data=realityCheck_4, columns=species)
    realityCheck_4['time'] = realityCheck_ABC.t
    return realityCheck_4

def objectiveFunction(x):
    # only primary producers should have intrinsic growth rate
    x = np.insert(x,0,0)
    x = np.insert(x,1,0)
    x = np.insert(x,3,0)
    x = np.insert(x,4,0)
    x = np.insert(x,5,0)
    x = np.insert(x,6,0)
    r =  x[0:9]
    # insert interaction matrices of 0
    # pony
    x = np.insert(x,9,-0.04)
    x = np.insert(x,10,0)
    x = np.insert(x,11,0)
    x = np.insert(x,12,0)
    x = np.insert(x,13,0)
    x = np.insert(x,14,0)
    x = np.insert(x,15,0)
    x = np.insert(x,16,0)
    x = np.insert(x,17,0)
    # fallow
    x = np.insert(x,18,0)
    x = np.insert(x,19,-0.14)
    x = np.insert(x,20,0.38)
    x = np.insert(x,21,0)
    x = np.insert(x,22,0)
    x = np.insert(x,23,0)
    x = np.insert(x,24,0)
    x = np.insert(x,25,0.07)
    x = np.insert(x,26,0.12)

    # cattle
    x = np.insert(x,36,0)
    x = np.insert(x,37,0)
    x = np.insert(x,38,0.80)
    x = np.insert(x,39,-0.59)
    x = np.insert(x,40,0)
    x = np.insert(x,41,0)
    x = np.insert(x,42,0)
    x = np.insert(x,43,0.073)
    x = np.insert(x,44,0.14)
    # red deer
    x = np.insert(x,45,0)
    x = np.insert(x,46,0)
    x = np.insert(x,47,0.66)
    x = np.insert(x,48,0)
    x = np.insert(x,49,-0.40)
    x = np.insert(x,50,0)
    x = np.insert(x,51,0)
    x = np.insert(x,52,0.052)
    x = np.insert(x,53,0.083)

    # roe
    x = np.insert(x,54,0)
    x = np.insert(x,55,0)
    x = np.insert(x,56,0.38)
    x = np.insert(x,57,0) 
    x = np.insert(x,58,0)
    x = np.insert(x,59,-0.76)
    x = np.insert(x,60,0)
    x = np.insert(x,61,0.26)
    x = np.insert(x,62,0.28)

    # pig
    x = np.insert(x,63,0)
    x = np.insert(x,64,0)
    x = np.insert(x,65,0.34)
    x = np.insert(x,66,0)
    x = np.insert(x,67,0)
    x = np.insert(x,68,0)
    x = np.insert(x,69,-0.75)
    x = np.insert(x,70,0.062)
    x = np.insert(x,71,0.16)

    # scrub
    x = np.insert(x,74,0)
    # wood
    x = np.insert(x,83,0)
    # define X0
    X0 = [0,0,1,0,0,1,0,1,1]
    # define interaction strength
    interaction_strength = x[9:90]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(9,9),index = species, columns=species)
    A = interaction_strength.to_numpy()

    # run the model
    last_year_1, last_year_2, last_values_2015, last_values_2016, last_values_2017, last_values_2018, last_values_2019, y_2 = run_model(X0, A, r)
    # run the reality checks
    realityCheck_2 = reality_2(A,r)
    realityCheck_3 = reality_3(A,r)
    realityCheck_2_fifteen_years = realityCheck_2.loc[realityCheck_2['time'] == 2020]
    realityCheck_2_fifty_years = realityCheck_3.loc[realityCheck_3['time'] == 2055]

    # find runs with outputs closest to the middle of the filtering conditions
    result = ( 
        # 2009 filtering conditions
        # (((last_year_1[2]-0.6)/0.6)**2) +  
        # (((last_year_1[5]-2.2)/2.2)**2) + 
        # (((last_year_1[7]-3)/3)**2) + 
        # (((last_year_1[8]-2)/2)**2) + 
        # 2020
        # (((last_year_2[0]-0.7)/0.7)**2) + 
        (((last_year_2[2]-0.3))**2) +
        # (((last_year_2[5]-4.2)/4.2)**2) + 
        # (((last_year_2[6]-1)/1)**2) + 
        (((last_year_2[7]-12.1)/12.1)**2) + 
        (((last_year_2[8]-3.5))**2)
        
        # sensitivity test 1 - no primary producers, so consumers should = 0
        # (((realityCheck_1_last_year.iloc[0]['exmoorPony']-0))**2) +
        # (((realityCheck_1_last_year.iloc[0]['fallowDeer']-0))**2) +
        # (((realityCheck_1_last_year.iloc[0]['longhornCattle']-0))**2) +
        # (((realityCheck_1_last_year.iloc[0]['roeDeer']-0))**2) +
        # (((realityCheck_1_last_year.iloc[0]['redDeer']-0))**2) +
        # (((realityCheck_1_last_year.iloc[0]['tamworthPig']-0))**2) +
        # sensitivity test 2 - no consumers, so approx. 80% woodland after 50 yrs, 50% scrub in 15 years 
        # scrub - 4.3% in 2005 
        # # woodland - 5.8% in 2005
        # (((realityCheck_2_fifteen_years.iloc[0]['thornyScrub']-23.6))**2) +
        # (((realityCheck_2_fifty_years.iloc[0]['woodland']-17.2))**2) +
        # (((realityCheck_2_fifty_years.iloc[0]['thornyScrub']-4))**2) + # scrub should decline as woodland increases
        # (((realityCheck_2_fifty_years.iloc[0]['grasslandParkland']-0))**2) # scrub should decline as woodland increases

    )

    # print the output
    if result < 0.1:
        print(result)
    return (result)



def run_optimizer():
    bds = np.array([
        # growth
        [0.63,0.64],[0.4,0.45],[0.3,0.35],
        # grassland parkland
        [-0.025,0],[-0.025,0],[-0.77,-0.76],[-0.025,0],[-0.025,0],[-0.025,0],[-0.025,0],[-0.03,-0.02],[-0.03,-0.02],
        # thorny scrubland 
        [-0.1,-0.001],[-0.1,-0.001],[-0.1,-0.001],[-0.1,-0.001],[-0.1,-0.001],[-0.1,-0.001],[-0.01,-0.002],[-0.1,-0.01],
        # woodland
        [-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[0,0.2],[-0.25,-0.15],
    ])


    algorithm_param = {'max_num_iteration': 15,\
                    'population_size':5000,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

    optimization =  ga(function = objectiveFunction, dimension = 28, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=30)
    optimization.run()
    print(optimization)
    with open('optimization_outputs_ps1.txt', 'w') as f:
        print(optimization.output_dict, file=f)
    return optimization.output_dict




def graph_results():

    output_parameters = run_optimizer()
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
    output_parameters["variable"] = np.insert(output_parameters["variable"],9,-0.04)
    output_parameters["variable"] = np.insert(output_parameters["variable"],10,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],11,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],12,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],13,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],14,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],15,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],16,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],17,0)
    # fallow
    output_parameters["variable"] = np.insert(output_parameters["variable"],18,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],19,-0.14)
    output_parameters["variable"] = np.insert(output_parameters["variable"],20,0.38)
    output_parameters["variable"] = np.insert(output_parameters["variable"],21,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],22,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],23,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],24,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],25,0.07)
    output_parameters["variable"] = np.insert(output_parameters["variable"],26,0.12)

    # cattle
    output_parameters["variable"] = np.insert(output_parameters["variable"],36,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],37,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],38,0.80)
    output_parameters["variable"] = np.insert(output_parameters["variable"],39,-0.59)
    output_parameters["variable"] = np.insert(output_parameters["variable"],40,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],41,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],42,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],43,0.073)
    output_parameters["variable"] = np.insert(output_parameters["variable"],44,0.14)
    # red deer
    output_parameters["variable"] = np.insert(output_parameters["variable"],45,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],46,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],47,0.66)
    output_parameters["variable"] = np.insert(output_parameters["variable"],48,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],49,-0.4)
    output_parameters["variable"] = np.insert(output_parameters["variable"],50,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],51,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],52,0.052)
    output_parameters["variable"] = np.insert(output_parameters["variable"],53,0.083)

    # roe
    output_parameters["variable"] = np.insert(output_parameters["variable"],54,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],55,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],56,0.38)
    output_parameters["variable"] = np.insert(output_parameters["variable"],57,0) 
    output_parameters["variable"] = np.insert(output_parameters["variable"],58,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],59,-0.76)
    output_parameters["variable"] = np.insert(output_parameters["variable"],60,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],61,0.26)
    output_parameters["variable"] = np.insert(output_parameters["variable"],62,0.28)

    # pig
    output_parameters["variable"] = np.insert(output_parameters["variable"],63,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],64,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],65,0.34)
    output_parameters["variable"] = np.insert(output_parameters["variable"],66,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],67,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],68,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],69,-0.75)
    output_parameters["variable"] = np.insert(output_parameters["variable"],70,0.062)
    output_parameters["variable"] = np.insert(output_parameters["variable"],71,0.16)
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
    f.axes[8].vlines(x=15.75,ymin=3.3,ymax=3.7, color='r')
    f.axes[0].plot(15, 0.65, 'go',markersize=2.5) # and forcings
    f.axes[1].plot(15, 5.9, 'go',markersize=2.5) # and forcings
    f.axes[3].plot(15, 1.5, 'go',markersize=2.5) # and forcings
    f.axes[4].plot(15, 2.7, 'go',markersize=2.5) # and forcings
    f.axes[6].plot(15, 0.55, 'go',markersize=2.5) # and forcings
 
    # show plot
    f.fig.suptitle('Optimiser outputs: second parameter set')
    plt.tight_layout()
    plt.savefig('optimiser_outputs.png')
    plt.show()



    # now reality check 1
    t = np.linspace(2015, 2016, 10)
    X0 = [1, 1, 0, 1, 1, 1, 1, 0, 0] # no primary producers, only consumers
    realityCheck_ABC = solve_ivp(ecoNetwork, (2015, 2020), X0,  t_eval = t, args=(A, r), method = 'RK23') 
    realityCheck_1 = (np.vstack(np.hsplit(realityCheck_ABC.y.reshape(len(species), 10).transpose(),1)))
    realityCheck_1 = pd.DataFrame(data=realityCheck_1, columns=species)
    realityCheck_1['time'] = realityCheck_ABC.t
    # extract the node values from all dataframes
    final_runs1 = realityCheck_1.drop(['time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (10))
    # time 
    firstODEyears = np.repeat(realityCheck_1['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
    # calculate median 
    m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    perc2.name = "fivePerc"
    final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    g.fig.suptitle('Reality check: No primary producers')
    plt.tight_layout()
    plt.savefig('reality_check_noFood.png')
    plt.show()


    # now reality check 2
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
    # calculate median 
    m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    perc2.name = "fivePerc"
    final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    g.fig.suptitle('Reality check: No consumers')
    plt.tight_layout()
    plt.savefig('reality_check_noConsumers.png')
    plt.show()



    # now reality check 3
    t = np.linspace(2005, 2105, 100)
    X0 = [2, 2, 1, 2, 2, 2, 2, 1, 1] # overloaded consumers
    A[[0],0] = 0
    A[[1],1] = 0
    A[[3],3] = 0
    A[[4],4] = 0
    A[[5],5] = 0
    A[[6],6] = 0
    realityCheck_ABC = solve_ivp(ecoNetwork, (2005, 2105), X0,  t_eval = t, args=(A, r), method = 'RK23') 
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
    # calculate median 
    m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    perc2.name = "fivePerc"
    final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    g.fig.suptitle('Reality check: Overloaded consumers')
    plt.tight_layout()
    plt.savefig('reality_check_overloadConsumers.png')
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
    # calculate median 
    m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    perc2.name = "fivePerc"
    final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    g.fig.suptitle('Reality check: No woodland or consumers')
    plt.tight_layout()
    plt.savefig('reality_check_scrubLevelling.png')
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
    # calculate median 
    m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    perc2.name = "fivePerc"
    final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
    # graph it
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=3, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    g.fig.suptitle('Reality check: Grassland only')
    plt.tight_layout()
    plt.savefig('reality_check_grassLevelling.png')
    plt.show()




graph_results()
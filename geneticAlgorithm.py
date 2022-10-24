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
    starting_2009[6] = 1
    t_01 = np.linspace(4, 4.95, 3)
    second_ABC = solve_ivp(ecoNetwork, (4,4.95), starting_2009,  t_eval = t_01, args=(A, r), method = 'RK23')
    # 2010
    last_values_2009 = second_ABC.y[0:11, 2:3].flatten()
    starting_values_2010 = last_values_2009.copy()
    starting_values_2010[0] = 0.57
    # fallow deer reintroduced
    starting_values_2010[1] = 1
    starting_values_2010[3] = 1.5
    starting_values_2010[6] = 0.85
    t_1 = np.linspace(5, 5.95, 3)
    third_ABC = solve_ivp(ecoNetwork, (5,5.95), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
    # 2011
    last_values_2010 = third_ABC.y[0:11, 2:3].flatten()
    starting_values_2011 = last_values_2010.copy()
    starting_values_2011[0] = 0.65
    starting_values_2011[1] = 1.9
    starting_values_2011[3] = 1.7
    starting_values_2011[6] = 1.1
    t_2 = np.linspace(6, 6.95, 3)
    fourth_ABC = solve_ivp(ecoNetwork, (6,6.95), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
    # 2012
    last_values_2011 = fourth_ABC.y[0:11, 2:3].flatten()
    starting_values_2012 = last_values_2011.copy()
    starting_values_2012[0] = 0.74
    starting_values_2012[1] = 2.4
    starting_values_2012[3] = 2.2
    starting_values_2012[6] = 1.7
    t_3 = np.linspace(7, 7.95, 3)
    fifth_ABC = solve_ivp(ecoNetwork, (7,7.95), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
    # 2013
    last_values_2012 = fifth_ABC.y[0:11, 2:3].flatten()
    starting_values_2013 = last_values_2012.copy()
    starting_values_2013[0] = 0.43
    starting_values_2013[1] = 2.4
    starting_values_2013[3] = 2.4
    # red deer reintroduced
    starting_values_2013[4] = 1
    starting_values_2013[6] = 0.3
    t_4 = np.linspace(8, 8.95, 3)
    sixth_ABC = solve_ivp(ecoNetwork, (8,8.95), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
    # 2014
    last_values_2013 = sixth_ABC.y[0:11, 2:3].flatten()
    starting_values_2014 = last_values_2013.copy()
    starting_values_2014[0] = 0.44
    starting_values_2014[1] = 2.4
    starting_values_2014[3] = 5
    starting_values_2014[4] = 1
    starting_values_2014[6] = 0.9
    t_5 = np.linspace(9, 9.95, 3)
    seventh_ABC = solve_ivp(ecoNetwork, (9,9.95), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23') 
    # 2015
    last_values_2014 = seventh_ABC.y[0:11, 2:3].flatten()
    starting_values_2015 = last_values_2014
    starting_values_2015[0] = 0.43
    starting_values_2015[1] = 2.4
    starting_values_2015[3] = 2
    starting_values_2015[4] = 1
    starting_values_2015[6] = 0.71
    t_2015 = np.linspace(10, 10.95, 3)
    ABC_2015 = solve_ivp(ecoNetwork, (10,10.95), starting_values_2015,  t_eval = t_2015, args=(A, r), method = 'RK23')
    # 2016
    last_values_2015 = ABC_2015.y[0:11, 2:3].flatten()
    starting_values_2016 = last_values_2015.copy()
    starting_values_2016[0] = 0.48
    starting_values_2016[1] = 3.3
    starting_values_2016[3] = 1.7
    starting_values_2016[4] = 2
    starting_values_2016[6] = 0.7
    t_2016 = np.linspace(11, 11.95, 3)
    ABC_2016 = solve_ivp(ecoNetwork, (11,11.95), starting_values_2016,  t_eval = t_2016, args=(A, r), method = 'RK23')       
    # 2017
    last_values_2016 = ABC_2016.y[0:11, 2:3].flatten()
    starting_values_2017 = last_values_2016.copy()
    starting_values_2017[0] = 0.43
    starting_values_2017[1] = 3.9
    starting_values_2017[3] = 1.7
    starting_values_2017[4] = 1.1
    starting_values_2017[6] = 0.95
    t_2017 = np.linspace(12, 12.95, 3)
    ABC_2017 = solve_ivp(ecoNetwork, (12,12.95), starting_values_2017,  t_eval = t_2017, args=(A, r), method = 'RK23')     
    # 2018
    last_values_2017 = ABC_2017.y[0:11, 2:3].flatten()
    starting_values_2018 = last_values_2017.copy()
    # pretend bison were reintroduced (to estimate growth rate / interaction values)
    starting_values_2018[0] = 0
    starting_values_2018[1] = 6.0
    starting_values_2018[3] = 1.9
    starting_values_2018[4] = 1.9
    starting_values_2018[6] = 0.84
    t_2018 = np.linspace(13, 13.95, 3)
    ABC_2018 = solve_ivp(ecoNetwork, (13,13.95), starting_values_2018,  t_eval = t_2018, args=(A, r), method = 'RK23')     
    # 2019
    last_values_2018 = ABC_2018.y[0:11, 2:3].flatten()
    starting_values_2019 = last_values_2018.copy()
    starting_values_2019[0] = 0
    starting_values_2019[1] = 6.6
    starting_values_2019[3] = 1.7
    starting_values_2019[4] = 2.9
    starting_values_2019[6] = 0.44
    t_2019 = np.linspace(14, 14.95, 3)
    ABC_2019 = solve_ivp(ecoNetwork, (14,14.95), starting_values_2019,  t_eval = t_2019, args=(A, r), method = 'RK23')
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
    last_year_1 = y.loc[y['time'] == 3.95]
    last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
    last_year_2 = y_2.loc[y_2['time'] == 16]
    last_year_2 = last_year_2.drop('time', axis=1).values.flatten()
    return last_year_1, last_year_2, last_values_2015, last_values_2016, last_values_2017, last_values_2018, last_values_2019, y_2


# check for stability
def calcJacobian(A, r, n):
    # make an empty array to fill (with diagonals = 1, zeros elsewhere since we want eigenalue)
    i_matrix = np.eye(len(n))
    # put n into an array to multiply by A
    n_array = np.matlib.repmat(n, 1, len(n))
    n_array = np.reshape (n_array, (9,9))
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
    r =  x[0:9]
    # insert interaction matrices of 0
    # pony
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
    x = np.insert(x,21,0)
    x = np.insert(x,22,0)
    x = np.insert(x,23,0)
    x = np.insert(x,24,0)
    # cattle
    x = np.insert(x,36,0)
    x = np.insert(x,37,0)
    x = np.insert(x,40,0)
    x = np.insert(x,41,0)
    x = np.insert(x,42,0)
    # red deer
    x = np.insert(x,45,0)
    x = np.insert(x,46,0)
    x = np.insert(x,48,0)
    x = np.insert(x,50,0)
    x = np.insert(x,51,0)
    # roe
    x = np.insert(x,54,0)
    x = np.insert(x,55,0)
    x = np.insert(x,57,0) 
    x = np.insert(x,58,0)
    x = np.insert(x,60,0)
    # pig
    x = np.insert(x,63,0)
    x = np.insert(x,64,0)
    x = np.insert(x,66,0)
    x = np.insert(x,67,0)
    x = np.insert(x,68,0)
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
    # check viability of the parameter set (is it stable?)
    ia = np.linalg.pinv(A)
    # n is the equilibrium state; calc as inverse of -A*r
    n = -np.matmul(ia, r)
    isStable = calcStability(A, r, n)

    # if all the values of n are above zero at equilibrium, & if the parameter set is viable (stable & all n > 0 at equilibrium); do the calculation
    # if np.all(n > 0) &  isStable == True:
    # run the model
    last_year_1, last_year_2, last_values_2015, last_values_2016, last_values_2017, last_values_2018, last_values_2019, y_2 = run_model(X0, A, r)
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
        # 2016
        (((last_values_2016[0]-0.47)/0.47)**2) + 
        (((last_values_2016[1]-3.9)/3.9)**2) + 
        (((last_values_2016[3]-2.1)/2.1)**2) +
        (((last_values_2016[4]-2)/2)**2) +
        (((last_values_2016[6]-0.88)/0.88)**2) +
        # 2017
        (((last_values_2017[0]-0.44)/0.44)**2) + 
        (((last_values_2017[1]-6)/6)**2) +
        (((last_values_2017[3]-2.1)/2.1)**2) +
        (((last_values_2017[4]-1.1)/1.1)**2) +
        (((last_values_2017[6]-1.1)/1.1)**2) +
        # 2018
        (((last_values_2018[1]-7.5)/7.5)**2) +
        (((last_values_2018[3]-2.2)/2.2)**2) + 
        (((last_values_2018[4]-1.9)/1.9)**2) + 
        (((last_values_2018[6]-1.2)/1.2)**2) +
        # 2019
        (((last_values_2019[1]-8.3)/8.3)**2) + 
        (((last_values_2019[3]-2.1)/2.1)**2) + 
        (((last_values_2019[4]-2.9)/2.9)**2) + 
        (((last_values_2019[6]-0.5)/0.5)**2) + 
        # 2020
        (((last_year_2[0]-0.7)/0.7)**2) + 
        (((last_year_2[2]-0.3))**2) +
        (((last_year_2[5]-4.2)/4.2)**2) + 
        (((last_year_2[6]-1)/1)**2) + 
        (((last_year_2[7]-12.1)/12.1)**2) + 
        (((last_year_2[8]-3.5))**2))
    # print the output
    if result < 5:
        print(result)
    return (result)

    # else:
    #     return 1e5
    
#  PS1:                                                                          
#  [ 0.78682475  0.31741647  0.09996903 -0.05014727 -0.30114024  0.50259624
#   0.15556726  0.21236035 -0.00752402 -0.00426753 -0.99655814 -0.00269986
#  -0.00240701 -0.00305205 -0.01926811 -0.02339029 -0.07712044  0.73625849
#  -0.7074312   0.11399694  0.11607176  0.55157858 -0.75375631  0.13736587
#   0.24774307  0.40221588 -0.76514982  0.26742526  0.26136752  0.49424095
#  -0.81545783  0.10409119  0.10722544 -0.00855345 -0.00511527 -0.00421695
#  -0.00487159 -0.00507799 -0.02539618 -0.00774082 -0.02427517 -0.00415809
#  -0.00209986 -0.00304694 -0.00197189 -0.00140445 -0.00248006  0.00129173
#  -0.00552949]


def run_optimizer():
    bds = np.array([
        # growth
        [0.82,0.83],[0.44,0.45],[0.12,0.12],
        # exmoor pony
        [-0.04,-0.03],
        # fallow deer
        [-0.15,-0.14],[0.41,0.42],[0.084,0.085],[0.15,0.16],    
        # grassland parkland
        [-0.0033,-0.0032],[-0.0044,-0.0042],[-0.98,-0.97],[-0.0042,-0.0041],[-0.0019,-0.0018],[-0.00093,-0.00092],[-0.0059,-0.0058],[-0.034,-0.033],[-0.04,-0.034],
        # longhorn cattle
        [0.79,0.8],[-0.59,-0.58],[0.073,0.074],[0.13,0.14],
        # red deer
        [0.66,0.67],[-0.4,-0.39],[0.052,0.053],[0.082,0.083],
        # roe deer
        [0.59,0.6],[-0.76,-0.75],[0.26,0.27],[0.27,0.28],
        # tamworth pig  
        [0.34,0.35],[-0.75,-0.74],[0.062,0.063],[0.16,0.17],
        # thorny scrubland
        [-0.006,-0.006],[-0.00012,-0.00012],[-0.0033,-0.0033],[-0.007,-0.007],[-0.0029,-0.0029],[-0.0061,-0.0061],[-0.02,-0.02],[-0.042,-0.042],
        # woodland
        [-0.0026,-0.0025],[-0.0026,-0.0026],[-0.0011,-0.0011],[-0.0021,-0.0021],[-0.0045,-0.0045],[-0.0036,-0.0036],[0.0027,0.0027],[-0.008,0.0075],
    ])


    algorithm_param = {'max_num_iteration': 1,\
                    'population_size':1,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

    optimization =  ga(function = objectiveFunction, dimension = 49, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=30)
    optimization.run()
    print(optimization)
    with open('optimization_outputs.txt', 'w') as f:
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
    output_parameters["variable"] = np.insert(output_parameters["variable"],21,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],22,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],23,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],24,0)
    # cattle
    output_parameters["variable"] = np.insert(output_parameters["variable"],36,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],37,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],40,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],41,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],42,0)
    # red deer
    output_parameters["variable"] = np.insert(output_parameters["variable"],45,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],46,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],48,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],50,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],51,0)
    # roe
    output_parameters["variable"] = np.insert(output_parameters["variable"],54,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],55,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],57,0) 
    output_parameters["variable"] = np.insert(output_parameters["variable"],58,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],60,0)
    # pig
    output_parameters["variable"] = np.insert(output_parameters["variable"],63,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],64,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],66,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],67,0)
    output_parameters["variable"] = np.insert(output_parameters["variable"],68,0)
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
    f.axes[2].vlines(x=3.950,ymin=0.18,ymax=1, color='r')
    f.axes[5].vlines(x=3.950,ymin=1.7,ymax=3.3, color='r')
    f.axes[7].vlines(x=3.950,ymin=1,ymax=4.9, color='r')
    f.axes[8].vlines(x=3.950,ymin=1,ymax=2.9, color='r')
    # 2015
    f.axes[0].vlines(x=10.950,ymin=0.39,ymax=0.49, color='r')
    f.axes[1].vlines(x=10.950,ymin=2.7,ymax=3.9, color='r')
    f.axes[3].vlines(x=10.950,ymin=2,ymax=2.9, color='r')
    f.axes[4].vlines(x=10.950,ymin=0.6,ymax=1.4, color='r')
    f.axes[6].vlines(x=10.950,ymin=0.6,ymax=1.6, color='r')
    # 2016
    f.axes[0].vlines(x=11.950,ymin=0.43,ymax=0.5, color='r')
    f.axes[1].vlines(x=11.950,ymin=3.3,ymax=4.5, color='r')
    f.axes[3].vlines(x=11.950,ymin=1.6,ymax=2.5, color='r')
    f.axes[4].vlines(x=11.950,ymin=1.6,ymax=2.4, color='r')
    f.axes[6].vlines(x=11.950,ymin=0.35,ymax=1.4, color='r')
    # 2017
    f.axes[0].vlines(x=12.950,ymin=0.39,ymax=0.48, color='r')
    f.axes[1].vlines(x=12.950,ymin=5.4,ymax=6.6, color='r')
    f.axes[3].vlines(x=12.950,ymin=1.6,ymax=2.5, color='r')
    f.axes[4].vlines(x=12.950,ymin=0.7,ymax=1.5, color='r')
    f.axes[6].vlines(x=12.950,ymin=0.6,ymax=1.6, color='r')
    # 2018
    f.axes[1].vlines(x=13.950,ymin=6.9,ymax=8.1, color='r')
    f.axes[3].vlines(x=13.950,ymin=1.7,ymax=2.7, color='r')
    f.axes[4].vlines(x=13.950,ymin=1.5,ymax=2.2, color='r')
    f.axes[6].vlines(x=13.950,ymin=0.7,ymax=1.7, color='r')
    # 2019
    f.axes[1].vlines(x=14.950,ymin=7.7,ymax=8.9, color='r')
    f.axes[3].vlines(x=14.950,ymin=1.6,ymax=2.5, color='r')
    f.axes[4].vlines(x=14.950,ymin=2.5,ymax=3.2, color='r')
    f.axes[6].vlines(x=14.950,ymin=0,ymax=1, color='r')
    # 2020
    f.axes[0].vlines(x=15.950,ymin=0.61,ymax=0.7, color='r')
    f.axes[6].vlines(x=15.950,ymin=0.5,ymax=1.5, color='r')
    f.axes[2].vlines(x=15.950,ymin=0.18,ymax=0.41, color='r')
    f.axes[5].vlines(x=15.950,ymin=1.7,ymax=6.7, color='r')
    f.axes[7].vlines(x=15.950,ymin=9.7,ymax=14.4, color='r')
    f.axes[8].vlines(x=15.950,ymin=3.3,ymax=3.7, color='r')

    # show plot
    f.fig.suptitle('Optimiser outputs: first parameter set')

    plt.tight_layout()
    plt.savefig('optimiser_outputs.png')
    plt.show()

graph_results()
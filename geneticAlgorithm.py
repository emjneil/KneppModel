# ---- Genetic Algorithm ------
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
from geneticalgorithm import geneticalgorithm as ga


# # # # # --------- MINIMIZATION ----------- # # # # # # # 

species = ['arableGrass','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    X[X>1e3] = 1000
    # return array
    return X * (r + np.matmul(A, X))


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
    x = np.insert(x,44,0)
    x = np.insert(x,51,0)
    # define X0, growthRate, interactionMatrix
    X0 = [1,0,1,1,0,1,1]
    # growth rates
    interaction_strength = x[7:56]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(7,7),index = species, columns=species)
    # with pd.option_context('display.max_columns',None):
    #     print(interaction_strength)
    A = interaction_strength.to_numpy()
    # ODE1
    t_1 = np.linspace(0, 5, 50)
    results = solve_ivp(ecoNetwork, (0, 5), X0,  t_eval = t_1, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # ODE 2
    t = np.linspace(0, 1, 5)
    last_results = y[49:50,:].flatten()
    last_results[1] = 1
    last_results[4] = 1
    second_ABC = solve_ivp(ecoNetwork, (0, 1), last_results,  t_eval = t, args=(A, r), method = 'RK23')   
    # take those values and re-run for another year, adding forcings
    starting_2010 = second_ABC.y[0:7, 4:5]
    starting_values_2010 = starting_2010.flatten()
    starting_values_2010[1] = last_results[1]*2.0
    starting_values_2010[4] = last_results[4]*0.5
    # run the model for another year 2010-2011
    third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2011 = third_ABC.y[0:7, 4:5]
    starting_values_2011 = starting_2011.flatten()
    starting_values_2011[1] = last_results[1]*1.1
    starting_values_2011[4] = last_results[4]*1.3
    # run the model for 2011-2012
    fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2012 = fourth_ABC.y[0:7, 4:5]
    starting_values_2012 = starting_2012.flatten()
    starting_values_2012[1] = last_results[1]*1.1
    starting_values_2012[4] = last_results[4]*1.5
    # run the model for 2012-2013
    fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2013 = fifth_ABC.y[0:7, 4:5]
    starting_values_2013 = starting_2013.flatten()
    starting_values_2013[1] = last_results[1]*1.8
    starting_values_2013[4] = last_results[4]*0.18
    # run the model for 2011-2012
    sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2014 = sixth_ABC.y[0:7, 4:5]
    starting_values_2014 = starting_2014.flatten()
    starting_values_2014[1] = last_results[1]*0.6
    starting_values_2014[4] = last_results[4]*3
    # run the model for 2011-2012
    seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2015 = seventh_ABC.y[0:7, 4:5]
    starting_values_2015 = starting_2015.flatten()
    starting_values_2015[1] = last_results[1]*1.2
    starting_values_2015[4] = last_results[4]*0.5
    # run the model for 2011-2012
    eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2016 = eighth_ABC.y[0:7, 4:5]
    starting_values_2016 = starting_2016.flatten()
    starting_values_2016[1] = last_results[1]*1.21
    starting_values_2016[4] = last_results[4]*0.5
    # run the model for 2011-2012
    ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2016,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2017 = ninth_ABC.y[0:7, 4:5]
    starting_values_2017 = starting_2017.flatten()
    starting_values_2017[1] = np.random.uniform(low=0.56,high=2.0)
    starting_values_2017[4] = np.random.uniform(low=0.18,high=3)
    # run the model for 2011-2012
    tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2017,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2018 = tenth_ABC.y[0:7, 4:5]
    starting_values_2018 = starting_2018.flatten()
    starting_values_2018[1] = np.random.uniform(low=0.56,high=2.0)
    starting_values_2018[4] = np.random.uniform(low=0.18,high=3)
    # run the model for 2011-2012
    eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2018,  t_eval = t, args=(A, r), method = 'RK23')
    # concatenate & append all the runs
    combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y))
     # reshape the outputs
    y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters)
    print((y_2[49:50,:]))
    result = ((((y[49:50, 0]-0.86)**2) +  ((y[49:50, 2]-1.4)**2) + ((y[49:50, 3]-2.2)**2) + ((y[49:50, 5]-11.1)**2) + ((y[49:50, 6]-0.91)**2) + ((y_2[49:50, 0]-0.72)**2) +  ((y_2[49:50, 2]-2)**2) + ((y_2[49:50, 3]-4.1)**2) + ((y_2[49:50, 5]-28.8)**2) + ((y_2[49:50, 6]-0.9)**2)))
    # result = (((y_2[49:50, 0]-0.72)**2) +  ((y_2[49:50, 2]-2)**2) + ((y_2[49:50, 3]-4.1)**2) + ((y_2[49:50, 5]-28.8)**2) + ((y_2[49:50, 6]-0.9)**2))
    print(result)    
    return (result)


# order of outputs   
# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.86            1.4        2.2              7               0.91


bds = np.array([
    # growth
    [0.9,1],[0.2,0.25],[0,0.1],[0.2,0.25],[0.2,0.3],[0.8,0.85],[0.3,0.4],
    # interaction bds
    [-0.9,-0.8],[0.1,0.3],[0.1,0.25],[0.1,0.45],[-0.1,0],[-0.2,0],
    [0,0.1],[-1,-0.8],[0,0.1],[0,0.1],
    [0,1],[0,1],[-0.5,0],[0,1],[0,1],[0,1],[0,1],
    [0,0.5],[-0.9,-0.7],[0,0.5],[0,0.5],
    [0,0.1],[-0.9,-0.7],[0,0.1],[0,0.1],
    [-0.5,0.5],[-0.3,0],[-0.1,0],[-0.3,1],[-0.1,0],[-0.9,-0.7],
    [-0.5,0.5],[-0.7,-0.4],[-0.3,-0.01],[-0.5,-0.1],[0,0.5],[-0.6,-0.4]
])

algorithm_param = {'max_num_iteration': 5000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}


optimization =  ga(function = objectiveFunction, dimension = 44, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=20)
optimization.run()
print(optimization)

# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# download packages
# from scipy import integrate
from scipy.integrate import solve_ivp
# from scipy import optimize
from scipy.optimize import differential_evolution
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string
import timeit
import seaborn as sns
import random

##### ------------------- DEFINE THE FUNCTION ------------------------

# time the program
start = timeit.default_timer()


# define the number of simulations to try. Bode et al. ran a million
NUMBER_OF_SIMULATIONS = 500
# store species in a list
species = ['arableGrass','fox','largeHerb','organicCarbon','rabbits','roeDeer','tamworthPig','thornyScrub','woodland']


def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    # return array
    # return X * (r + np.matmul(A, X))
    return r * X + np.matmul(A, X) * X

# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():
    # pull the original sign of the interaction
    interaction_matrix = [
                [-0.18,0,-1,0,-0.39,-0.95,-1,-0.96,-0.19],
                [0,-0.09,0,0,0.19,0,0,0,0],
                [1,0,-1,0,0,0,0,1,1],
                [0.96,0.9,1,-0.18,0.15,0.52,1,0.2,0.56],
                [0.03,-0.06,0,0,-0.87,0,0,0.8,0.23],
                [0.6,0,-1,0,0,-0.17,0,0.63,0.44],
                [1,0,0,0,0,0,-1,0,0],
                [0,0,-1,0,-0.6,-0.32,-1,-0.09,-0.87],
                [0,0,-1,0,-0.32,-0.78,-1,-0.9,-0.67]
                ]

    # generate random uniform numbers
    variation = np.random.uniform(low = 0.5, high = 1.5, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # make lots of these arrays, half-to-double minimization outputs & consistent with sign
    return interaction_matrix


def generateGrowth():
    growthRates = [0.037, 0.17, 0, 0.002, 0.07, 0.05, 0, 0.16, 0.04]
    # multiply by a range
    variation = np.random.uniform(low = 0.5, high= 1.5, size = (len(species),))
    growth = growthRates * variation
    return growth
    
def generateX0():
    X0 = [0.00038, 0.000096, 0, 0.0001, 0.18, 0.00034, 0, 0.000022, 0.000067]
    variation = np.random.uniform(low = 0.5, high= 1.5, size = (len(species),))
    X0 = X0 * variation
    return X0


# # # # --- MAKE NETWORK X VISUAL OF INTERACTION MATRIX --

# def networkVisual():
    # import networkx as nx
    # interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
#     networkVisual = nx.DiGraph(interactionMatrix_csv)
#     # increase distance between nodes
#     pos = nx.circular_layout(networkVisual, scale=4)
#     # draw graph
#     plt.title('Ecological Network of the Knepp Estate')
#     # assign colors
#     nx.draw(networkVisual, pos, with_labels=True, font_size = 6, node_size = 4000, node_color = 'lightgray')
#     nx.draw(networkVisual.subgraph('beaver'), pos=pos, font_size=6, node_size = 4000, node_color='lightblue')
#     nx.draw(networkVisual.subgraph('largeHerb'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('tamworthPig'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     plt.show()
# networkVisual()


# # # # --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -------

def runODE_1():
    # Define time points: first 10 years (2000-2009)
    t = np.linspace(0, 10, 50)
    all_runs = []
    all_parameters = []
    for _ in range(NUMBER_OF_SIMULATIONS):
        A = generateInteractionMatrix()
        r = generateGrowth()
        X0 = generateX0()
        # remember the parameters used
        X0_growth = pd.concat([pd.DataFrame(X0), pd.DataFrame(r)], axis = 1)
        X0_growth.columns = ['X0','growth']
        parameters_used = pd.concat([X0_growth, pd.DataFrame(A, index = species, columns = species)])
        all_parameters.append(parameters_used)
        # run the ODE
        first_ABC = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(A, r), method = 'RK23')
        # append all the runs
        all_runs = np.append(all_runs, first_ABC.y)
        # check the final runs
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 50).transpose(),NUMBER_OF_SIMULATIONS)))
    final_runs = pd.DataFrame(data=final_runs, columns=species)
    # append all the parameters to a dataframe
    all_parameters = pd.concat(all_parameters)
    # add ID to all_parameters
    all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])
    return final_runs,all_parameters, X0


# --------- FILTER OUT UNREALISTIC RUNS -----------

def filterRuns_1():
    final_runs,all_parameters,X0 = runODE_1()
    # add ID to dataframe
    IDs = np.arange(1,1 + NUMBER_OF_SIMULATIONS)
    final_runs['ID'] = np.repeat(IDs,50)
    # select only the last run (filter to the year 2009)
    accepted_year = final_runs.iloc[49::50, :]
    # print that
    with pd.option_context('display.max_columns',None):
        print(accepted_year*10000)
    # add filtering criteria 
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= 9/10000) & (accepted_year['roeDeer'] >= 4.9/10000) &
    (accepted_year['fox'] <= 13.9/10000) & (accepted_year['fox'] >= 0.22/10000) &
    (accepted_year['rabbits'] <= 2260/10000) & (accepted_year['rabbits'] >= 0/10000) &
    (accepted_year['arableGrass'] <= 4/10000) & (accepted_year['arableGrass'] >= 3.1/10000) &
    (accepted_year['woodland'] <=0.8/10000) & (accepted_year['woodland'] >= 0.2/10000) &
    (accepted_year['thornyScrub'] <= 0.9/10000) & (accepted_year['thornyScrub'] >= 0.05/10000) &
    (accepted_year['organicCarbon'] <= 2/10000) & (accepted_year['organicCarbon'] >= 1/10000)
    ]
    print(accepted_simulations.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # add accepted ID to original dataframe
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Yes', 'No')
    return accepted_parameters, accepted_simulations, final_runs




# # # # ---------------------- ODE #2: Years 2009-2018 -------------------------

def generateParameters2():
    accepted_parameters, accepted_simulations, final_runs = filterRuns_1()
    # # select growth rates 
    growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    r = growthRates_2.to_numpy()
    # make the final runs of ODE #1 the initial conditions
    accepted_simulations = accepted_simulations.drop('ID', axis=1)
    accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']] = accepted_simulations.values.flatten()
    # select X0 
    X0_2 = accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']]
    X0_2 = pd.DataFrame(X0_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    # # add reintroduced species
    X0_2.loc[:, 'largeHerb'] = [26.5/10000 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [7.9/10000 for i in X0_2.index]
    X0_2 = X0_2.to_numpy()
    # # select interaction matrices part of the dataframes 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    # make the interactions with the reintroduced species random between -1 and 1 (depending on sign)
    # rows
    herbRows = interaction_strength_2.loc[interaction_strength_2['largeHerb'] > 0, 'largeHerb'] 
    interaction_strength_2.loc[interaction_strength_2['largeHerb'] > 0, 'largeHerb'] = [np.random.uniform(low=0, high=1) for i in herbRows.index]
    tamRows = interaction_strength_2.loc[interaction_strength_2['tamworthPig'] > 0, 'tamworthPig'] 
    interaction_strength_2.loc[interaction_strength_2['tamworthPig'] > 0, 'tamworthPig'] = [np.random.uniform(low=0, high=1) for i in tamRows.index]
    # columns
    interaction_strength_2.loc['largeHerb','thornyScrub']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['largeHerb','arableGrass']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['largeHerb','woodland']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['tamworthPig','thornyScrub']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['tamworthPig','arableGrass']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['tamworthPig','woodland']  = [np.random.uniform(low=-1, high=0)]
    A = interaction_strength_2.to_numpy()
    return r, X0_2, A, accepted_simulations, accepted_parameters, final_runs




# # # # # ------ SOLVE ODE #2: Pre-reintroductions (2009-2018) -------

def runODE_2():
    r, X0_2, A, accepted_simulations, accepted_parameters, final_runs  = generateParameters2()
    all_runs_2 = []
    all_parameters_2 = []
    t_2 = np.linspace(0, 1, 5)
    # loop through each row of accepted parameters
    for X0_2, r, A in zip(X0_2,r, np.array_split(A,len(accepted_simulations))):
        # concantenate the parameters
        X0_growth_2 = pd.concat([pd.DataFrame(X0_2), pd.DataFrame(r)], axis = 1)
        X0_growth_2.columns = ['X0','growth']
        parameters_used_2 = pd.concat([X0_growth_2, pd.DataFrame(A, index = species, columns = species)])
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        second_ABC = solve_ivp(ecoNetwork, (0, 1), X0_2,  t_eval = t_2, args=(A, r), method = 'RK23')        
        # take those values and re-run for another year, adding forcings
        starting_2010 = second_ABC.y[0:9, 4:5]
        starting_values_2010 = starting_2010.flatten()
        starting_values_2010[2] = 41.8/10000
        starting_values_2010[6] = 3.8/10000
        # run the model for another year 2010-2011
        third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2011 = third_ABC.y[0:9, 4:5]
        starting_values_2011 = starting_2011.flatten()
        starting_values_2011[2] = 47.6/10000
        starting_values_2011[6] = 4.9/10000
        # run the model for 2011-2012
        fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2012 = fourth_ABC.y[0:9, 4:5]
        starting_values_2012 = starting_2012.flatten()
        starting_values_2012[2] = 55.3/10000
        starting_values_2012[6] = 7.4/10000
        # run the model for 2012-2013
        fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2013 = fifth_ABC.y[0:9, 4:5]
        starting_values_2013 = starting_2013.flatten()
        starting_values_2013[2] = 87/10000
        starting_values_2013[6] = 1.3/10000
        # run the model for 2011-2012
        sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2014 = sixth_ABC.y[0:9, 4:5]
        starting_values_2014 = starting_2014.flatten()
        starting_values_2014[2] = 52.4/10000
        starting_values_2014[6] = 4/10000
        # run the model for 2011-2012
        seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2015 = seventh_ABC.y[0:9, 4:5]
        starting_values_2015 = starting_2015.flatten()
        starting_values_2015[2] = 60.7/10000
        starting_values_2015[6] = 2/10000
        # run the model for 2011-2012
        eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2016 = eighth_ABC.y[0:9, 4:5]
        starting_values_2016 = starting_2016.flatten()
        starting_values_2016[2] = 63.6/10000
        starting_values_2016[6] = 1.6/10000
        # run the model for 2011-2012
        ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2016,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2017 = ninth_ABC.y[0:9, 4:5]
        starting_values_2017 = starting_2017.flatten()
        starting_values_2017[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_values_2017[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2017,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2018 = tenth_ABC.y[0:9, 4:5]
        starting_values_2018 = starting_2018.flatten()
        starting_values_2018[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_values_2018[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2018,  t_eval = t_2, args=(A, r), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y))
        # print(combined_runs)
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)   
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 50).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    # append all the parameters to a dataframe
    all_parameters_2 = pd.concat(all_parameters_2)
    # add ID to the dataframe & parameters
    all_parameters_2['ID'] = ([(x+1) for x in range(len(accepted_simulations)) for _ in range(len(parameters_used_2))])
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,50)
    return final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = runODE_2()
    # select only the last run (filter to the year 2010); make sure these are in line with the new min/max X0 values (from the minimization), not the original X0 bounds
    accepted_year_2018 = final_runs_2.iloc[49::50, :]
    with pd.option_context('display.max_columns',None):
        print(accepted_year_2018*10000)
    accepted_simulations_2018 = accepted_year_2018[
    (accepted_year_2018['roeDeer'] <= 18/10000) & (accepted_year_2018['roeDeer'] >= 4.5/10000) &
    (accepted_year_2018['fox'] <= 13.9/10000) & (accepted_year_2018['fox'] >= 0.22/10000) &
    (accepted_year_2018['rabbits'] <= 2260/10000) & (accepted_year_2018['rabbits'] >= 1/10000) &
    (accepted_year_2018['arableGrass'] <= 2.7/10000) & (accepted_year_2018['arableGrass'] >= 2.5/10000) &
    (accepted_year_2018['woodland'] <= 0.8/10000) & (accepted_year_2018['woodland'] >= 0.3/10000) &
    (accepted_year_2018['thornyScrub'] <= 1.6/10000) & (accepted_year_2018['thornyScrub'] >= 0.9/10000) & 
    (accepted_year_2018['organicCarbon'] <= 2.3/10000) & (accepted_year_2018['organicCarbon'] >= 1.8/10000)
    ]
    print(accepted_simulations_2018.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2018 = all_parameters_2[all_parameters_2['ID'].isin(accepted_simulations_2018['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2018['ID']), 'Yes', 'No')
    return accepted_simulations_2018, accepted_parameters_2018, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs







# # # # ---------------------- ODE #3: Years 2018-2043 -------------------------

def generateParameters3():
    accepted_simulations_2018, accepted_parameters_2018, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = filterRuns_2()
    # # select growth rates 
    growthRates_3 = accepted_parameters_2018.loc[accepted_parameters_2018['growth'].notnull(), ['growth']]
    growthRates_3 = pd.DataFrame(growthRates_3.values.reshape(len(accepted_simulations_2018), len(species)), columns = species)
    r = growthRates_3.to_numpy()
    # make the final runs of ODE #2 the initial conditions
    accepted_simulations_2 = accepted_simulations_2018.drop('ID', axis=1)
    accepted_parameters_2018.loc[accepted_parameters_2018['X0'].notnull(), ['X0']] = accepted_simulations_2.values.flatten()
    # select X0 
    X0_3 = accepted_parameters_2018.loc[accepted_parameters_2018['X0'].notnull(), ['X0']]
    X0_3 = pd.DataFrame(X0_3.values.reshape(len(accepted_simulations_2), len(species)), columns = species)
    # # add reintroduced species
    X0_3.loc[:, 'largeHerb'] = [np.random.uniform(low=26.5/10000,high=87/10000) for i in X0_3.index]
    X0_3.loc[:,'tamworthPig'] = [np.random.uniform(low=1.3/10000,high=7.9/10000) for i in X0_3.index]
    X0_3 = X0_3.to_numpy()
    # # select interaction matrices part of the dataframes 
    interaction_strength_3 = accepted_parameters_2018.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_3 = interaction_strength_3.dropna()
    A = interaction_strength_3.to_numpy()
    return r, X0_3, A, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs



# # # # # ------ SOLVE ODE #3: Projecting forwards 10 years (2018-2028) -------

def runODE_3():
    r, X0_3, A, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs  = generateParameters3()
    all_runs_3 = []
    all_parameters_3 = []
    t = np.linspace(0, 1, 5)
    # loop through each row of accepted parameters
    for X0_3, r, A in zip(X0_3,r, np.array_split(A,len(accepted_simulations_2))):
        # concantenate the parameters
        X0_growth_3 = pd.concat([pd.DataFrame(X0_3), pd.DataFrame(r)], axis = 1)
        X0_growth_3.columns = ['X0','growth']
        parameters_used_3 = pd.concat([X0_growth_3, pd.DataFrame(A, index = species, columns = species)])
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        third_ABC = solve_ivp(ecoNetwork, (0, 1), X0_3,  t_eval = t, args=(A, r), method = 'RK23')        
        # take those values and re-run for another year, adding forcings
        starting_2019 = third_ABC.y[0:9, 4:5]
        starting_2019 = starting_2019.flatten()
        starting_2019[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2019[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for another year 2010-2011
        fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2019,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2020 = fourth_ABC.y[0:9, 4:5]
        starting_2020 = starting_2020.flatten()
        starting_2020[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2020[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2020,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2021 = fifth_ABC.y[0:9, 4:5]
        starting_2021 = starting_2021.flatten()
        starting_2021[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2021[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2012-2013
        sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2021,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2022 = sixth_ABC.y[0:9, 4:5]
        starting_2022 = starting_2022.flatten()
        starting_2022[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2022[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2022,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2023 = seventh_ABC.y[0:9, 4:5]
        starting_2023 = starting_2023.flatten()
        starting_2023[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2023[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2023,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2024 = eighth_ABC.y[0:9, 4:5]
        starting_2024 = starting_2024.flatten()
        starting_2024[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2024[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2024,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2025 = ninth_ABC.y[0:9, 4:5]
        starting_2025 = starting_2025.flatten()
        starting_2025[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2025[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2025,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2026 = tenth_ABC.y[0:9, 4:5]
        starting_2026 = starting_2026.flatten()
        starting_2026[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2026[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2026,  t_eval = t, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2027 = eleventh_ABC.y[0:9, 4:5]
        starting_2027 = starting_2027.flatten()
        starting_2027[2] = np.random.uniform(low=26.5/10000,high=87/10000)
        starting_2027[6] = np.random.uniform(low=1.3/10000,high=7.9/10000)
        # run the model for 2011-2012
        twelfth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2027,  t_eval = t, args=(A, r), method = 'RK23')
        # concatenate & append all the runs
        combined_runs_2 = np.hstack((third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y, twelfth_ABC.y))
        # print(combined_runs)
        all_runs_3 = np.append(all_runs_3, combined_runs_2)
        # append all the parameters
        all_parameters_3.append(parameters_used_3)   
    # check the final runs
    final_runs_3 = (np.vstack(np.hsplit(all_runs_3.reshape(len(species)*len(accepted_simulations_2), 50).transpose(),len(accepted_simulations_2))))
    final_runs_3 = pd.DataFrame(data=final_runs_3, columns=species)
    print(final_runs_3)
    # append all the parameters to a dataframe
    all_parameters_3 = pd.concat(all_parameters_3)
    # add ID to the dataframe & parameters
    all_parameters_3['ID'] = ([(x+1) for x in range(len(accepted_simulations_2)) for _ in range(len(parameters_used_3))])
    IDs = np.arange(1,1 + len(accepted_simulations_2))
    final_runs_3['ID'] = np.repeat(IDs,50)
    final_runs_3['accepted?'] = np.repeat('Projection', len(final_runs_3))
    return final_runs_3, all_parameters_3, X0_3, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    final_runs_3, all_parameters_3, X0_3, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = runODE_3()
    # extract accepted nodes from all dataframes
    accepted_shape1 = np.repeat(final_runs['accepted?'], len(species))
    accepted_shape2 = np.repeat(final_runs_2['accepted?'], len(species))
    accepted_shape3 = np.repeat(final_runs_3['accepted?'], len(species))
    # concatenate them
    accepted_shape = pd.concat([accepted_shape1, accepted_shape2, accepted_shape3])
    # extract the node values from all dataframes
    final_runs1 = final_runs.drop(['ID', 'accepted?'], axis=1)
    final_runs1 = final_runs1.values.flatten()
    final_runs2 = final_runs_2.drop(['ID', 'accepted?'], axis=1)
    final_runs2 = final_runs2.values.flatten()
    final_runs3 = final_runs_3.drop(['ID', 'accepted?'], axis=1)
    final_runs3 = final_runs3.values.flatten()
    # concatenate them
    y_values = np.concatenate((final_runs1, final_runs2, final_runs3), axis=None)
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_firstRun = np.tile(species, 50*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 50*len(accepted_simulations))
    species_thirdRun = np.tile(species, 50*len(accepted_simulations_2))
    species_list = np.concatenate((species_firstRun, species_secondRun, species_thirdRun), axis=None)
    # add a grouping variable to graph each run separately
    grouping1 = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    grouping_variable1 = np.repeat(grouping1,50*len(species))
    grouping2 = np.arange(NUMBER_OF_SIMULATIONS+2, NUMBER_OF_SIMULATIONS + 2 + len(accepted_simulations))
    grouping_variable2 = np.repeat(grouping2,50*len(species))
    grouping3 = np.arange(NUMBER_OF_SIMULATIONS+2, NUMBER_OF_SIMULATIONS + 2 + len(accepted_simulations_2))
    grouping_variable3 = np.repeat(grouping3,50*len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping_variable1, grouping_variable2, grouping_variable3), axis=None)
    # years - we've looked at 19 so far (2000-2018)
    year = np.arange(1,11)
    year2 = np.arange(11,21)
    year3 = np.arange(21,31)
    indices1 = np.repeat(year,len(species)*5)
    indices1 = np.tile(indices1, NUMBER_OF_SIMULATIONS)
    indices2 = np.repeat(year2,len(species)*5)
    indices2 = np.tile(indices2, len(accepted_simulations))
    indices3 = np.repeat(year3,len(species)*5)
    indices3 = np.tile(indices3, len(accepted_simulations_2))
    indices = np.concatenate((indices1, indices2, indices3), axis=None)
    # put it in a dataframe; scale or unscale it here (e.g. y_values * 10k)
    final_df = pd.DataFrame(
        {'nodeValues': y_values, 'runNumber': grouping_variable, 'species': species_list, 'time': indices, 'runAccepted': accepted_shape})
    # color palette
    palette = dict(zip(final_df.runAccepted.unique(),
                    sns.color_palette("mako", 3)))
    # plot
    sns.relplot(x="time", y="nodeValues",
                hue="runAccepted", col="species",
                height=2.5, aspect=.75, facet_kws=dict(sharey=False),
                kind="line", legend="full", palette=palette, col_wrap=3, data=final_df)
    plt.xticks([0, 10, 20, 30])
    # stop the plots from overlapping
    plt.tight_layout()
    plt.show()


plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
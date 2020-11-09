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

# define the number of simulations to try
NUMBER_OF_SIMULATIONS = 1000
# store species in a list
species = ['grasslandParkland','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    X[X<1e-6] = 0
    return X * (r + np.matmul(A, X))


# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():
    # define the array
    interaction_matrix = [
                [-0.74,0.33,0,0.11,0.2,-0.05,-0.009],
                [0.005,-0.33,0,0,0,0.007,0.002],
                [0.37,0.04,-0.93,0.16,0.06,0.02,0.4],
                [0.22,0,0,-0.83,0,0.08,0.36],
                [0.004,0,0,0,-0.07,0.006,0.003],
                [-0.03,-0.05,0,-0.08,-0.006,-0.009,-0.05],
                [-0.06,-0.02,0,-0.07,-0.09,0.01,-0.07]
                ]
    # generate random uniform numbers
    variation = np.random.uniform(low = 0.9, high = 1.1, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix


def generateGrowth():
    growthRates = [0.91, 0.22, 0.04, 0.49, 0.15, 0.74, 0.18]
    # multiply by a range
    variation = np.random.uniform(low = 0.9, high= 1.1, size = (len(species),))
    growth = growthRates * variation
    return growth
    
def generateX0():
    # initially scale everything to abundance of zero (except species to be reintroduced)
    X0 = [1, 0, 1, 1, 0, 1, 1]
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
    # Define time points: first 5 years (2005-2009)
    t = np.linspace(0, 5, 50)
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
        first_ABC = solve_ivp(ecoNetwork, (0, 5), X0,  t_eval = t, args=(A, r), method = 'RK23')
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
        print(accepted_year)
    # add filtering criteria 
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= (X0[3]*3.3)) & (accepted_year['roeDeer'] >= (X0[3]*1)) &
    (accepted_year['grasslandParkland'] <= (X0[0]*1)) & (accepted_year['grasslandParkland'] >= (X0[0]*0.7)) &
    (accepted_year['woodland'] <=(X0[6]*1.3)) & (accepted_year['woodland'] >= (X0[6]*0.6)) &
    (accepted_year['thornyScrub'] <= (X0[5]*20.9)) & (accepted_year['thornyScrub'] >= (X0[5]*1)) &
    (accepted_year['organicCarbon'] <= (X0[2]*1.9)) & (accepted_year['organicCarbon'] >= (X0[2]*0.95))
    ]
    print(accepted_simulations.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # add accepted ID to original dataframe
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Accepted', 'Rejected')
    # with pd.option_context('display.max_columns',None):
    #     print(final_runs)
    return accepted_parameters, accepted_simulations, final_runs, X0




# # # # ---------------------- ODE #2: Years 2009-2018 -------------------------

def generateParameters2():
    accepted_parameters, accepted_simulations, final_runs, X0 = filterRuns_1()
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
    X0_2.loc[:, 'largeHerb'] = [1 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [1 for i in X0_2.index]
    X0_2 = X0_2.to_numpy()
    # # select interaction matrices part of the dataframes 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    # turn to array
    A = interaction_strength_2.to_numpy()
    return r, X0_2, A, accepted_simulations, accepted_parameters, final_runs, X0




# # # # # ------ SOLVE ODE #2: Pre-reintroductions (2009-2018) -------

def runODE_2():
    r, X0_2, A, accepted_simulations, accepted_parameters, final_runs, X0  = generateParameters2()
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
        starting_2010 = second_ABC.y[0:7, 4:5]
        starting_values_2010 = starting_2010.flatten()
        starting_values_2010[1] = X0_2[1]*2.0
        starting_values_2010[4] = X0_2[4]*0.5
        # run the model for another year 2010-2011
        third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2011 = third_ABC.y[0:7, 4:5]
        starting_values_2011 = starting_2011.flatten()
        starting_values_2011[1] = X0_2[1]*1.1
        starting_values_2011[4] = X0_2[4]*1.3
        # run the model for 2011-2012
        fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2012 = fourth_ABC.y[0:7, 4:5]
        starting_values_2012 = starting_2012.flatten()
        starting_values_2012[1] = X0_2[1]*1.1
        starting_values_2012[4] = X0_2[4]*1.5
        # run the model for 2012-2013
        fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2013 = fifth_ABC.y[0:7, 4:5]
        starting_values_2013 = starting_2013.flatten()
        starting_values_2013[1] = X0_2[1]*1.8
        starting_values_2013[4] = X0_2[4]*0.18
        # run the model for 2011-2012
        sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2014 = sixth_ABC.y[0:7, 4:5]
        starting_values_2014 = starting_2014.flatten()
        starting_values_2014[1] = X0_2[1]*0.6
        starting_values_2014[4] = X0_2[4]*3
        # run the model for 2011-2012
        seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2015 = seventh_ABC.y[0:7, 4:5]
        starting_values_2015 = starting_2015.flatten()
        starting_values_2015[1] = X0_2[1]*1.2
        starting_values_2015[4] = X0_2[4]*0.5
        # run the model for 2011-2012
        eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2016 = eighth_ABC.y[0:7, 4:5]
        starting_values_2016 = starting_2016.flatten()
        starting_values_2016[1] = X0_2[1]*1.21
        starting_values_2016[4] = X0_2[4]*0.5
        # run the model for 2011-2012
        ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2016,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2017 = ninth_ABC.y[0:7, 4:5]
        starting_values_2017 = starting_2017.flatten()
        starting_values_2017[1] = np.random.uniform(low=0.56,high=2.0)
        starting_values_2017[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2011-2012
        tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2017,  t_eval = t_2, args=(A, r), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2018 = tenth_ABC.y[0:7, 4:5]
        starting_values_2018 = starting_2018.flatten()
        starting_values_2018[1] = np.random.uniform(low=0.56,high=2.0)
        starting_values_2018[4] = np.random.uniform(low=0.18,high=3)
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
    return final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs, X0




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs, X0 = runODE_2()
    # select only the last run (filter to the year 2010); make sure these are in line with the new min/max X0 values (from the minimization), not the original X0 bounds
    accepted_year_2018 = final_runs_2.iloc[49::50, :]
    with pd.option_context('display.max_columns',None):
        print(accepted_year_2018)
    accepted_simulations_2018 = accepted_year_2018[
    (accepted_year_2018['roeDeer'] <= X0[3]*6.7) & (accepted_year_2018['roeDeer'] >= X0[3]*1.7) &
    (accepted_year_2018['grasslandParkland'] <= X0[0]*0.86) & (accepted_year_2018['grasslandParkland'] >= X0[0]*0.59) &
    (accepted_year_2018['woodland'] <= X0[6]*1.3) & (accepted_year_2018['woodland'] >= X0[6]*0.6) &
    (accepted_year_2018['thornyScrub'] <= X0[5]*35.1) & (accepted_year_2018['thornyScrub'] >= X0[5]*22.5) & 
    (accepted_year_2018['organicCarbon'] <= X0[2]*2.2) & (accepted_year_2018['organicCarbon'] >= X0[2]*1.7)
    ]
    print(accepted_simulations_2018.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2018 = all_parameters_2[all_parameters_2['ID'].isin(accepted_simulations_2018['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2018['ID']), 'Accepted', 'Rejected')
    # final_runs_2.to_csv('finalRuns_bothODEs_50%.csv')
    return accepted_simulations_2018, accepted_parameters_2018, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs




# # # # ---------------------- ODE #3: projecting 10 years (2018-2028) -------------------------

def generateParameters3():
    accepted_simulations_2018, accepted_parameters_2018, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = filterRuns_2()
    # # select growth rates 
    growthRates_3 = accepted_parameters_2018.loc[accepted_parameters_2018['growth'].notnull(), ['growth']]
    growthRates_3 = pd.DataFrame(growthRates_3.values.reshape(len(accepted_simulations_2018), len(species)), columns = species)
    r_3 = growthRates_3.to_numpy()
    # make the final runs of ODE #2 the initial conditions
    accepted_simulations_2 = accepted_simulations_2018.drop('ID', axis=1)
    accepted_parameters_2018.loc[accepted_parameters_2018['X0'].notnull(), ['X0']] = accepted_simulations_2.values.flatten()
    # select X0 
    X0_3 = accepted_parameters_2018.loc[accepted_parameters_2018['X0'].notnull(), ['X0']]
    X0_3 = pd.DataFrame(X0_3.values.reshape(len(accepted_simulations_2), len(species)), columns = species)
    X0_3 = X0_3.to_numpy()
    # # select interaction matrices part of the dataframes 
    interaction_strength_3 = accepted_parameters_2018.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_3 = interaction_strength_3.dropna()
    A_3 = interaction_strength_3.to_numpy()
    return r_3, X0_3, A_3, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs



# # # # # ------ SOLVE ODE #3: Projecting forwards 10 years (2018-2028) -------

def runODE_3():
    r_3, X0_3, A_3, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs  = generateParameters3()
    all_runs_3 = []
    all_parameters_3 = []
    t = np.linspace(0, 1, 5)
    # loop through each row of accepted parameters
    for X0_3, r_3, A_3 in zip(X0_3,r_3, np.array_split(A_3,len(accepted_simulations_2))):
        # concantenate the parameters
        X0_growth_3 = pd.concat([pd.DataFrame(X0_3), pd.DataFrame(r_3)], axis = 1)
        X0_growth_3.columns = ['X0','growth']
        parameters_used_3 = pd.concat([X0_growth_3, pd.DataFrame(A_3, index = species, columns = species)])
        # run the model for one year 2018-2019 (to account for herbivore numbers being manually controlled every year)
        third_ABC = solve_ivp(ecoNetwork, (0, 1), X0_3,  t_eval = t, args=(A_3, r_3), method = 'RK23')        
        # take those values and re-run for another year, adding forcings
        starting_2019 = third_ABC.y[0:7, 4:5]
        starting_2019 = starting_2019.flatten()
        starting_2019[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2019[4] =  np.random.uniform(low=0.18,high=3)
        # run the model for another year 2020
        fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2019,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2020 = fourth_ABC.y[0:9, 4:5]
        starting_2020 = starting_2020.flatten()
        starting_2020[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2020[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2021
        fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2020,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2021 = fifth_ABC.y[0:9, 4:5]
        starting_2021 = starting_2021.flatten()
        starting_2021[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2021[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2022
        sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2021,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2022 = sixth_ABC.y[0:9, 4:5]
        starting_2022 = starting_2022.flatten()
        starting_2022[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2022[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2023
        seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2022,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2023 = seventh_ABC.y[0:9, 4:5]
        starting_2023 = starting_2023.flatten()
        starting_2023[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2023[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2024
        eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2023,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2024 = eighth_ABC.y[0:9, 4:5]
        starting_2024 = starting_2024.flatten()
        starting_2024[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2024[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2011-2012
        ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2024,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2025 = ninth_ABC.y[0:9, 4:5]
        starting_2025 = starting_2025.flatten()
        starting_2025[1] =np.random.uniform(low=0.56,high=2.0)
        starting_2025[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2011-2012
        tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2025,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2026 = tenth_ABC.y[0:9, 4:5]
        starting_2026 = starting_2026.flatten()
        starting_2026[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2026[4] = np.random.uniform(low=0.18,high=3)
        # run the model for 2011-2012
        eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2026,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2027 = eleventh_ABC.y[0:9, 4:5]
        starting_2027 = starting_2027.flatten()
        starting_2027[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2027[4] = np.random.uniform(low=0.18,high=3)
        # run the model to 2028
        twelfth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2027,  t_eval = t, args=(A_3, r_3), method = 'RK23')

        # just to check longer (up to 25 yrs)
        starting_2028 = twelfth_ABC.y[0:9, 4:5]
        starting_2028 = starting_2028.flatten()
        starting_2028[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2028[4] = np.random.uniform(low=0.18,high=3)
        # 2029
        thirteenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2028,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2029 = thirteenth_ABC.y[0:9, 4:5]
        starting_2029 = starting_2029.flatten()
        starting_2029[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2029[4] = np.random.uniform(low=0.18,high=3)
        # 2030
        fourteenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2029,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2030 = fourteenth_ABC.y[0:9, 4:5]
        starting_2030 = starting_2030.flatten()
        starting_2030[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2030[4] = np.random.uniform(low=0.18,high=3)
        # 2031
        fifteenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2030,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2031 = fifteenth_ABC.y[0:9, 4:5]
        starting_2031 = starting_2031.flatten()
        starting_2031[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2031[4] = np.random.uniform(low=0.18,high=3)
        # 2032
        sixteenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2031,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2032 = sixteenth_ABC.y[0:9, 4:5]
        starting_2032 = starting_2032.flatten()
        starting_2032[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2032[4] = np.random.uniform(low=0.18,high=3)
        # 2033
        seventeenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2032,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2033 = seventeenth_ABC.y[0:9, 4:5]
        starting_2033 = starting_2033.flatten()
        starting_2033[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2033[4] = np.random.uniform(low=0.18,high=3)
        # 2034
        eighteenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2033,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2034 = eighteenth_ABC.y[0:9, 4:5]
        starting_2034 = starting_2034.flatten()
        starting_2034[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2034[4] = np.random.uniform(low=0.18,high=3)
        # 2035
        nineteenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2034,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2035 = nineteenth_ABC.y[0:9, 4:5]
        starting_2035 = starting_2035.flatten()
        starting_2035[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2035[4] = np.random.uniform(low=0.18,high=3)
        # 2036
        twentieth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2035,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2036 = twentieth_ABC.y[0:9, 4:5]
        starting_2036 = starting_2036.flatten()
        starting_2036[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2036[4] = np.random.uniform(low=0.18,high=3)
        # 2037
        twentyone_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2036,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2037 = twentyone_ABC.y[0:9, 4:5]
        starting_2037 = starting_2037.flatten()
        starting_2037[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2037[4] = np.random.uniform(low=0.18,high=3)
        # 2038
        twentytwo_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2037,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2038 = twentytwo_ABC.y[0:9, 4:5]
        starting_2038 = starting_2038.flatten()
        starting_2038[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2038[4] = np.random.uniform(low=0.18,high=3)
        # 2039
        twentythree_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2038,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2039 = twentythree_ABC.y[0:9, 4:5]
        starting_2039 = starting_2039.flatten()
        starting_2039[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2039[4] = np.random.uniform(low=0.18,high=3)
        # 2040
        twentyfour_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2039,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2040 = twentyfour_ABC.y[0:9, 4:5]
        starting_2040 = starting_2040.flatten()
        starting_2040[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2040[4] = np.random.uniform(low=0.18,high=3)
        # 2041
        twentyfive_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2040,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2041 = twentyfive_ABC.y[0:9, 4:5]
        starting_2041 = starting_2041.flatten()
        starting_2041[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2041[4] = np.random.uniform(low=0.18,high=3)
        # 2042
        twentysix_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2041,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        starting_2042 = twentysix_ABC.y[0:9, 4:5]
        starting_2042 = starting_2042.flatten()
        starting_2042[1] = np.random.uniform(low=0.56,high=2.0)
        starting_2042[4] = np.random.uniform(low=0.18,high=3)
        # 2043
        twentyseven_ABC = solve_ivp(ecoNetwork, (0, 1), starting_2042,  t_eval = t, args=(A_3, r_3), method = 'RK23')
        # concatenate & append all the runs
        combined_runs_2 = np.hstack((third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y, twelfth_ABC.y, thirteenth_ABC.y, fourteenth_ABC.y, fifteenth_ABC.y, sixteenth_ABC.y, seventeenth_ABC.y, eighteenth_ABC.y, nineteenth_ABC.y, twentieth_ABC.y, twentyone_ABC.y, twentytwo_ABC.y, twentythree_ABC.y, twentyfour_ABC.y, twentyfive_ABC.y, twentysix_ABC.y, twentyseven_ABC.y))
        # print(combined_runs)
        all_runs_3 = np.append(all_runs_3, combined_runs_2)
        # append all the parameters
        all_parameters_3.append(parameters_used_3)   
    # check the final runs
    final_runs_3 = (np.vstack(np.hsplit(all_runs_3.reshape(len(species)*len(accepted_simulations_2), 125).transpose(),len(accepted_simulations_2))))
    final_runs_3 = pd.DataFrame(data=final_runs_3, columns=species)
    # append all the parameters to a dataframe
    all_parameters_3 = pd.concat(all_parameters_3)
    # add ID to the dataframe & parameters
    all_parameters_3['ID'] = ([(x+1) for x in range(len(accepted_simulations_2)) for _ in range(len(parameters_used_3))])
    IDs = np.arange(1,1 + len(accepted_simulations_2))
    final_runs_3['ID'] = np.repeat(IDs,125)
    final_runs_3['accepted?'] = np.repeat('Accepted', len(final_runs_3))
    return final_runs_3, all_parameters_3, X0_3, accepted_simulations_2, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    final_runs_3, all_parameters_3, X0_3, accepted_simulations_2018, accepted_parameters_2018, final_runs_2,  all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = runODE_3()
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
    species_thirdRun = np.tile(species, 125*len(accepted_simulations_2018))
    species_list = np.concatenate((species_firstRun, species_secondRun, species_thirdRun), axis=None)
    # add a grouping variable to graph each run separately
    grouping1 = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    grouping_variable1 = np.repeat(grouping1,50*len(species))
    grouping2 = np.arange(NUMBER_OF_SIMULATIONS+1, NUMBER_OF_SIMULATIONS + len(accepted_simulations)+1)
    grouping_variable2 = np.repeat(grouping2,50*len(species))
    grouping3 = np.arange(NUMBER_OF_SIMULATIONS + 1 + len(accepted_simulations), NUMBER_OF_SIMULATIONS + len(accepted_simulations) + len(accepted_simulations_2018) + 1)
    grouping_variable3 = np.repeat(grouping3,125*len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping_variable1, grouping_variable2, grouping_variable3), axis=None)
    # years - we've looked at 23 so far (2005-2009; 2009-2018; 2018-2027)
    year = np.arange(1,6) 
    year2 = np.arange(5,15) 
    year3 = np.arange(14,39) 
    indices1 = np.repeat(year,len(species)*10)
    indices1 = np.tile(indices1, NUMBER_OF_SIMULATIONS)
    indices2 = np.repeat(year2,len(species)*5)
    indices2 = np.tile(indices2, len(accepted_simulations))
    indices3 = np.repeat(year3,len(species)*5)
    indices3 = np.tile(indices3, len(accepted_simulations_2018))
    indices = np.concatenate((indices1, indices2, indices3), axis=None)

    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': y_values, 'runNumber': grouping_variable, 'Ecosystem Element': species_list, 'Time': indices, 'runType': accepted_shape})
    # calculate median 
    m = final_df.groupby(['Time', 'runType','Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time', 'runType','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time', 'runType','Ecosystem Element'])['Abundance %'].quantile(.9)
    perc1.name = 'ninetyPerc'
    final_df = final_df.join(perc1, on=['Time', 'runType','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'runType','Ecosystem Element'])['Abundance %'].quantile(.1)
    perc2.name = "tenPerc"
    final_df = final_df.join(perc2, on=['Time','runType', 'Ecosystem Element'])
    # copy to CSV
    # final_df.to_csv('medianQuantiles_final.csv')
    # filter the accepted runs to graph
    filtered_df = final_df.loc[(final_df['runType'] == "Accepted")]

    #### ACCEPTED RUNS ONLY ### 

    # color palette
    palette = dict(zip(filtered_df.runType.unique(),
                    sns.color_palette("mako", 1)))
                    # ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    # plot
    g = sns.relplot(x="Time", y="Median",
                col="Ecosystem Element",
                facet_kws=dict(sharey=False),
                kind="line", palette=palette, col_wrap=4, legend=False, ci = None, data=filtered_df)
    g.set(xticks=[0, 5, 15, 40])
    # add quantile lines
    g.map_dataframe(sns.lineplot, 'Time', 'tenPerc', palette=palette)
    g.map_dataframe(sns.lineplot, 'Time', 'ninetyPerc', palette=palette)
    # fill in the lines
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[0].get_xdata(), ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), alpha=0.1)
    axes = g.axes.flatten()
    axes[0].set_title("Grassland & parkland")
    axes[1].set_title("Large herbivores")
    axes[2].set_title("Organic carbon")
    axes[3].set_title("Roe deer")
    axes[4].set_title("Tamworth pigs")
    axes[5].set_title("Thorny scrubland")
    axes[6].set_title("Woodland")
    # stop the plots from overlapping
    plt.tight_layout()
    plt.show()



    # #### REJECTED AND ACCEPTED RUNS ####

    # # color palette
    # palette = dict(zip(final_df.runType.unique(),
    #                 sns.color_palette("mako", 2)))
    #                 # ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
    # # plot
    # g = sns.relplot(x="Time", y="Median",
    #             hue = "runType",col="Ecosystem Element",
    #             facet_kws=dict(sharey=False),
    #             kind="line", palette=palette, col_wrap=4, legend=False, ci = None, data=final_df)
    # g.set(xticks=[0, 5, 15, 40])
    # # add quantile lines
    # g.map_dataframe(sns.lineplot, 'Time', 'tenPerc', hue = "runType", palette=palette)
    # g.map_dataframe(sns.lineplot, 'Time', 'ninetyPerc', hue = "runType", palette=palette)
    # # fill in the lines
    # for ax in g.axes.flat:
    #     if ax.lines[1]:
    #         ax.fill_between(ax.lines[1].get_xdata(), ax.lines[3].get_ydata(), ax.lines[4].get_ydata(), alpha=0.1)
    #     else:
    #         ax.fill_between(ax.lines[0].get_xdata(), ax.lines[2].get_ydata(), ax.lines[5].get_ydata(), alpha=0.1)
    # # stop the plots from overlapping
    # plt.tight_layout()
    # # change subplot titles
    # axes = g.axes.flatten()
    # axes[0].set_title("Grassland & parkland")
    # axes[1].set_title("Large herbivores")
    # axes[2].set_title("Organic carbon")
    # axes[3].set_title("Roe deer")
    # axes[4].set_title("Tamworth pigs")
    # axes[5].set_title("Thorny scrubland")
    # axes[6].set_title("Woodland")
    # plt.legend(labels=['Rejected', 'Accepted'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    # plt.show()

plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
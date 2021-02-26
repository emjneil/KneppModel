# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# download packages
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
# from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
import seaborn as sns
import numpy.matlib
from scipy import stats
import csv

# time the program
start = timeit.default_timer()

# define the number of simulations to try
totalSimulations = 1000

# store species in a list
species = ['grasslandParkland','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']
# define the Lotka-Volterra equation
def ecoNetwork(t, X, A, r):
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))


# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():
    # define the array
    interaction_matrix = [
                [-0.77,-0.005,0,-0.0031,-0.0073,-0.012,-0.013],
                [0.7,-0.36,0,0,0,0.042,0.2],
                [0.095,0.0052,-0.098,0.0026,0.0053,0.0002,0.083],
                [0.58,0,0,-0.69,0,0.079,0.31],
                [0.66,0,0,0,-0.38,0.01,0.31],
                [0,-0.08,0,-0.044,-0.054,-0.0018,-0.016], 
                [0,-0.00032,0,-0.00016,-0.0004,0.00028,-0.0087]
                ]
    # generate random uniform numbers
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix


def generateGrowth():
    growthRates = [0.93, 0, 0, 0, 0, 0.67, 0.027] 
    # multiply by a range
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),))
    growth = growthRates * variation
    # give wider bounds for woodland
    # growth[6] = np.random.uniform(low = 0, high=0.1, size = (1,))
    return growth
    

def generateX0():
    # scale everything to abundance of zero (except species to be reintroduced)
    X0 = [1, 0, 1, 1, 0, 1, 1]
    return X0


# # # # --- MAKE NETWORK X VISUAL OF INTERACTION MATRIX --

# def networkVisual():
#     import networkx as nx
#     interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
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


# check for stability
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


def runODE_1():
    # Define time points: first 5 years (2005-2009)
    t = np.linspace(0, 4, 12)
    t_eco = np.linspace(0, 1, 2)
    all_runs = []
    all_times = []
    all_parameters = []
    NUMBER_OF_SIMULATIONS = 0
    NUMBER_STABLE = 0
    # number of simulations starts at 0 but will be added to (depending on how many parameters pass stability constraint)
    for _ in range(totalSimulations):
        A = generateInteractionMatrix()
        r = generateGrowth()
        X0 = generateX0()
        # check viability of the parameter set (is it stable?)
        ia = np.linalg.inv(A)
        # n is the equilibrium state; calc as inverse of -A*r
        n = -np.matmul(ia, r)
        isStable = calcStability(A, r, n)
        # if all the values of n are above zero at equilibrium, & if the parameter set is viable (stable & all n > 0 at equilibrium); do the calculation
        if np.all(n > 0) &  isStable == True:
            NUMBER_STABLE += 1
            # check ecological parameters (primary producers shouldn't go neg on their own)
            all_ecoCheck = []
            for i in range(len(species)):
                X0_ecoCheck = [0, 0, 0, 0, 0, 0, 0]
                X0_ecoCheck[i] = 1
                ecoCheck_ABC = solve_ivp(ecoNetwork, (0, 1), X0_ecoCheck,  t_eval = t_eco, args=(A, r), method = 'RK23') 
                all_ecoCheck = np.append(all_ecoCheck, ecoCheck_ABC.y)
            all_ecoCheck_results = (np.vstack(np.hsplit(all_ecoCheck.reshape(len(species), 14).transpose(),1)))
            all_ecoCheck_results = pd.DataFrame(data=all_ecoCheck_results, columns=species)  

            # primary producers should be >= 0
            if (all_ecoCheck_results.loc[1,'grasslandParkland'] >= 1) & (all_ecoCheck_results.loc[11,'thornyScrub'] >= 1) & (all_ecoCheck_results.loc[13,'woodland'] >= 1):
                # remember the parameters used
                X0_growth = pd.concat([pd.DataFrame(X0), pd.DataFrame(r)], axis = 1)
                X0_growth.columns = ['X0','growth']
                parameters_used = pd.concat([X0_growth, pd.DataFrame(A, index = species, columns = species)])
                all_parameters.append(parameters_used)
                # run the ODE
                first_ABC = solve_ivp(ecoNetwork, (0, 4), X0,  t_eval = t, args=(A, r), method = 'RK23')
                # append all the runs
                all_runs = np.append(all_runs, first_ABC.y)
                all_times = np.append(all_times, first_ABC.t)
                # add one to the counter (so we know how many simulations were run in the end)
                NUMBER_OF_SIMULATIONS += 1
                # print(NUMBER_OF_SIMULATIONS)

    # check the final runs
    print("number of stable simulations", NUMBER_STABLE)
    print("number of stable & ecologically sound simulations", NUMBER_OF_SIMULATIONS)
    
    # put together the final runs
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 12).transpose(),NUMBER_OF_SIMULATIONS)))
    final_runs = pd.DataFrame(data=final_runs, columns=species)
    final_runs['time'] = all_times
    with pd.option_context('display.max_columns',None):
        print(final_runs)

    
    # append all the parameters to a dataframe
    all_parameters = pd.concat(all_parameters)
    # add ID to all_parameters
    all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])
    return final_runs, all_parameters, NUMBER_OF_SIMULATIONS


# --------- FILTER OUT UNREALISTIC RUNS -----------

def filterRuns_1():
    final_runs, all_parameters, NUMBER_OF_SIMULATIONS = runODE_1()
    # add ID to dataframe
    IDs = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    final_runs['ID'] = np.repeat(IDs,12)
    # select only the last year
    accepted_year = final_runs.loc[final_runs['time'] == 4]
    with pd.option_context('display.max_columns',None):
        print(accepted_year)
    # add filtering criteria 
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= 3.3) & (accepted_year['roeDeer'] >= 1) &
    (accepted_year['grasslandParkland'] <= 1) & (accepted_year['grasslandParkland'] >= 0.9) &
    (accepted_year['woodland'] <=1.56) & (accepted_year['woodland'] >= 0.85) &
    (accepted_year['thornyScrub'] <= 19) & (accepted_year['thornyScrub'] >= 1) &
    (accepted_year['organicCarbon'] <= 1.9) & (accepted_year['organicCarbon'] >= 0.95) 
    ]
    print(accepted_simulations.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # add accepted ID to original dataframe
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Accepted', 'Rejected')
    return accepted_parameters, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS




# # # # ---------------------- ODE #2: Years 2009-2018 -------------------------

def generateParameters2():
    accepted_parameters, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS = filterRuns_1()
    # # select growth rates 
    growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    r_secondRun = growthRates_2.to_numpy()
    # make the final runs of ODE #1 the initial conditions
    accepted_simulations = accepted_simulations.drop(['ID', 'time'], axis=1)
    accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']] = accepted_simulations.values.flatten()
    # select X0 
    X0_2 = accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']]
    X0_2 = pd.DataFrame(X0_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    # # add reintroduced species
    X0_2.loc[:, 'largeHerb'] = [1 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [1 for i in X0_2.index]
    X0_secondRun = X0_2.to_numpy()
    # # select interaction matrices part of the dataframes 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    # turn to array
    A_secondRun = interaction_strength_2.to_numpy()
    return r_secondRun, X0_secondRun, A_secondRun, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS



# # # # # ------ SOLVE ODE #2: Pre-reintroductions (2009-2018) -------

def runODE_2():
    r_secondRun, X0_secondRun, A_secondRun, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS  = generateParameters2()
    all_runs_2 = []
    all_times_2 = []
    all_parameters_2 = []
    t_2 = np.linspace(4, 5, 3)
    # loop through each row of accepted parameters
    for X0_3, r_2, A_2 in zip(X0_secondRun,r_secondRun, np.array_split(A_secondRun,len(accepted_simulations))):
        # concantenate the parameters
        X0_growth_2 = pd.concat([pd.DataFrame(X0_3), pd.DataFrame(r_2)], axis = 1)
        X0_growth_2.columns = ['X0','growth']
        parameters_used_2 = pd.concat([X0_growth_2, pd.DataFrame(A_2, index = species, columns = species)])
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        second_ABC = solve_ivp(ecoNetwork, (4, 5), X0_3,  t_eval = t_2, args=(A_2, r_2), method = 'RK23')        
        # take those values and re-run for another year, adding forcings
        starting_values_2010 = second_ABC.y[0:7, 2:3].flatten()
        starting_values_2010[1] = 1.9
        starting_values_2010[4] = 0.9
        t_3 = np.linspace(5, 6, 3)
        # run the model for another year 2010-2011
        third_ABC = solve_ivp(ecoNetwork, (5, 6), starting_values_2010,  t_eval = t_3, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2011 = third_ABC.y[0:7, 2:3].flatten()
        starting_values_2011[1] = 2.8
        starting_values_2011[4] = 1.1
        t_4 = np.linspace(6, 7, 3)
        # run the model for 2011-2012
        fourth_ABC = solve_ivp(ecoNetwork, (6, 7), starting_values_2011,  t_eval = t_4, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2012 = fourth_ABC.y[0:7, 2:3].flatten()
        starting_values_2012[1] = 3.2
        starting_values_2012[4] = 1.7
        t_5 = np.linspace(7, 8, 3)
        # run the model for 2012-2013
        fifth_ABC = solve_ivp(ecoNetwork, (7, 8), starting_values_2012,  t_eval = t_5, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2013 = fifth_ABC.y[0:7, 2:3].flatten()
        starting_values_2013[1] = 5.1
        starting_values_2013[4] = 0.3
        t_6 = np.linspace(8, 9, 3)
        # run the model for 2013-2014
        sixth_ABC = solve_ivp(ecoNetwork, (8, 9), starting_values_2013,  t_eval = t_6, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2014 = sixth_ABC.y[0:7, 2:3].flatten()
        starting_values_2014[1] = 3.1
        starting_values_2014[4] = 0.9
        t_7 = np.linspace(9, 10, 3)
        # run the model for 2014-2015
        seventh_ABC = solve_ivp(ecoNetwork, (9, 10), starting_values_2014,  t_eval = t_7, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2015 = seventh_ABC.y[0:7, 2:3].flatten()
        starting_values_2015[1] = 2.4
        starting_values_2015[4] = 0.4
        t_8 = np.linspace(10, 11, 3)
        # run the model for 2015-2016
        eighth_ABC = solve_ivp(ecoNetwork, (10, 11), starting_values_2015,  t_eval = t_8, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2016 = eighth_ABC.y[0:7, 2:3].flatten()
        starting_values_2016[1] = 3.1
        starting_values_2016[4] = 0.4
        t_9 = np.linspace(11, 12, 3)
        # run the model for 2016-2017
        ninth_ABC = solve_ivp(ecoNetwork, (11, 12), starting_values_2016,  t_eval = t_9, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2017 = ninth_ABC.y[0:7, 2:3].flatten()
        starting_values_2017[1] = 2.6
        starting_values_2017[4] = 0.6
        t_10 = np.linspace(12, 13, 3)
        # run the model for 2017-2018
        tenth_ABC = solve_ivp(ecoNetwork, (12, 13), starting_values_2017,  t_eval = t_10, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_values_2018 = tenth_ABC.y[0:7, 2:3].flatten()
        starting_values_2018[1] = 4.2
        starting_values_2018[4] = 0.5
        t_11 = np.linspace(13, 14, 3)
        # run the model for 2018-2019
        eleventh_ABC = solve_ivp(ecoNetwork, (13, 14), starting_values_2018,  t_eval = t_11, args=(A_2, r_2), method = 'RK23')
        starting_values_2019 = eleventh_ABC.y[0:7, 2:3].flatten()
        starting_values_2019[1] = 4.1
        starting_values_2019[4] = 0.4
        t_12 = np.linspace(14, 15, 3)
        # run the model for 2019-2020
        twelfth_ABC = solve_ivp(ecoNetwork, (14,15), starting_values_2019,  t_eval = t_12, args=(A_2, r_2), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y, twelfth_ABC.y))
        combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, eighth_ABC.t, ninth_ABC.t, tenth_ABC.t, eleventh_ABC.t, twelfth_ABC.t))
        # print(combined_runs)
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)   
        all_times_2 = np.append(all_times_2, combined_times)
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 33).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    final_runs_2['time'] = all_times_2
    with pd.option_context('display.max_columns',None):
        print(final_runs_2)
    # append all the parameters to a dataframe
    all_parameters_2 = pd.concat(all_parameters_2)
    # add ID to the dataframe & parameters
    all_parameters_2['ID'] = ([(x+1) for x in range(len(accepted_simulations)) for _ in range(len(parameters_used_2))])
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,33)
    return final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = runODE_2()
    # select only the last run (filter to the year 2010); make sure these are in line with the new min/max X0 values (from the minimization), not the original X0 bounds
    accepted_year_2018 = final_runs_2.loc[final_runs_2['time'] == 15]
    with pd.option_context('display.max_columns',None):
        print(accepted_year_2018)
    accepted_simulations_2018 = accepted_year_2018[
    (accepted_year_2018['roeDeer'] <= 6.7) & (accepted_year_2018['roeDeer'] >= 1.7) &
    (accepted_year_2018['grasslandParkland'] <= 0.79) & (accepted_year_2018['grasslandParkland'] >= 0.67) &
    (accepted_year_2018['woodland'] <= 1.73) & (accepted_year_2018['woodland'] >= 0.98) &
    (accepted_year_2018['thornyScrub'] <= 35.1) & (accepted_year_2018['thornyScrub'] >= 22.5) & 
    (accepted_year_2018['organicCarbon'] <= 2.2) & (accepted_year_2018['organicCarbon'] >= 1.7)
    ]

    print(accepted_simulations_2018.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2018 = all_parameters_2[all_parameters_2['ID'].isin(accepted_simulations_2018['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2018['ID']), 'Accepted', 'Rejected')
    return accepted_simulations_2018, accepted_parameters_2018, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# # # # ---------------------- ODE #3: projecting 10 years (2018-2028) -------------------------

def generateParameters3():
    accepted_simulations_2018, accepted_parameters_2018, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun = filterRuns_2()
    # # select growth rates 
    growthRates_3 = accepted_parameters_2018.loc[accepted_parameters_2018['growth'].notnull(), ['growth']]
    growthRates_3 = pd.DataFrame(growthRates_3.values.reshape(len(accepted_simulations_2018), len(species)), columns = species)
    r_thirdRun = growthRates_3.to_numpy()
    # make the final runs of ODE #2 the initial conditions
    accepted_simulations_2 = accepted_simulations_2018.drop(['ID', 'time'], axis=1)
    accepted_parameters_2018.loc[accepted_parameters_2018['X0'].notnull(), ['X0']] = accepted_simulations_2.values.flatten()
    # select X0 
    X0_3 = accepted_parameters_2018.loc[accepted_parameters_2018['X0'].notnull(), ['X0']]
    X0_3 = pd.DataFrame(X0_3.values.reshape(len(accepted_simulations_2), len(species)), columns = species)
    # # select interaction matrices part of the dataframes 
    interaction_strength_3 = accepted_parameters_2018.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_3 = interaction_strength_3.dropna()
    A_thirdRun = interaction_strength_3.to_numpy()
    # check histograms for growth
    growth_filtered = growthRates_3[["grasslandParkland","thornyScrub","woodland"]]
    fig, axes = plt.subplots(len(growth_filtered.columns)//3,3, figsize=(25, 10))
    for col, axis in zip(growth_filtered.columns, axes):
        growth_filtered.hist(column = col, ax = axis, bins = 25)
    # check correlation matrix
    growthRates_3.columns = ['grasslandParkland_growth', 'largeHerb_growth', 'organicCarbon_growth', 'roeDeer_growth', 'tamworthPig_growth', 'thornyScrub_growth', 'woodland_growth']
    # reshape int matrix
    arableInts = interaction_strength_3[interaction_strength_3.index=='grasslandParkland']
    arableInts.columns = ['grass_grass', 'grass_largeHerb', 'grass_carb','grass_roe', 'grass_tam', 'grass_thorn', 'grass_wood']
    arableInts = arableInts.reset_index(drop=True)
    largeHerbInts = interaction_strength_3[interaction_strength_3.index=='largeHerb']
    largeHerbInts.columns = ['largeHerb_grass', 'largeHerb_largeHerb', 'largeHerb_carb','largeHerb_roe', 'largeHerb_tam', 'largeHerb_thorn', 'largeHerb_wood']
    largeHerbInts = largeHerbInts.reset_index(drop=True)
    orgCarbInts = interaction_strength_3[interaction_strength_3.index=='organicCarbon']
    orgCarbInts.columns = ['organicCarbon_grass', 'organicCarbon_largeHerb', 'organicCarboncarb','organicCarbon_roe', 'organicCarbon_tam', 'organicCarbon_thorn', 'organicCarbon_wood']
    orgCarbInts = orgCarbInts.reset_index(drop=True)
    roeDeerInts = interaction_strength_3[interaction_strength_3.index=='roeDeer']
    roeDeerInts.columns = ['roeDeer_grass', 'roeDeer_largeHerb', 'roeDeer_carb','roeDeer_roe', 'roeDeer_tam', 'roeDeer_thorn', 'roeDeer_wood']
    roeDeerInts = roeDeerInts.reset_index(drop=True)
    tamworthPigInts = interaction_strength_3[interaction_strength_3.index=='tamworthPig']
    tamworthPigInts.columns = ['tamworthPig_grass', 'tamworthPig_largeHerb', 'tamworthPigcarb','tamworthPig_roe', 'tamworthPig_tam', 'tamworthPig_thorn', 'tamworthPig_wood']
    tamworthPigInts = tamworthPigInts.reset_index(drop=True)
    thornyScrubInts = interaction_strength_3[interaction_strength_3.index=='thornyScrub']
    thornyScrubInts.columns = ['thornyScrub_grass', 'thornyScrub_largeHerb', 'thornyScrub_carb','thornyScrub_roe', 'thornyScrub_tam', 'thornyScrub_thorn', 'thornyScrub_wood']
    thornyScrubInts = thornyScrubInts.reset_index(drop=True)
    woodlandInts = interaction_strength_3[interaction_strength_3.index=='woodland']
    woodlandInts.columns = ['woodland_grass', 'woodland_largeHerb', 'woodland_carb','woodland_roe', 'woodland_tam', 'woodland_thorn', 'woodland_wood']
    woodlandInts = woodlandInts.reset_index(drop=True)
    # combine dataframes
    combined = pd.concat([growthRates_3, arableInts, largeHerbInts, orgCarbInts, roeDeerInts, tamworthPigInts, thornyScrubInts, woodlandInts], axis=1)
    combined = combined.loc[:, (combined != 0).any(axis=0)]
    correlationMatrix = combined.corr()

    # check histograms for interaction matrix (split into 7 for viewing)
    # fig_2, axes_2 = plt.subplots(len(arableInts.columns)//7,7)
    # for col_2, axis_2 in zip(arableInts.columns, axes_2):
    #     arableInts.hist(column = col_2, ax = axis_2, bins = 15)
    # plt.show()

    # fig_3, axes_3 = plt.subplots(len(largeHerbInts.columns)//7,7)
    # for col_3, axis_3 in zip(largeHerbInts.columns, axes_3):
    #     largeHerbInts.hist(column = col_3, ax = axis_3, bins = 15)
    # plt.show()

    # fig_4, axes_4 = plt.subplots(len(orgCarbInts.columns)//7,7)
    # for col_4, axis_4 in zip(orgCarbInts.columns, axes_4):
    #     orgCarbInts.hist(column = col_4, ax = axis_4, bins = 15)
    # plt.show()

    # fig_5, axes_5 = plt.subplots(len(roeDeerInts.columns)//7,7)
    # for col_5, axis_5 in zip(roeDeerInts.columns, axes_5):
    #     roeDeerInts.hist(column = col_5, ax = axis_5, bins = 15)
    # plt.show()

    # fig_6, axes_6 = plt.subplots(len(tamworthPigInts.columns)//7,7)
    # for col_6, axis_6 in zip(tamworthPigInts.columns, axes_6):
    #     tamworthPigInts.hist(column = col_6, ax = axis_6, bins = 15)
    # plt.show()

    # fig_7, axes_7 = plt.subplots(len(thornyScrubInts.columns)//7,7)
    # for col_7, axis_7 in zip(thornyScrubInts.columns, axes_7):
    #     thornyScrubInts.hist(column = col_7, ax = axis_7, bins = 15)
    # plt.show()

    # fig_8, axes_8 = plt.subplots(len(woodlandInts.columns)//7,7)
    # for col_8, axis_8 in zip(woodlandInts.columns, axes_8):
    #     woodlandInts.hist(column = col_8, ax = axis_8, bins = 15)
    # plt.show()

    # calculate p values and remove non-significant ones
    p_matrix = np.zeros(shape=(correlationMatrix.shape[1],correlationMatrix.shape[1]))
    for col in correlationMatrix.columns:
            for col2 in correlationMatrix.drop(col,axis=1).columns:
                _ , p = stats.pearsonr(correlationMatrix[col],correlationMatrix[col2])
                p_matrix[correlationMatrix.columns.to_list().index(col),correlationMatrix.columns.to_list().index(col2)] = p
    p_matrix = pd.DataFrame(data=p_matrix, index=correlationMatrix.index, columns=correlationMatrix.index)
    # select only the significant ones, show their corr
    signif_Matrix = correlationMatrix.where(p_matrix.values < 0.05)
    # plot it
    plt.subplots(figsize=(10,10))
    ax = sns.heatmap(
    signif_Matrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    # annot = True,
    linewidths=.5
            )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right',
        fontsize = 5)
    ax.set_yticklabels(
        ax.get_yticklabels(), 
        fontsize = 5)
    plt.savefig('corrMatrix_5mil_practice.png')
    plt.show()
    return r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2018, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun



# # # # # ------ SOLVE ODE #3: Projecting forwards 10 years (2018-2028) -------

def runODE_3():
    r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2018, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = generateParameters3()
    # project into the future
    X0_thirdRun = X0_3.to_numpy()
    all_runs_3 = []
    all_parameters_3 = []
    all_times_3 = []
    t = np.linspace(15, 16, 3)
    # loop through each row of accepted parameters
    for X0_4, r_4, A_4 in zip(X0_thirdRun,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2018))):
        # start reintroduced species at culled value (otherwise it starts high, at the value to which it grew over the year)
        X0_4[1] = np.random.uniform(low=3.9,high=4.7)
        X0_4[4] =  np.random.uniform(low=0.49,high=0.6)
        # concantenate the parameters
        X0_growth_3 = pd.concat([pd.DataFrame(X0_4), pd.DataFrame(r_4)], axis = 1)
        X0_growth_3.columns = ['X0','growth']
        parameters_used_3 = pd.concat([X0_growth_3, pd.DataFrame(A_4, index = species, columns = species)])
        # run the model for one year 2018-2019 (to account for herbivore numbers being manually controlled every year)
        third_ABC = solve_ivp(ecoNetwork, (15, 16), X0_4,  t_eval = t, args=(A_4, r_4), method = 'RK23')        
        # take those values and re-run for another year, adding forcings
        starting_2019 = third_ABC.y[0:7, 2:3].flatten()
        starting_2019[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2019[4] =  np.random.uniform(low=0.3,high=0.4)
        t_1 = np.linspace(16, 17, 3)
        # run the model for another year 2020
        fourth_ABC = solve_ivp(ecoNetwork, (16, 17), starting_2019,  t_eval = t_1, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2020 = fourth_ABC.y[0:9, 2:3].flatten()
        starting_2020[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2020[4] = np.random.uniform(low=0.3,high=0.4)
        t_2 = np.linspace(17, 18, 3)
        # run the model for 2021
        fifth_ABC = solve_ivp(ecoNetwork, (17, 18), starting_2020,  t_eval = t_2, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2021 = fifth_ABC.y[0:9, 2:3].flatten()
        starting_2021[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2021[4] = np.random.uniform(low=0.3,high=0.4)
        t_3 = np.linspace(18, 19, 3)
        # run the model for 2022
        sixth_ABC = solve_ivp(ecoNetwork, (18, 19), starting_2021,  t_eval = t_3, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2022 = sixth_ABC.y[0:9, 2:3].flatten()
        starting_2022[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2022[4] = np.random.uniform(low=0.3,high=0.4)
        t_4 = np.linspace(19, 20, 3)
        # run the model for 2023
        seventh_ABC = solve_ivp(ecoNetwork, (19, 20), starting_2022,  t_eval = t_4, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2023 = seventh_ABC.y[0:9, 2:3].flatten()
        starting_2023[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2023[4] = np.random.uniform(low=0.3,high=0.4)
        t_5 = np.linspace(20, 21, 3)
        # run the model for 2024
        eighth_ABC = solve_ivp(ecoNetwork, (20, 21), starting_2023,  t_eval = t_5, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2024 = eighth_ABC.y[0:9, 2:3].flatten()
        starting_2024[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2024[4] = np.random.uniform(low=0.3,high=0.4)
        t_6 = np.linspace(21, 22, 3)
        # run the model for 2011-2012
        ninth_ABC = solve_ivp(ecoNetwork, (21, 22), starting_2024,  t_eval = t_6, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2025 = ninth_ABC.y[0:9, 2:3].flatten()
        starting_2025[1] =np.random.uniform(low=4.0,high=4.9)
        starting_2025[4] = np.random.uniform(low=0.3,high=0.4)
        t_7 = np.linspace(22, 23, 3)
        # run the model for 2011-2012
        tenth_ABC = solve_ivp(ecoNetwork, (22,23), starting_2025,  t_eval = t_7, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2026 = tenth_ABC.y[0:9, 2:3].flatten()
        starting_2026[1] =np.random.uniform(low=4.0,high=4.9)
        starting_2026[4] = np.random.uniform(low=0.3,high=0.4)
        t_8 = np.linspace(23, 24, 3)
        # run the model for 2011-2012
        eleventh_ABC = solve_ivp(ecoNetwork, (23, 24), starting_2026,  t_eval = t_8, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2027 = eleventh_ABC.y[0:9, 2:3].flatten()
        starting_2027[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2027[4] = np.random.uniform(low=0.3,high=0.4)
        t_9 = np.linspace(24, 25, 3)
        # run the model to 2028
        twelfth_ABC = solve_ivp(ecoNetwork, (24, 25), starting_2027,  t_eval = t_9, args=(A_4, r_4), method = 'RK23')
        # just to check longer (up to 25 yrs)
        starting_2028 = twelfth_ABC.y[0:9,2:3].flatten()
        starting_2028[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2028[4] = np.random.uniform(low=0.3,high=0.4)
        t_10 = np.linspace(25, 26, 3)
        # 2029
        thirteenth_ABC = solve_ivp(ecoNetwork, (25, 26), starting_2028,  t_eval = t_10, args=(A_4, r_4), method = 'RK23')
        starting_2029 = thirteenth_ABC.y[0:9, 2:3].flatten()
        starting_2029[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2029[4] = np.random.uniform(low=0.3,high=0.4)
        t_11 = np.linspace(26, 27, 3)
        # 2030
        fourteenth_ABC = solve_ivp(ecoNetwork, (26, 27), starting_2029,  t_eval = t_11, args=(A_4, r_4), method = 'RK23')
        starting_2030 = fourteenth_ABC.y[0:9, 2:3].flatten()
        starting_2030[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2030[4] = np.random.uniform(low=0.3,high=0.4)
        t_12 = np.linspace(27, 28, 3)
        # 2031
        fifteenth_ABC = solve_ivp(ecoNetwork, (27, 28), starting_2030,  t_eval = t_12, args=(A_4, r_4), method = 'RK23')
        starting_2031 = fifteenth_ABC.y[0:9, 2:3].flatten()
        starting_2031[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2031[4] = np.random.uniform(low=0.3,high=0.4)
        t_13 = np.linspace(28, 29, 3)
        # 2032
        sixteenth_ABC = solve_ivp(ecoNetwork, (28, 29), starting_2031,  t_eval = t_13, args=(A_4, r_4), method = 'RK23')
        starting_2032 = sixteenth_ABC.y[0:9, 2:3].flatten()
        starting_2032[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2032[4] = np.random.uniform(low=0.3,high=0.4)
        t_14 = np.linspace(29, 30, 3)
        # 2033
        seventeenth_ABC = solve_ivp(ecoNetwork, (29, 30), starting_2032,  t_eval = t_14, args=(A_4, r_4), method = 'RK23')
        starting_2033 = seventeenth_ABC.y[0:9, 2:3].flatten()
        starting_2033[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2033[4] = np.random.uniform(low=0.3,high=0.4)
        t_15 = np.linspace(30, 31, 3)
        # 2034
        eighteenth_ABC = solve_ivp(ecoNetwork, (30, 31), starting_2033,  t_eval = t_15, args=(A_4, r_4), method = 'RK23')
        starting_2034 = eighteenth_ABC.y[0:9, 2:3].flatten()
        starting_2034[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2034[4] = np.random.uniform(low=0.3,high=0.4)
        t_16 = np.linspace(31, 32, 3)
        # 2035
        nineteenth_ABC = solve_ivp(ecoNetwork, (31, 32), starting_2034,  t_eval = t_16, args=(A_4, r_4), method = 'RK23')
        starting_2035 = nineteenth_ABC.y[0:9, 2:3].flatten()
        starting_2035[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2035[4] = np.random.uniform(low=0.3,high=0.4)
        t_17 = np.linspace(32, 33, 3)
        # 2036
        twentieth_ABC = solve_ivp(ecoNetwork, (32, 33), starting_2035,  t_eval = t_17, args=(A_4, r_4), method = 'RK23')
        starting_2036 = twentieth_ABC.y[0:9, 2:3].flatten()
        starting_2036[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2036[4] = np.random.uniform(low=0.3,high=0.4)
        t_18 = np.linspace(33, 34, 3)
        # 2037
        twentyone_ABC = solve_ivp(ecoNetwork, (33, 34), starting_2036,  t_eval = t_18, args=(A_4, r_4), method = 'RK23')
        starting_2037 = twentyone_ABC.y[0:9, 2:3].flatten()
        starting_2037[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2037[4] = np.random.uniform(low=0.3,high=0.4)
        t_19 = np.linspace(34, 35, 3)
        # 2038
        twentytwo_ABC = solve_ivp(ecoNetwork, (34, 35), starting_2037,  t_eval = t_19, args=(A_4, r_4), method = 'RK23')
        starting_2038 = twentytwo_ABC.y[0:9, 2:3].flatten()
        starting_2038[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2038[4] = np.random.uniform(low=0.3,high=0.4)
        t_20 = np.linspace(35, 36, 3)
        # 2039
        twentythree_ABC = solve_ivp(ecoNetwork, (35, 36), starting_2038,  t_eval = t_20, args=(A_4, r_4), method = 'RK23')
        starting_2039 = twentythree_ABC.y[0:9,2:3].flatten()
        starting_2039[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2039[4] = np.random.uniform(low=0.3,high=0.4)
        t_21 = np.linspace(36, 37, 3)
        # 2040
        twentyfour_ABC = solve_ivp(ecoNetwork, (36, 37), starting_2039,  t_eval = t_21, args=(A_4, r_4), method = 'RK23')
        starting_2040 = twentyfour_ABC.y[0:9, 2:3].flatten()
        starting_2040[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2040[4] = np.random.uniform(low=0.3,high=0.4)
        t_22 = np.linspace(37, 38, 3)
        # 2041
        twentyfive_ABC = solve_ivp(ecoNetwork, (37, 38), starting_2040,  t_eval = t_22, args=(A_4, r_4), method = 'RK23')
        starting_2041 = twentyfive_ABC.y[0:9, 2:3].flatten()
        starting_2041[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2041[4] = np.random.uniform(low=0.3,high=0.4)
        t_23 = np.linspace(38, 39, 3)
        # 2042
        twentysix_ABC = solve_ivp(ecoNetwork, (38, 39), starting_2041,  t_eval = t_23, args=(A_4, r_4), method = 'RK23')
        starting_2042 = twentysix_ABC.y[0:9, 2:3].flatten()
        starting_2042[1] = np.random.uniform(low=4.0,high=4.9)
        starting_2042[4] = np.random.uniform(low=0.3,high=0.4)
        t_24 = np.linspace(39, 40, 3)
        # 2043
        twentyseven_ABC = solve_ivp(ecoNetwork, (39, 40), starting_2042,  t_eval = t_24, args=(A_4, r_4), method = 'RK23')
        # concatenate & append all the runs
        combined_runs_2 = np.hstack((third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y, twelfth_ABC.y, thirteenth_ABC.y, fourteenth_ABC.y, fifteenth_ABC.y, sixteenth_ABC.y, seventeenth_ABC.y, eighteenth_ABC.y, nineteenth_ABC.y, twentieth_ABC.y, twentyone_ABC.y, twentytwo_ABC.y, twentythree_ABC.y, twentyfour_ABC.y, twentyfive_ABC.y, twentysix_ABC.y, twentyseven_ABC.y))
        combined_times_2 = np.hstack((third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, eighth_ABC.t, ninth_ABC.t, tenth_ABC.t, eleventh_ABC.t, twelfth_ABC.t, thirteenth_ABC.t, fourteenth_ABC.t, fifteenth_ABC.t, sixteenth_ABC.t, seventeenth_ABC.t, eighteenth_ABC.t, nineteenth_ABC.t, twentieth_ABC.t, twentyone_ABC.t, twentytwo_ABC.t, twentythree_ABC.t, twentyfour_ABC.t, twentyfive_ABC.t, twentysix_ABC.t, twentyseven_ABC.t))
        # print(combined_runs)
        all_runs_3 = np.append(all_runs_3, combined_runs_2)
        # append all the parameters
        all_parameters_3.append(parameters_used_3)   
        # append the times
        all_times_3 = np.append(all_times_3, combined_times_2)
    # check the final runs
    final_runs_3 = (np.vstack(np.hsplit(all_runs_3.reshape(len(species)*len(accepted_simulations_2018), 75).transpose(),len(accepted_simulations_2018))))
    final_runs_3 = pd.DataFrame(data=final_runs_3, columns=species)
    # append all the parameters to a dataframe
    all_parameters_3 = pd.concat(all_parameters_3)
    # add ID to the dataframe & parameters
    all_parameters_3['ID'] = ([(x+1) for x in range(len(accepted_simulations_2018)) for _ in range(len(parameters_used_3))])
    IDs = np.arange(1,1 + len(accepted_simulations_2018))
    final_runs_3['ID'] = np.repeat(IDs,75)
    final_runs_3['time'] = all_times_3
    final_runs_3['accepted?'] = np.repeat('Accepted', len(final_runs_3))


    # set reintroduced species to zero to see what would've happened without reintroductions
    X0_3_noReintro = pd.DataFrame(data=X0_secondRun, columns=species)
    # put herbivores to zero
    X0_3_noReintro.loc[:, 'largeHerb'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro.loc[:,'tamworthPig'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro = X0_3_noReintro.to_numpy()
    # loop through each row of accepted parameters
    all_runs_2 = []
    all_times_2 = []
    t_noReintro = np.linspace(4, 40, 105)
    for X0_noReintro, r_4, A_4 in zip(X0_3_noReintro,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2018))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        noReintro_ABC = solve_ivp(ecoNetwork, (4, 40), X0_noReintro,  t_eval = t_noReintro, args=(A_4, r_4), method = 'RK23') 
        all_runs_2 = np.append(all_runs_2, noReintro_ABC.y)
        all_times_2 = np.append(all_times_2, noReintro_ABC.t)
    no_reintro = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations_2018), 105).transpose(),len(accepted_simulations_2018))))
    no_reintro = pd.DataFrame(data=no_reintro, columns=species)
    IDs_2 = np.arange(1,1 + len(accepted_simulations_2018))
    no_reintro['ID'] = np.repeat(IDs_2,105)
    no_reintro['time'] = all_times_2
    # concantenate this will the accepted runs from years 1-5
    filtered_FinalRuns = final_runs.loc[(final_runs['accepted?'] == "Accepted") ]
    no_reintro = pd.concat([filtered_FinalRuns, no_reintro])
    no_reintro['accepted?'] = "noReintro"


    # what if there had been no culling?
    all_runs_noCulls = []
    all_times_noCulls = []
    t_noCulls = np.linspace(4, 40, 105)
    X0_noCull = X0_secondRun
    # loop through each row of accepted parameters
    for X0_noCulling, r_5, A_5 in zip(X0_noCull,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2018))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        noCull_ABC = solve_ivp(ecoNetwork, (4, 40), X0_noCulling,  t_eval = t_noCulls, args=(A_5, r_5), method = 'RK23') 
        all_runs_noCulls = np.append(all_runs_noCulls, noCull_ABC.y)
        all_times_noCulls = np.append(all_times_noCulls, noCull_ABC.t)
    no_Cull = (np.vstack(np.hsplit(all_runs_noCulls.reshape(len(species)*len(accepted_simulations_2018), 105).transpose(),len(accepted_simulations_2018))))
    no_Cull = pd.DataFrame(data=no_Cull, columns=species)
    IDs_3 = np.arange(1,1 + len(accepted_simulations_2018))
    no_Cull['ID'] = np.repeat(IDs_3,105)
    no_Cull['time'] = all_times_noCulls
    no_Cull = pd.concat([filtered_FinalRuns, no_Cull])
    no_Cull['accepted?'] = "noCulls"



    # # how many herbivores are needed to change scrubland to grassland? 
    # all_runs_howManyHerbs = []
    # all_times_howManyHerbs = []
    # t_howManyHerbs = np.linspace(4, 39, 105)
    # X0_howManyHerbs = [0.1, 1, 1, 1, 1, 1, 0.1]
    # # loop through each row of accepted parameters
    # for X0_6, r_6, A_6 in zip(X0_howManyHerbs,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2018))):
    #     # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
    #     howManyHerbs_ABC = solve_ivp(ecoNetwork, (4, 39), X0_6,  t_eval = t_howManyHerbs, args=(A_6, r_6), method = 'RK23') 
    #     all_runs_howManyHerbs = np.append(all_runs_howManyHerbs, howManyHerbs_ABC.y)
    #     all_times_howManyHerbs = np.append(all_times_howManyHerbs, howManyHerbs_ABC.t)
    # howManyHerbs = (np.vstack(np.hsplit(all_runs_howManyHerbs.reshape(len(species)*len(accepted_simulations_2018), 105).transpose(),len(accepted_simulations_2018))))
    # howManyHerbs = pd.DataFrame(data=howManyHerbs, columns=species)
    # IDs_6 = np.arange(1,1 + len(accepted_simulations_2018))
    # howManyHerbs['ID'] = np.repeat(IDs_6,105)
    # howManyHerbs['time'] = all_times_howManyHerbs
    # howManyHerbs = pd.concat([filtered_FinalRuns, howManyHerbs])
    # howManyHerbs['accepted?'] = "howManyHerbs"


    # reality checks
    all_runs_realityCheck = []
    all_times_realityCheck = []
    t_realityCheck = np.linspace(0, 40, 20)
    # change X0 depending on what's needed for the reality check
    X0_5 = [1, 1, 1, 1, 1, 1, 1]
    for r_5, A_5 in zip(r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2018))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        A_5 = pd.DataFrame(data = A_5, index = species, columns = species)
        A_5['largeHerb']['largeHerb'] = -0.01
        A_5['tamworthPig']['tamworthPig'] = -0.01
        A_5['roeDeer']['roeDeer'] = -0.01
        A_5 = A_5.to_numpy()
        realityCheck_ABC = solve_ivp(ecoNetwork, (0, 40), X0_5,  t_eval = t_realityCheck, args=(A_5, r_5), method = 'RK23') 
        all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
        all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
    realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*len(accepted_simulations_2018), 20).transpose(),len(accepted_simulations_2018))))
    realityCheck = pd.DataFrame(data=realityCheck, columns=species)
    IDs_reality = np.arange(1,1 + len(accepted_simulations_2018))
    realityCheck['ID'] = np.repeat(IDs_reality,20)
    realityCheck['time'] = all_times_realityCheck
    # plot reality check
    grouping1 = np.repeat(realityCheck['ID'], len(species))
    # # extract the node values from all dataframes
    final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_realityCheck = np.tile(species, (20*len(accepted_simulations_2018)))
    # time 
    firstODEyears = np.repeat(realityCheck['time'],len(species))
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
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
    g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=4, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    plt.tight_layout()
    plt.show()

    return final_runs_3, accepted_simulations_2018, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, no_Cull




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    final_runs_3, accepted_simulations_2018, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, no_Cull = runODE_3()
    # extract accepted nodes from all dataframes
    accepted_shape1 = np.repeat(final_runs['accepted?'], len(species))
    accepted_shape2 = np.repeat(final_runs_2['accepted?'], len(species))
    accepted_shape3 = np.repeat(final_runs_3['accepted?'], len(species))
    accepted_shape4 = np.repeat(no_reintro['accepted?'], len(species))
    accepted_shape5 = np.repeat(no_Cull['accepted?'], len(species))
    # concatenate them
    accepted_shape = pd.concat([accepted_shape1, accepted_shape2, accepted_shape3, accepted_shape4, accepted_shape5], axis=0)
    # add a grouping variable to graph each run separately
    grouping1 = np.repeat(final_runs['ID'], len(species))
    grouping2 = np.repeat(final_runs_2['ID'], len(species))
    grouping3 = np.repeat(final_runs_3['ID'], len(species))
    grouping4 = np.repeat(no_reintro['ID'], len(species))
    grouping5 = np.repeat(no_Cull['ID'], len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping1, grouping2, grouping3, grouping4, grouping5), axis=0)
    # # extract the node values from all dataframes
    final_runs1 = final_runs.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    final_runs2 = final_runs_2.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    final_runs3 = final_runs_3.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    y_noReintro = no_reintro.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    y_noCull= no_Cull.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    # concatenate them
    y_values = np.concatenate((final_runs1, final_runs2, final_runs3, y_noReintro, y_noCull), axis=0)
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_firstRun = np.tile(species, 12*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 33*len(accepted_simulations))
    species_thirdRun = np.tile(species, 75*len(accepted_simulations_2018))
    species_noReintro = np.tile(species, (105*len(accepted_simulations_2018)) + (12*len(accepted_simulations)))
    species_noCull = np.tile(species, (105*len(accepted_simulations_2018)) + (12*len(accepted_simulations)))
    species_list = np.concatenate((species_firstRun, species_secondRun, species_thirdRun, species_noReintro, species_noCull), axis=0)
    # time 
    firstODEyears = np.repeat(final_runs['time'],len(species))
    secondODEyears = np.repeat(final_runs_2['time'],len(species))
    thirdODEyears = np.repeat(final_runs_3['time'],len(species))
    indices_noReintro = np.repeat(no_reintro['time'],len(species))
    indices_noCull = np.repeat(no_Cull['time'],len(species))
    indices = pd.concat([firstODEyears, secondODEyears, thirdODEyears, indices_noReintro, indices_noCull], axis=0)
    # put it in a dataframe
    final_df = pd.DataFrame(
        {'Abundance %': y_values, 'runNumber': grouping_variable, 'Ecosystem Element': species_list, 'Time': indices, 'runType': accepted_shape})
    # calculate median 
    m = final_df.groupby(['Time', 'runType','Ecosystem Element'])[['Abundance %']].apply(np.median)
    m.name = 'Median'
    final_df = final_df.join(m, on=['Time', 'runType','Ecosystem Element'])
    # calculate quantiles
    perc1 = final_df.groupby(['Time', 'runType','Ecosystem Element'])['Abundance %'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_df = final_df.join(perc1, on=['Time', 'runType','Ecosystem Element'])
    perc2 = final_df.groupby(['Time', 'runType','Ecosystem Element'])['Abundance %'].quantile(.05)
    perc2.name = "fivePerc"
    final_df = final_df.join(perc2, on=['Time','runType', 'Ecosystem Element'])
    # filter the accepted runs to graph
    filtered_df = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "noReintro") ]
    filtered_rejectedAccepted = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "Rejected") ]
    filtered_noCull= final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "noCulls") ]
    # filtered_df.to_csv("filtered_df.csv", index=False)

    # Accepted vs. Counterfactual graph (no reintroductions vs. reintroductions)
    colors = ["#6788ee", "#e26952"]
    g = sns.FacetGrid(filtered_df, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=4, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
    g.set(xticks=[0, 4, 14, 38])
    # add subplot titles
    axes = g.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("Grassland & parkland")
    axes[1].set_title("Large herbivores")
    axes[2].set_title("Organic carbon")
    axes[3].set_title("Roe deer")
    axes[4].set_title("Tamworth pigs")
    axes[5].set_title("Thorny scrubland")
    axes[6].set_title("Woodland")
    # add filter lines
    g.axes[0].vlines(x=4,ymin=0.9,ymax=1, color='r')
    g.axes[2].vlines(x=4,ymin=0.95,ymax=1.9, color='r')
    g.axes[3].vlines(x=4,ymin=1,ymax=3.3, color='r')
    g.axes[5].vlines(x=4,ymin=1,ymax=19, color='r')
    g.axes[6].vlines(x=4,ymin=0.85,ymax=1.56, color='r')
    # plot next set of filter lines
    g.axes[0].vlines(x=15,ymin=0.67,ymax=0.79, color='r')
    g.axes[2].vlines(x=15,ymin=1.7,ymax=2.2, color='r')
    g.axes[3].vlines(x=15,ymin=1.7,ymax=6.7, color='r')
    g.axes[5].vlines(x=15,ymin=22.5,ymax=35.1, color='r')
    g.axes[6].vlines(x=15,ymin=0.98,ymax=1.7, color='r')
    # make sure they all start from 0 
    g.axes[2].set(ylim =(0,None))
    g.axes[3].set(ylim =(0,None))
    g.axes[6].set(ylim =(0,None))
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Reintroductions', 'No reintroductions'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    plt.savefig('reintroNoReintro_1mil_practice.png')
    plt.show()


    # # Accepted vs. rejected runs graph
    # r = sns.FacetGrid(filtered_rejectedAccepted, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=4, sharey = False)
    # r.map(sns.lineplot, 'Time', 'Median')
    # r.map(sns.lineplot, 'Time', 'fivePerc')
    # r.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    # for ax in r.axes.flat:
    #     ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
    #     ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
    #     ax.set_ylabel('Abundance')
    # r.set(xticks=[0, 4, 14, 38])
    # # add subplot titles
    # axes = r.axes.flatten()
    # # fill between the quantiles
    # axes[0].set_title("Grassland & parkland")
    # axes[1].set_title("Large herbivores")
    # axes[2].set_title("Organic carbon")
    # axes[3].set_title("Roe deer")
    # axes[4].set_title("Tamworth pigs")
    # axes[5].set_title("Thorny scrubland")
    # axes[6].set_title("Woodland")
    # # add filter lines
    # r.axes[0].vlines(x=4,ymin=0.9,ymax=1, color='r')
    # r.axes[2].vlines(x=4,ymin=0.95,ymax=1.9, color='r')
    # r.axes[3].vlines(x=4,ymin=1,ymax=3.3, color='r')
    # r.axes[5].vlines(x=4,ymin=1,ymax=19, color='r')
    # r.axes[6].vlines(x=4,ymin=0.85,ymax=1.56, color='r')
    # # plot next set of filter lines
    # r.axes[0].vlines(x=15,ymin=0.67,ymax=0.79, color='r')
    # r.axes[2].vlines(x=15,ymin=1.7,ymax=2.2, color='r')
    # r.axes[3].vlines(x=15,ymin=1.7,ymax=6.7, color='r')
    # r.axes[5].vlines(x=15,ymin=22.5,ymax=35.1, color='r')
    # r.axes[6].vlines(x=15,ymin=0.98,ymax=1.7, color='r')
    # # make sure they all start from 0 
    # r.axes[2].set(ylim =(0,None))
    # r.axes[3].set(ylim =(0,None))
    # r.axes[6].set(ylim =(0,None))
    # # stop the plots from overlapping
    # plt.tight_layout()
    # plt.legend(labels=['Rejected Runs', 'Accepted Runs'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    # plt.savefig('acceptedRejected_1mil_practice.png')
    # plt.show()

    # Different culling levels graph
    n = sns.FacetGrid(filtered_noCull, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=4, sharey = False)
    n.map(sns.lineplot, 'Time', 'Median')
    n.map(sns.lineplot, 'Time', 'fivePerc')
    n.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in n.axes.flat:
        ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
    n.set(xticks=[0, 4, 14, 38])
    # add subplot titles
    axes = n.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("Grassland & parkland")
    axes[1].set_title("Large herbivores")
    axes[2].set_title("Organic carbon")
    axes[3].set_title("Roe deer")
    axes[4].set_title("Tamworth pigs")
    axes[5].set_title("Thorny scrubland")
    axes[6].set_title("Woodland")
    # add filter lines
    n.axes[0].vlines(x=4,ymin=0.9,ymax=1, color='r')
    n.axes[2].vlines(x=4,ymin=0.95,ymax=1.9, color='r')
    n.axes[3].vlines(x=4,ymin=1,ymax=3.3, color='r')
    n.axes[5].vlines(x=4,ymin=1,ymax=19, color='r')
    n.axes[6].vlines(x=4,ymin=0.85,ymax=1.56, color='r')
    # plot next set of filter lines
    n.axes[0].vlines(x=15,ymin=0.67,ymax=0.79, color='r')
    n.axes[2].vlines(x=15,ymin=1.7,ymax=2.2, color='r')
    n.axes[3].vlines(x=15,ymin=1.7,ymax=6.7, color='r')
    n.axes[5].vlines(x=15,ymin=22.5,ymax=35.1, color='r')
    n.axes[6].vlines(x=15,ymin=0.98,ymax=1.7, color='r')
    # make sure they all start from 0 
    n.axes[2].set(ylim =(0,None))
    n.axes[3].set(ylim =(0,None))
    n.axes[6].set(ylim =(0,None))
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Normal culling rates', 'No culling'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    # ten digit random number
    plt.savefig('cullsNoculls_1mil_practice.png')
    plt.show()


plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
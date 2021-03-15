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
totalSimulations = 1500

# store species in a list
species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','organicCarbon','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']
# define the Lotka-Volterra equation
def ecoNetwork(t, X, A, r):
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))


# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():
    # define the array
    interaction_matrix = [
                # exmoor pony
                [-0.011, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # fallow deer
                [0, -0.093, 0.36, 0, 0, 0, 0, 0, 0.012, 0.21],
                # grassland parkland
                [-0.002, -0.0032, -0.8, -0.0038, 0, -0.0047, -0.0027, -0.0068, -0.015, -0.02],
                # longhorn cattle
                [0, 0, 0.24, -0.17, 0, 0, 0, 0, 0.013, 0.17],
                # organic carbon
                [0.0026, 0.0014, 0.09, 0.0015, -0.096, 0.0018, 0.0011, 0.0012, 0.0011, 0.071],
                # red deer
                [0, 0, 0.33, 0, 0, -0.32, 0, 0, 0.017, 0.45],
                # roe deer
                [0, 0, 0.59, 0, 0, 0, -0.66, 0, 0.072, 0.34],
                # tamworth pig
                [0, 0, 0.11, 0, 0, 0, 0, -0.8, 0.055, 0.11],
                # thorny scrub
                [-0.083, -0.01, 0, -0.089, 0, -0.01, -0.052, -0.087, -0.0017, -0.022],
                # woodland
                [-0.00089, -0.0008, 0, -0.00073, 0, -0.00063, -0.00039, -0.00072, 0.00027, -0.0092]
                ]
    # generate random uniform numbers
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix


def generateGrowth():
    growthRates = [0, 0, 0.92, 0, 0, 0, 0, 0, 0.7, 0.033] 
    # multiply by a range
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),))
    growth = growthRates * variation
    # give wider bounds for woodland
    # growth[6] = np.random.uniform(low = 0, high=0.1, size = (1,))
    return growth
    

def generateX0():
    # scale everything to abundance of zero (except species to be reintroduced)
    X0 = [0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
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
#     nx.draw(networkVisual.subgraph('exmoorPony'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('tamworthPig'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('fallowDeer'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('longhornCattle'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('redDeer'), pos=pos, font_size=6, node_size = 4000, node_color='red')

#     plt.show()
# networkVisual()


# # # # --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -------


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


def runODE_1():
    # Define time points: first 5 years (2005, 2006, 2007, 2008), take 3 points per month
    t = np.linspace(0, 3.95, 12)
    t_eco = np.linspace(0, 1, 2)
    all_runs = []
    all_times = []
    all_parameters = []
    NUMBER_OF_SIMULATIONS = 0
    NUMBER_STABLE = 0
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
            # check ecological parameters (primary producers shouldn't go negative when on their own)
            all_ecoCheck = []
            for i in range(len(species)):
                X0_ecoCheck = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                X0_ecoCheck[i] = 1
                ecoCheck_ABC = solve_ivp(ecoNetwork, (0, 1), X0_ecoCheck,  t_eval = t_eco, args=(A, r), method = 'RK23') 
                all_ecoCheck = np.append(all_ecoCheck, ecoCheck_ABC.y)
            all_ecoCheck_results = (np.vstack(np.hsplit(all_ecoCheck.reshape(len(species), 20).transpose(),1)))
            all_ecoCheck_results = pd.DataFrame(data=all_ecoCheck_results, columns=species)
            # ecological reality check: primary producers should not decline with no herbivores present
            if (all_ecoCheck_results.loc[5,'grasslandParkland'] >= 1) & (all_ecoCheck_results.loc[17,'thornyScrub'] >= 1) & (all_ecoCheck_results.loc[19,'woodland'] >= 1):
                # remember the parameters used
                X0_growth = pd.concat([pd.DataFrame(X0), pd.DataFrame(r)], axis = 1)
                X0_growth.columns = ['X0','growth']
                parameters_used = pd.concat([X0_growth, pd.DataFrame(A, index = species, columns = species)])
                all_parameters.append(parameters_used)
                # run the ODE
                first_ABC = solve_ivp(ecoNetwork, (0, 3.95), X0,  t_eval = t, args=(A, r), method = 'RK23')
                # append all the runs
                all_runs = np.append(all_runs, first_ABC.y)
                all_times = np.append(all_times, first_ABC.t)
                # add one to the counter (so we know how many simulations were run in the end)
                NUMBER_OF_SIMULATIONS += 1

    # check the final runs
    print("number of stable simulations", NUMBER_STABLE)
    print("number of stable & ecologically sound simulations", NUMBER_OF_SIMULATIONS)
    
    # combine the final runs into one dataframe
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 12).transpose(),NUMBER_OF_SIMULATIONS)))
    final_runs = pd.DataFrame(data=final_runs, columns=species)
    final_runs['time'] = all_times
    # put all the parameters tried into a dataframe, and give them IDs
    all_parameters = pd.concat(all_parameters)
    all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])
    return final_runs, all_parameters, NUMBER_OF_SIMULATIONS


# --------- FILTER OUT UNREALISTIC RUNS -----------

def filterRuns_1():
    final_runs, all_parameters, NUMBER_OF_SIMULATIONS = runODE_1()
    # also ID the runs 
    IDs = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    final_runs['ID'] = np.repeat(IDs,12)
    # select only the last year
    accepted_year = final_runs.loc[final_runs['time'] == 3.95]
    with pd.option_context('display.max_columns',None):
        print(accepted_year)
    # filter the runs through conditions
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= 3.3) & (accepted_year['roeDeer'] >= 1) &
    (accepted_year['grasslandParkland'] <= 1) & (accepted_year['grasslandParkland'] >= 0.74) &
    (accepted_year['woodland'] <=1.56) & (accepted_year['woodland'] >= 0.85) &
    (accepted_year['thornyScrub'] <= 19) & (accepted_year['thornyScrub'] >= 1) &
    (accepted_year['organicCarbon'] <= 1.9) & (accepted_year['organicCarbon'] >= 0.95) 
    ]
    print("number accepted, first ODE:", accepted_simulations.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # show which runs were accepted and which were rejected
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Accepted', 'Rejected')
    return accepted_parameters, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS




# # # # ---------------------- ODE #2: Years 2009-2018 -------------------------

def generateParameters2():
    accepted_parameters, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS = filterRuns_1()
    # take the accepted growth rates
    growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    r_secondRun = growthRates_2.to_numpy()
    # make the initial conditions for this ODE = the final conditions of the first ODE
    shortenedAcceptedsimulations = accepted_simulations.drop(['ID', 'time'], axis=1).copy()
    accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']] = shortenedAcceptedsimulations.values.flatten()
    X0_2 = accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']]
    X0_2 = pd.DataFrame(X0_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    # add reintroduced species; ponies, longhorn cattle, and tamworth were reintroduced in 2009
    X0_2.loc[:, 'exmoorPony'] = [1 for i in X0_2.index]
    X0_2.loc[:, 'longhornCattle'] = [1 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [1 for i in X0_2.index]
    X0_secondRun = X0_2.to_numpy()
    # select accepted interaction strengths
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    A_secondRun = interaction_strength_2.to_numpy()
    return r_secondRun, X0_secondRun, A_secondRun, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS



# # # # # ------ SOLVE ODE #2: Post-reintroductions (2009-2018) -------

def runODE_2():
    r_secondRun, X0_secondRun, A_secondRun, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS  = generateParameters2()
    all_runs_2 = []
    all_times_2 = []
    all_parameters_2 = []
    # loop through each row of accepted parameters
    for X0_3, r_2, A_2 in zip(X0_secondRun,r_secondRun, np.array_split(A_secondRun,len(accepted_simulations))):
        # concantenate the parameters
        X0_growth_2 = pd.concat([pd.DataFrame(X0_3), pd.DataFrame(r_2)], axis = 1)
        X0_growth_2.columns = ['X0','growth']
        parameters_used_2 = pd.concat([X0_growth_2, pd.DataFrame(A_2, index = species, columns = species)])
        # 2009
        t = np.linspace(4, 4.95, 3)
        second_ABC = solve_ivp(ecoNetwork, (4,4.95), X0_3,  t_eval = t, args=(A_2, r_2), method = 'RK23')
        # 2010: fallow deer reintroduced
        starting_values_2010 = second_ABC.y[0:10, 2:3].flatten()
        starting_values_2010[0] = 0.57
        starting_values_2010[1] = 1
        starting_values_2010[3] = 1.45
        starting_values_2010[7] = 0.85
        t_1 = np.linspace(5, 5.95, 3)
        third_ABC = solve_ivp(ecoNetwork, (5,5.95), starting_values_2010,  t_eval = t_1, args=(A_2, r_2), method = 'RK23')
        # 2011
        starting_values_2011 = third_ABC.y[0:10, 2:3].flatten()
        starting_values_2011[0] = 0.65
        starting_values_2011[1] = 1.93
        starting_values_2011[3] = 1.74
        starting_values_2011[7] = 1.1
        t_2 = np.linspace(6, 6.95, 3)
        fourth_ABC = solve_ivp(ecoNetwork, (6,6.95), starting_values_2011,  t_eval = t_2, args=(A_2, r_2), method = 'RK23')
        # 2012
        starting_2012 = fourth_ABC.y[0:10, 2:3].flatten()
        starting_values_2012 = starting_2012.copy()
        starting_values_2012[0] = 0.74
        starting_values_2012[1] = 2.38
        starting_values_2012[3] = 2.19
        starting_values_2012[7] = 1.65
        t_3 = np.linspace(7, 7.95, 3)
        fifth_ABC = solve_ivp(ecoNetwork, (7,7.95), starting_values_2012,  t_eval = t_3, args=(A_2, r_2), method = 'RK23')
        # 2013: red deer reintroduced
        starting_2013 = fifth_ABC.y[0:10, 2:3].flatten()
        starting_values_2013 = starting_2013.copy()
        starting_values_2013[0] = 0.43
        starting_values_2013[1] = 2.38
        starting_values_2013[3] = 2.43
        starting_values_2013[5] = 1
        starting_values_2013[7] = 0.3
        t_4 = np.linspace(8, 8.95, 3)
        sixth_ABC = solve_ivp(ecoNetwork, (8,8.95), starting_values_2013,  t_eval = t_4, args=(A_2, r_2), method = 'RK23')
        # 2014
        starting_2014 = sixth_ABC.y[0:10, 2:3].flatten()
        starting_values_2014 = starting_2014.copy()
        starting_values_2014[0] = 0.43
        starting_values_2014[1] = 2.38
        starting_values_2014[3] = 4.98
        starting_values_2014[5] = 1
        starting_values_2014[7] = 0.9
        t_5 = np.linspace(9, 9.95, 3)
        seventh_ABC = solve_ivp(ecoNetwork, (9,9.95), starting_values_2014,  t_eval = t_5, args=(A_2, r_2), method = 'RK23')
        # 2015
        starting_values_2015 = seventh_ABC.y[0:10, 2:3].flatten()
        starting_values_2015[0] = 0.43
        starting_values_2015[1] = 2.38
        starting_values_2015[3] = 2.01
        starting_values_2015[5] = 1
        starting_values_2015[7] = 0.9
        t_2015 = np.linspace(10, 10.95, 3)
        ABC_2015 = solve_ivp(ecoNetwork, (10,10.95), starting_values_2015,  t_eval = t_2015, args=(A_2, r_2), method = 'RK23')
        last_values_2015 = ABC_2015.y[0:10, 2:3].flatten()
        # 2016
        starting_values_2016 = last_values_2015.copy()
        starting_values_2016[0] = 0.48
        starting_values_2016[1] = 3.33
        starting_values_2016[3] = 1.62
        starting_values_2016[5] = 2
        starting_values_2016[7] = 0.4
        t_2016 = np.linspace(11, 11.95, 3)
        ABC_2016 = solve_ivp(ecoNetwork, (11,11.95), starting_values_2016,  t_eval = t_2016, args=(A_2, r_2), method = 'RK23')
        last_values_2016 = ABC_2016.y[0:10, 2:3].flatten()
        # 2017
        starting_values_2017 = last_values_2016.copy()
        starting_values_2017[0] = 0.43
        starting_values_2017[1] = 3.33
        starting_values_2017[3] = 1.49
        starting_values_2017[5] = 1.08
        starting_values_2017[7] = 0.35
        t_2017 = np.linspace(12, 12.95, 3)
        ABC_2017 = solve_ivp(ecoNetwork, (12,12.95), starting_values_2017,  t_eval = t_2017, args=(A_2, r_2), method = 'RK23')
        last_values_2017 = ABC_2017.y[0:10, 2:3].flatten()
        # 2018
        starting_values_2018 = last_values_2017.copy()
        starting_values_2018[0] = 0.39
        starting_values_2018[1] = 5.98
        starting_values_2018[3] = 1.66
        starting_values_2018[5] = 1.85
        starting_values_2018[7] = 0.8
        t_2018 = np.linspace(13, 13.95, 3)
        ABC_2018 = solve_ivp(ecoNetwork, (13,13.95), starting_values_2018,  t_eval = t_2018, args=(A_2, r_2), method = 'RK23')
        last_values_2018 = ABC_2018.y[0:10, 2:3].flatten()
        # 2019
        starting_values_2019 = last_values_2018.copy()
        starting_values_2019[0] = 0
        starting_values_2019[1] = 6.62
        starting_values_2019[3] = 1.64
        starting_values_2019[5] = 2.85
        starting_values_2019[7] = 0.45
        t_2019 = np.linspace(14, 14.95, 3)
        ABC_2019 = solve_ivp(ecoNetwork, (14,14.95), starting_values_2019,  t_eval = t_2019, args=(A_2, r_2), method = 'RK23')
        last_values_2019 = ABC_2019.y[0:10, 2:3].flatten()
        # 2020
        starting_values_2020 = last_values_2019.copy()
        starting_values_2020[0] = 0.65
        starting_values_2020[1] = 5.88
        starting_values_2020[3] = 1.53
        starting_values_2020[5] = 2.7
        starting_values_2020[7] = 0.35
        t_2020 = np.linspace(15, 16, 3)
        ABC_2020 = solve_ivp(ecoNetwork, (15,16), starting_values_2020,  t_eval = t_2020, args=(A_2, r_2), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # append to dataframe
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)   
        all_times_2 = np.append(all_times_2, combined_times)
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 36).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    final_runs_2['time'] = all_times_2
    # append all the parameters to a dataframe
    all_parameters_2 = pd.concat(all_parameters_2)
    # add ID to the dataframe & parameters
    all_parameters_2['ID'] = ([(x+1) for x in range(len(accepted_simulations)) for _ in range(len(parameters_used_2))])
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,36)
    return final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = runODE_2()
    # filter 2015 values : ponies = the same, fallow deer started at 3.33 next year but 0.88 were culled; longhorn got to maximum 2.45; red got to 2 next year; pigs got to maximum 1.1
    accepted_simulations_2015 = final_runs_2[(final_runs_2['time'] == 10.95) & (final_runs_2['exmoorPony'] <= 0.47) & (final_runs_2['exmoorPony'] >= 0.39) & (final_runs_2['fallowDeer'] <= 4.63) & (final_runs_2['fallowDeer'] >= 3.79) & (final_runs_2['longhornCattle'] <= 2.78) & (final_runs_2['longhornCattle'] >= 2.23) & (final_runs_2['redDeer'] <= 2.2) & (final_runs_2['redDeer'] >= 1.8) & (final_runs_2['tamworthPig'] <= 1.27) & (final_runs_2['tamworthPig'] >= 1.04)]
    filtered_2015 = final_runs_2[final_runs_2['ID'].isin(accepted_simulations_2015['ID'])]
    print("number passed 2015 filters:", filtered_2015.shape[0]/36)
    # filter 2016 values : ponies = the same, fallow deer = 3.9 but 25 were culled; longhorn got to maximum 2.03; pig got to maximum 0.85
    accepted_simulations_2016 = filtered_2015[(filtered_2015['time'] == 11.95) & (filtered_2015['exmoorPony'] <= 0.53) & (filtered_2015['exmoorPony'] >= 0.43) & (filtered_2015['fallowDeer'] <= 4.95) & (filtered_2015['fallowDeer'] >= 4.05) & (filtered_2015['longhornCattle'] <= 2.60) & (filtered_2015['longhornCattle'] >= 2.12) & (filtered_2015['tamworthPig'] <= 0.94) & (filtered_2015['tamworthPig'] >= 0.77)]
    filtered_2016 = filtered_2015[filtered_2015['ID'].isin(accepted_simulations_2016['ID'])]
    print("number passed 2016 filters:", filtered_2016.shape[0]/36)
    # filter 2017 values : ponies = the same; fallow = 7.34 + 1.36 culled (this was maybe supplemented so no filter); cows got to max 2.06; red deer got to 1.85 + 2 culled; pig got to 1.1
    accepted_simulations_2017 = filtered_2016[(filtered_2016['time'] == 12.95) & (filtered_2016['exmoorPony'] <= 0.47) & (filtered_2016['exmoorPony'] >= 0.39) & (filtered_2016['longhornCattle'] <= 2.45) & (filtered_2016['longhornCattle'] >= 2.0) & (filtered_2016['redDeer'] <= 2.2) & (filtered_2016['redDeer'] >= 1.8) & (filtered_2016['tamworthPig'] <= 1.27) & (filtered_2016['tamworthPig'] >= 1.04)]
    filtered_2017 = filtered_2016[filtered_2016['ID'].isin(accepted_simulations_2017['ID'])]
    print("number passed 2017 filters:", filtered_2017.shape[0]/36)
    # filter 2018 values : p ponies = same, fallow = 6.62 + 57 culled; cows got to max 2.21; reds got to 2.85 + 3 culled; pigs got to max 1.15
    accepted_simulations_2018 = filtered_2017[(filtered_2017['time'] == 13.95) & (filtered_2017['exmoorPony'] <= 0.43) & (filtered_2017['exmoorPony'] >= 0.35) & (filtered_2017['fallowDeer'] <= 8.78) & (filtered_2017['fallowDeer'] >= 7.18) & (filtered_2017['longhornCattle'] <= 2.55) & (filtered_2017['longhornCattle'] >= 2.09) & (filtered_2017['redDeer'] <= 3.39) & (filtered_2017['redDeer'] >= 2.77) & (filtered_2017['tamworthPig'] <= 1.27) & (filtered_2017['tamworthPig'] >= 1.04)]
    filtered_2018 = filtered_2017[filtered_2017['ID'].isin(accepted_simulations_2018['ID'])]
    print("number passed 2018 filters:", filtered_2018.shape[0]/36)
    # filter 2019 values : ponies = 0, fallow = 6.62 + 1.36 culled; longhorn maximum 2
    accepted_simulations_2019 = filtered_2018[(filtered_2018['time'] == 14.95) & (filtered_2018['fallowDeer'] <= 8.31) & (filtered_2018['fallowDeer'] >= 6.8) & (filtered_2018['longhornCattle'] <= 2.43) & (filtered_2018['longhornCattle'] >= 1.99) & (filtered_2018['redDeer'] <= 3.73) & (filtered_2018['redDeer'] >= 3.05) & (filtered_2018['tamworthPig'] <= 1.98) & (filtered_2018['tamworthPig'] >= 1.62)]
    filtered_2019 = filtered_2018[filtered_2018['ID'].isin(accepted_simulations_2019['ID'])]
    print("number passed 2019 filters:", filtered_2019.shape[0]/36)
    # now choose just the final years (these will become the starting conditions in the next model)
    filtered_2020 = filtered_2019.loc[final_runs_2['time'] == 16]
    # filter the final 2020 runs
    accepted_simulations_2020 = filtered_2020[
    # 2020  - no filtering for fallow or red deer bc we don't know what they've grown to yet (next survey March 2021); we only know minimum pig numbers
    (final_runs_2['exmoorPony'] <= 0.72) & (final_runs_2['exmoorPony'] >= 0.59) &
    (final_runs_2['tamworthPig'] >= 0.95) &
    (final_runs_2['roeDeer'] <= 6.7) & (final_runs_2['roeDeer'] >= 1.7) &
    (final_runs_2['grasslandParkland'] <= 0.86) & (final_runs_2['grasslandParkland'] >= 0.61) &
    (final_runs_2['woodland'] <= 1.73) & (final_runs_2['woodland'] >= 0.98) &
    (final_runs_2['thornyScrub'] <= 35.1) & (final_runs_2['thornyScrub'] >= 22.5) & 
    (final_runs_2['organicCarbon'] <= 2.2) & (final_runs_2['organicCarbon'] >= 1.7)
    ]
    with pd.option_context('display.max_columns',None):
        print("number passed 2020 filters:", accepted_simulations_2020.shape, accepted_simulations_2020)

    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2020 = all_parameters_2[all_parameters_2['ID'].isin(accepted_simulations_2020['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2020['ID']), 'Accepted', 'Rejected')
    return accepted_simulations_2020, accepted_parameters_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# # # # ---------------------- ODE #3: projecting 10 years (2018-2028) -------------------------

def generateParameters3():
    accepted_simulations_2020, accepted_parameters_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun = filterRuns_2()
    # select accepted growth rates 
    growthRates_3 = accepted_parameters_2020.loc[accepted_parameters_2020['growth'].notnull(), ['growth']]
    growthRates_3 = pd.DataFrame(growthRates_3.values.reshape(len(accepted_simulations_2020), len(species)), columns = species)
    r_thirdRun = growthRates_3.to_numpy()
    # X0 - make the final conditions of ODE #2 the starting conditions
    accepted_simulations_2 = accepted_simulations_2020.drop(['ID', 'time'], axis=1).copy()
    accepted_parameters_2020.loc[accepted_parameters_2020['X0'].notnull(), ['X0']] = accepted_simulations_2.values.flatten()
    X0_3 = accepted_parameters_2020.loc[accepted_parameters_2020['X0'].notnull(), ['X0']]
    X0_3 = pd.DataFrame(X0_3.values.reshape(len(accepted_simulations_2), len(species)), columns = species)
    # select accepted interaction strengths
    interaction_strength_3 = accepted_parameters_2020.drop(['X0', 'growth', 'ID'], axis=1).copy()
    interaction_strength_3 = interaction_strength_3.dropna()
    A_thirdRun = interaction_strength_3.to_numpy()


    # HISTOGRAMS
    # growth rates
    growth_filtered = growthRates_3[["grasslandParkland","thornyScrub","woodland"]]
    fig, axes = plt.subplots(len(growth_filtered.columns)//3,3, figsize=(25, 10))
    for col, axis in zip(growth_filtered.columns, axes):
        growth_filtered.hist(column = col, ax = axis, bins = 25)

    # CORRELATION MATRIX
    growthRates_3.columns = ['exmoorPony_growth', 'fallowDeer_growth', 'grasslandParkland_growth', 'longhornCattle_growth','organicCarbon_growth', 'redDeer_growth', 'roeDeer_growth', 'tamworthPig_growth', 'thornyScrub_growth', 'woodland_growth']
    # reshape int matrix
    exmoorInts = interaction_strength_3[interaction_strength_3.index=='exmoorPony']
    exmoorInts.columns = ['pony_pony', 'pony_fallow','pony_grass','pony_cattle','pony_carbon','pony_red','pony_roe','pony_pig','pony_scrub','pony_wood']
    exmoorInts = exmoorInts.reset_index(drop=True)
    fallowInts = interaction_strength_3[interaction_strength_3.index=='fallowDeer']
    fallowInts.columns = ['fallow_pony', 'fallow_fallow', 'fallow_grass','fallow_cattle','fallow_carbon','fallow_red', 'fallow_roe', 'fallow_pig', 'fallow_scrub', 'fallow_wood']
    fallowInts = fallowInts.reset_index(drop=True)
    arableInts = interaction_strength_3[interaction_strength_3.index=='grasslandParkland']
    arableInts.columns = ['grass_pony', 'grass_fallow', 'grass_grass','grass_cattle','grass_carbon','grass_red', 'grass_roe', 'grass_pig', 'grass_scrub', 'grass_wood']
    arableInts = arableInts.reset_index(drop=True)
    longhornInts = interaction_strength_3[interaction_strength_3.index=='longhornCattle']
    longhornInts.columns = ['cattle_pony', 'cattle_fallow', 'cattle_grass','cattle_cattle','cattle_carbon','cattle_red', 'cattle_roe', 'cattle_pig', 'cattle_scrub', 'cattle_wood']
    longhornInts = longhornInts.reset_index(drop=True)
    orgCarbInts = interaction_strength_3[interaction_strength_3.index=='organicCarbon']
    orgCarbInts.columns = ['carbon_pony', 'carbon_fallow', 'carbon_grass','carbon_cattle','carbon_carbon','carbon_red', 'carbon_roe', 'carbon_pig', 'carbon_scrub', 'carbon_wood']
    orgCarbInts = orgCarbInts.reset_index(drop=True)
    redDeerInts = interaction_strength_3[interaction_strength_3.index=='redDeer']
    redDeerInts.columns = ['red_pony', 'red_fallow', 'red_grass','red_cattle','red_carbon','red_red', 'red_roe', 'red_pig', 'red_scrub', 'red_wood']
    redDeerInts = redDeerInts.reset_index(drop=True)
    roeDeerInts = interaction_strength_3[interaction_strength_3.index=='roeDeer']
    roeDeerInts.columns = ['roe_pony', 'roe_fallow', 'roe_grass','roe_cattle','roe_carbon','roe_red', 'roe_roe', 'roe_pig', 'roe_scrub', 'roe_wood']
    roeDeerInts = roeDeerInts.reset_index(drop=True)
    tamworthPigInts = interaction_strength_3[interaction_strength_3.index=='tamworthPig']
    tamworthPigInts.columns = ['pig_pony', 'pig_fallow', 'pig_grass','pig_cattle','pig_carbon','pig_red', 'pig_roe', 'pig_pig', 'pig_scrub', 'pig_wood']
    tamworthPigInts = tamworthPigInts.reset_index(drop=True)
    thornyScrubInts = interaction_strength_3[interaction_strength_3.index=='thornyScrub']
    thornyScrubInts.columns = ['scrub_pony', 'scrub_fallow', 'scrub_grass','scrub_cattle','scrub_carbon','scrub_red', 'scrub_roe', 'scrub_pig', 'scrub_scrub', 'scrub_wood']
    thornyScrubInts = thornyScrubInts.reset_index(drop=True)
    woodlandInts = interaction_strength_3[interaction_strength_3.index=='woodland']
    woodlandInts.columns = ['wood_pony', 'wood_fallow', 'wood_grass','wood_cattle','wood_carbon','wood_red', 'wood_roe', 'wood_pig', 'wood_scrub', 'wood_wood']
    woodlandInts = woodlandInts.reset_index(drop=True)
    combined = pd.concat([growthRates_3, exmoorInts, fallowInts, arableInts, longhornInts, orgCarbInts, redDeerInts, roeDeerInts, tamworthPigInts, thornyScrubInts, woodlandInts], axis=1)
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
    plt.show()
    return r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun



# # # # # ------ SOLVE ODE #3: Projecting forwards 10 years (2020-2030) -------

def runODE_3():
    r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = generateParameters3()
    # project into the future
    X0_thirdRun = X0_3.to_numpy()
    all_runs_3 = []
    all_parameters_3 = []
    all_times_3 = []
    t = np.linspace(17, 17.95, 3)

    # loop through each row of accepted parameters
    for X0_4, r_4, A_4 in zip(X0_thirdRun,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # starting values for 2021 - the stocking densities
        X0_4[0] =  0.65
        X0_4[1] =  5.88
        X0_4[3] =  1.53
        X0_4[5] =  2.69
        X0_4[7] =  0.95
        # concantenate the parameters
        X0_growth_3 = pd.concat([pd.DataFrame(X0_4), pd.DataFrame(r_4)], axis = 1)
        X0_growth_3.columns = ['X0','growth']
        parameters_used_3 = pd.concat([X0_growth_3, pd.DataFrame(A_4, index = species, columns = species)])
        # 2021
        ABC_2021 = solve_ivp(ecoNetwork, (17, 17.95), X0_4,  t_eval = t, args=(A_4, r_4), method = 'RK23')        
        # ten percent above/below 2021 values
        starting_2022 = ABC_2021.y[0:10, 2:3].flatten()
        starting_2022[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2022[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2022[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2022[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2022[7] =  np.random.uniform(low=0.86,high=1.0)
        t_1 = np.linspace(18, 18.95, 3)
        # 2022
        ABC_2022 = solve_ivp(ecoNetwork, (18, 18.95), starting_2022,  t_eval = t_1, args=(A_4, r_4), method = 'RK23')
        starting_2023 = ABC_2022.y[0:10, 2:3].flatten()
        starting_2023[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2023[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2023[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2023[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2023[7] =  np.random.uniform(low=0.86,high=1.0)
        t_2 = np.linspace(19, 19.95, 3)
        # 2023
        ABC_2023 = solve_ivp(ecoNetwork, (19, 19.95), starting_2023,  t_eval = t_2, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2024 = ABC_2023.y[0:10, 2:3].flatten()
        starting_2024[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2024[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2024[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2024[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2024[7] =  np.random.uniform(low=0.86,high=1.0)
        t_3 = np.linspace(20, 20.95, 3)
        # run the model for 2024
        ABC_2024 = solve_ivp(ecoNetwork, (20, 20.95), starting_2024,  t_eval = t_3, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2025 = ABC_2024.y[0:10, 2:3].flatten()
        starting_2025[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2025[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2025[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2025[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2025[7] =  np.random.uniform(low=0.86,high=1.0)
        t_4 = np.linspace(21, 21.95, 3)
        # run the model for 2025
        ABC_2025 = solve_ivp(ecoNetwork, (21, 21.95), starting_2025,  t_eval = t_4, args=(A_4, r_4), method = 'RK23')
        starting_2026 = ABC_2025.y[0:10, 2:3].flatten()
        starting_2026[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2026[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2026[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2026[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2026[7] =  np.random.uniform(low=0.86,high=1.0)
        t_5 = np.linspace(22, 22.95, 3)
        # 2026
        ABC_2026 = solve_ivp(ecoNetwork, (22, 22.95), starting_2026,  t_eval = t_5, args=(A_4, r_4), method = 'RK23')
        starting_2027 = ABC_2026.y[0:10, 2:3].flatten()
        starting_2027[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2027[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2027[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2027[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2027[7] =  np.random.uniform(low=0.86,high=1.0)
        t_6 = np.linspace(23, 23.95, 3)
        # 2027
        ABC_2027 = solve_ivp(ecoNetwork, (23, 23.95), starting_2027,  t_eval = t_6, args=(A_4, r_4), method = 'RK23')
        starting_2028 = ABC_2027.y[0:10, 2:3].flatten()
        starting_2028[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2028[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2028[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2028[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2028[7] =  np.random.uniform(low=0.86,high=1.0)
        t_7 = np.linspace(24, 24.95, 3)
        # 2028
        ABC_2028 = solve_ivp(ecoNetwork, (24, 24.95), starting_2028,  t_eval = t_7, args=(A_4, r_4), method = 'RK23')
        starting_2029 = ABC_2028.y[0:10, 2:3].flatten()
        starting_2029[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2029[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2029[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2029[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2029[7] =  np.random.uniform(low=0.86,high=1.0)
        t_8 = np.linspace(25, 25.95, 3)
        # 2029
        ABC_2029 = solve_ivp(ecoNetwork, (25, 25.95), starting_2028,  t_eval = t_8, args=(A_4, r_4), method = 'RK23')
        starting_2030 = ABC_2029.y[0:10, 2:3].flatten()
        starting_2030[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2030[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2030[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2030[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2030[7] =  np.random.uniform(low=0.86,high=1.0)
        t_9 = np.linspace(26, 27, 3)
        # 2030
        ABC_2030 = solve_ivp(ecoNetwork, (26, 27), starting_2030,  t_eval = t_9, args=(A_4, r_4), method = 'RK23')
        # concatenate & append all the runs
        combined_runs_2 = np.hstack((ABC_2021.y, ABC_2022.y, ABC_2023.y, ABC_2024.y, ABC_2025.y, ABC_2026.y, ABC_2027.y, ABC_2028.y, ABC_2029.y, ABC_2030.y))
        combined_times_2 = np.hstack((ABC_2021.t, ABC_2022.t, ABC_2023.t, ABC_2024.t, ABC_2025.t, ABC_2026.t, ABC_2027.t, ABC_2028.t, ABC_2029.t, ABC_2030.t))
        all_runs_3 = np.append(all_runs_3, combined_runs_2)
        # append all the parameters
        all_parameters_3.append(parameters_used_3)   
        # append the times
        all_times_3 = np.append(all_times_3, combined_times_2)
    # check the final runs
    final_runs_3 = (np.vstack(np.hsplit(all_runs_3.reshape(len(species)*len(accepted_simulations_2020), 30).transpose(),len(accepted_simulations_2020))))
    final_runs_3 = pd.DataFrame(data=final_runs_3, columns=species)
    # append all the parameters to a dataframe
    all_parameters_3 = pd.concat(all_parameters_3)
    # add ID to the dataframe & parameters
    all_parameters_3['ID'] = ([(x+1) for x in range(len(accepted_simulations_2020)) for _ in range(len(parameters_used_3))])
    IDs = np.arange(1,1 + len(accepted_simulations_2020))
    final_runs_3['ID'] = np.repeat(IDs,30)
    final_runs_3['time'] = all_times_3
    final_runs_3['accepted?'] = np.repeat('Accepted', len(final_runs_3))



    # EXPERIMENT 1: What would have happened if reintroductions hadn't occurred?
    X0_3_noReintro = pd.DataFrame(data=X0_secondRun, columns=species)
    # put herbivores to zero
    X0_3_noReintro.loc[:, 'exmoorPony'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro.loc[:, 'fallowDeer'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro.loc[:, 'longhornCattle'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro.loc[:, 'redDeer'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro.loc[:,'tamworthPig'] = [0 for i in X0_3_noReintro.index]
    X0_3_noReintro = X0_3_noReintro.to_numpy()
    # loop through each row of accepted parameters
    all_runs_2 = []
    all_times_2 = []
    t_noReintro = np.linspace(4, 27, 30)
    for X0_noReintro, r_4, A_4 in zip(X0_3_noReintro,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        noReintro_ABC = solve_ivp(ecoNetwork, (4, 27), X0_noReintro,  t_eval = t_noReintro, args=(A_4, r_4), method = 'RK23') 
        all_runs_2 = np.append(all_runs_2, noReintro_ABC.y)
        all_times_2 = np.append(all_times_2, noReintro_ABC.t)
    no_reintro = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations_2020), 30).transpose(),len(accepted_simulations_2020))))
    no_reintro = pd.DataFrame(data=no_reintro, columns=species)
    IDs_2 = np.arange(1,1 + len(accepted_simulations_2020))
    no_reintro['ID'] = np.repeat(IDs_2,30)
    no_reintro['time'] = all_times_2
    # concantenate this will the accepted runs from years 1-5
    filtered_FinalRuns = final_runs.loc[(final_runs['accepted?'] == "Accepted") ]
    no_reintro = pd.concat([filtered_FinalRuns, no_reintro])
    no_reintro['accepted?'] = "noReintro"


    # EXPERIMENT 2: What if stocking densities were double what they are currently?
    all_runs_stockingRate = []
    all_times_stockingRate = []
    t1_stockingRate = np.linspace(4, 4.95, 3)
    X0_stockingRate = X0_secondRun
    # starting values for 2009
    X0_stockingRate[0] =  0.65*2
    X0_stockingRate[3] =  1.53*2
    X0_stockingRate[7] =  0.95*2
    # loop through each row of accepted parameters
    for X0_stocking, r_5, A_5 in zip(X0_stockingRate,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        ABC_stockingRate_2009 = solve_ivp(ecoNetwork, (4,4.95), X0_stocking,  t_eval = t1_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2010 = ABC_stockingRate_2009.y[0:10, 2:3].flatten()
        # 2010: fallow deer reintroduced
        stocking_values_2010[0] =  0.65*2
        stocking_values_2010[1] =  5.88*2
        stocking_values_2010[3] =  1.53*2
        stocking_values_2010[7] =  0.95*2
        t2_stockingRate = np.linspace(5, 5.95, 3)
        ABC_stockingRate_2010 = solve_ivp(ecoNetwork, (5,5.95), stocking_values_2010,  t_eval = t2_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2011
        stocking_values_2011 = ABC_stockingRate_2010.y[0:10, 2:3].flatten()
        stocking_values_2011[0] =  0.65*2
        stocking_values_2011[1] =  5.88*2
        stocking_values_2011[3] =  1.53*2
        stocking_values_2011[7] =  0.95*2
        t3_stockingRate = np.linspace(6, 6.95, 3)
        ABC_stockingRate_2011 = solve_ivp(ecoNetwork, (6,6.95), stocking_values_2011,  t_eval = t3_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2012
        stocking_values_2012 = ABC_stockingRate_2011.y[0:10, 2:3].flatten()
        stocking_values_2012[0] =  0.65*2
        stocking_values_2012[1] =  5.88*2
        stocking_values_2012[3] =  1.53*2
        stocking_values_2012[7] =  0.95*2
        t4_stockingRate = np.linspace(7, 7.95, 3)
        ABC_stockingRate_2012 = solve_ivp(ecoNetwork, (7,7.95), stocking_values_2012,  t_eval = t4_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2013: red deer reintroduced
        stocking_values_2013 = ABC_stockingRate_2012.y[0:10, 2:3].flatten()
        stocking_values_2013[0] =  0.65*2
        stocking_values_2013[1] =  5.88*2
        stocking_values_2013[3] =  1.53*2
        stocking_values_2013[5] =  2.69*2
        stocking_values_2013[7] =  0.95*2
        t5_stockingRate = np.linspace(8, 8.95, 3)
        ABC_stockingRate_2013 = solve_ivp(ecoNetwork, (8,8.95), stocking_values_2013,  t_eval = t5_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2014
        stocking_values_2014 = ABC_stockingRate_2013.y[0:10, 2:3].flatten()
        stocking_values_2014[0] =  0.65*2
        stocking_values_2014[1] =  5.88*2
        stocking_values_2014[3] =  1.53*2
        stocking_values_2014[5] =  2.69*2
        stocking_values_2014[7] =  0.95*2
        t6_stockingRate = np.linspace(9, 9.95, 3)
        ABC_stockingRate_2014 = solve_ivp(ecoNetwork, (9,9.95), stocking_values_2014,  t_eval = t6_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2015
        stocking_values_2015 = ABC_stockingRate_2014.y[0:10, 2:3].flatten()
        stocking_values_2015[0] =  0.65*2
        stocking_values_2015[1] =  5.88*2
        stocking_values_2015[3] =  1.53*2
        stocking_values_2015[5] =  2.69*2
        stocking_values_2015[7] =  0.95*2
        t7_stockingRate = np.linspace(10, 10.95, 3)
        ABC_stockingRate_2015 = solve_ivp(ecoNetwork, (10,10.95), stocking_values_2015,  t_eval = t7_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2016 = ABC_stockingRate_2015.y[0:10, 2:3].flatten()
        # 2016
        stocking_values_2016[0] =  0.65*2
        stocking_values_2016[1] =  5.88*2
        stocking_values_2016[3] =  1.53*2
        stocking_values_2016[5] =  2.69*2
        stocking_values_2016[7] =  0.95*2
        t8_stockingRate = np.linspace(11, 11.95, 3)
        ABC_stockingRate_2016 = solve_ivp(ecoNetwork, (11,11.95), stocking_values_2016,  t_eval = t8_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2017 = ABC_stockingRate_2016.y[0:10, 2:3].flatten()
        # 2017
        stocking_values_2017[0] =  0.65*2
        stocking_values_2017[1] =  5.88*2
        stocking_values_2017[3] =  1.53*2
        stocking_values_2017[5] =  2.69*2
        stocking_values_2017[7] =  0.95*2
        t9_stockingRate = np.linspace(12, 12.95, 3)
        ABC_stockingRate_2017 = solve_ivp(ecoNetwork, (12,12.95), stocking_values_2017,  t_eval = t9_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2018
        stocking_values_2018 = ABC_stockingRate_2017.y[0:10, 2:3].flatten()
        stocking_values_2018[0] =  0.65*2
        stocking_values_2018[1] =  5.88*2
        stocking_values_2018[3] =  1.53*2
        stocking_values_2018[5] =  2.69*2
        stocking_values_2018[7] =  0.95*2
        t95_stockingRate = np.linspace(13, 13.95, 3)
        ABC_stockingRate_2018 = solve_ivp(ecoNetwork, (13,13.95), stocking_values_2018,  t_eval = t95_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2019 = ABC_stockingRate_2018.y[0:10, 2:3].flatten()
        # 2019
        stocking_values_2019[0] =  0.65*2
        stocking_values_2019[1] =  5.88*2
        stocking_values_2019[3] =  1.53*2
        stocking_values_2019[5] =  2.69*2
        stocking_values_2019[7] =  0.95*2
        t10_stockingRate = np.linspace(14, 14.95, 3)
        ABC_stockingRate_2019 = solve_ivp(ecoNetwork, (14,14.95), stocking_values_2019,  t_eval = t10_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2020 = ABC_stockingRate_2019.y[0:10, 2:3].flatten()
        # 2020
        stocking_values_2020[0] =  0.65*2
        stocking_values_2020[1] =  5.88*2
        stocking_values_2020[3] =  1.53*2
        stocking_values_2020[5] =  2.69*2
        stocking_values_2020[7] =  0.95*2
        t11_stockingRate = np.linspace(15, 15.95, 3)
        ABC_stockingRate_2020 = solve_ivp(ecoNetwork, (15,15.95), stocking_values_2020,  t_eval = t11_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2021 = ABC_stockingRate_2020.y[0:10, 2:3].flatten()
        # 2021 - future projections
        stocking_values_2021[0] =  0.65*2
        stocking_values_2021[1] =  5.88*2
        stocking_values_2021[3] =  1.53*2
        stocking_values_2021[5] =  2.69*2
        stocking_values_2021[7] =  0.95*2
        t12_stockingRate = np.linspace(16, 16.95, 3)
        ABC_stockingRate_2021 = solve_ivp(ecoNetwork, (16,16.95), stocking_values_2021,  t_eval = t12_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2022 = ABC_stockingRate_2021.y[0:10, 2:3].flatten()
        # 2022
        stocking_values_2022[0] =  0.65*2
        stocking_values_2022[1] =  5.88*2
        stocking_values_2022[3] =  1.53*2
        stocking_values_2022[5] =  2.69*2
        stocking_values_2022[7] =  0.95*2
        t13_stockingRate = np.linspace(17, 17.95, 3)
        ABC_stockingRate_2022 = solve_ivp(ecoNetwork, (17,17.95), stocking_values_2022,  t_eval = t13_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2023 = ABC_stockingRate_2022.y[0:10, 2:3].flatten()
        # 2023
        stocking_values_2023[0] =  0.65*2
        stocking_values_2023[1] =  5.88*2
        stocking_values_2023[3] =  1.53*2
        stocking_values_2023[5] =  2.69*2
        stocking_values_2023[7] =  0.95*2
        t14_stockingRate = np.linspace(18, 18.95, 3)
        ABC_stockingRate_2023 = solve_ivp(ecoNetwork, (18,18.95), stocking_values_2023,  t_eval = t14_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2024 = ABC_stockingRate_2023.y[0:10, 2:3].flatten()
        # 2024
        stocking_values_2024[0] =  0.65*2
        stocking_values_2024[1] =  5.88*2
        stocking_values_2024[3] =  1.53*2
        stocking_values_2024[5] =  2.69*2
        stocking_values_2024[7] =  0.95*2
        t15_stockingRate = np.linspace(19, 19.95, 3)
        ABC_stockingRate_2024 = solve_ivp(ecoNetwork, (19,19.95), stocking_values_2024,  t_eval = t15_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2025 = ABC_stockingRate_2024.y[0:10, 2:3].flatten()
        # 2025
        stocking_values_2025[0] =  0.65*2
        stocking_values_2025[1] =  5.88*2
        stocking_values_2025[3] =  1.53*2
        stocking_values_2025[5] =  2.69*2
        stocking_values_2025[7] =  0.95*2
        t16_stockingRate = np.linspace(20, 20.95, 3)
        ABC_stockingRate_2025 = solve_ivp(ecoNetwork, (20,20.95), stocking_values_2025,  t_eval = t16_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2026 = ABC_stockingRate_2025.y[0:10, 2:3].flatten()
        # 2026
        stocking_values_2026[0] =  0.65*2
        stocking_values_2026[1] =  5.88*2
        stocking_values_2026[3] =  1.53*2
        stocking_values_2026[5] =  2.69*2
        stocking_values_2026[7] =  0.95*2
        t17_stockingRate = np.linspace(21, 21.95, 3)
        ABC_stockingRate_2026 = solve_ivp(ecoNetwork, (21,21.95), stocking_values_2026,  t_eval = t17_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2027 = ABC_stockingRate_2026.y[0:10, 2:3].flatten()
        # 2027
        stocking_values_2027[0] =  0.65*2
        stocking_values_2027[1] =  5.88*2
        stocking_values_2027[3] =  1.53*2
        stocking_values_2027[5] =  2.69*2
        stocking_values_2027[7] =  0.95*2
        t18_stockingRate = np.linspace(22, 22.95, 3)
        ABC_stockingRate_2027 = solve_ivp(ecoNetwork, (22,22.95), stocking_values_2027,  t_eval = t18_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2028 = ABC_stockingRate_2027.y[0:10, 2:3].flatten()
        # 2028
        stocking_values_2028[0] =  0.65*2
        stocking_values_2028[1] =  5.88*2
        stocking_values_2028[3] =  1.53*2
        stocking_values_2028[5] =  2.69*2
        stocking_values_2028[7] =  0.95*2
        t19_stockingRate = np.linspace(23, 23.95, 3)
        ABC_stockingRate_2028 = solve_ivp(ecoNetwork, (23,23.95), stocking_values_2028,  t_eval = t19_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2029 = ABC_stockingRate_2028.y[0:10, 2:3].flatten()
        # 2029
        stocking_values_2029[0] =  0.65*2
        stocking_values_2029[1] =  5.88*2
        stocking_values_2029[3] =  1.53*2
        stocking_values_2029[5] =  2.69*2
        stocking_values_2029[7] =  0.95*2
        t20_stockingRate = np.linspace(24, 24.95, 3)
        ABC_stockingRate_2029 = solve_ivp(ecoNetwork, (24,24.95), stocking_values_2029,  t_eval = t20_stockingRate, args=(A_5, r_5), method = 'RK23')
        stocking_values_2030 = ABC_stockingRate_2029.y[0:10, 2:3].flatten()
        # 2030
        stocking_values_2030[0] =  0.65*2
        stocking_values_2030[1] =  5.88*2
        stocking_values_2030[3] =  1.53*2
        stocking_values_2030[5] =  2.69*2
        stocking_values_2030[7] =  0.95*2
        t21_stockingRate = np.linspace(25, 26, 3)
        ABC_stockingRate_2030 = solve_ivp(ecoNetwork, (25, 26), stocking_values_2030,  t_eval = t21_stockingRate, args=(A_5, r_5), method = 'RK23')
        # concantenate the runs
        combined_runs_stockingRate = np.hstack((ABC_stockingRate_2009.y, ABC_stockingRate_2010.y, ABC_stockingRate_2011.y, ABC_stockingRate_2012.y, ABC_stockingRate_2013.y, ABC_stockingRate_2014.y, ABC_stockingRate_2015.y, ABC_stockingRate_2016.y, ABC_stockingRate_2017.y, ABC_stockingRate_2018.y, ABC_stockingRate_2019.y, ABC_stockingRate_2020.y, ABC_stockingRate_2021.y, ABC_stockingRate_2022.y, ABC_stockingRate_2023.y, ABC_stockingRate_2024.y, ABC_stockingRate_2025.y, ABC_stockingRate_2026.y, ABC_stockingRate_2027.y, ABC_stockingRate_2028.y, ABC_stockingRate_2029.y, ABC_stockingRate_2030.y))
        combined_times_stockingRate = np.hstack((ABC_stockingRate_2009.t, ABC_stockingRate_2010.t, ABC_stockingRate_2011.t, ABC_stockingRate_2012.t, ABC_stockingRate_2013.t, ABC_stockingRate_2014.t, ABC_stockingRate_2015.t, ABC_stockingRate_2016.t, ABC_stockingRate_2017.t, ABC_stockingRate_2018.t, ABC_stockingRate_2019.t, ABC_stockingRate_2020.t, ABC_stockingRate_2021.t, ABC_stockingRate_2022.t, ABC_stockingRate_2023.t, ABC_stockingRate_2024.t, ABC_stockingRate_2025.t, ABC_stockingRate_2026.t, ABC_stockingRate_2027.t, ABC_stockingRate_2028.t, ABC_stockingRate_2029.t, ABC_stockingRate_2030.t))
        all_runs_stockingRate = np.append(all_runs_stockingRate, combined_runs_stockingRate)
        all_times_stockingRate = np.append(all_times_stockingRate, combined_times_stockingRate)
    stockingValues_double = (np.vstack(np.hsplit(all_runs_stockingRate.reshape(len(species)*len(accepted_simulations_2020), 66).transpose(),len(accepted_simulations_2020))))
    stockingValues_double = pd.DataFrame(data=stockingValues_double, columns=species)
    IDs_3 = np.arange(1,1 + len(accepted_simulations_2020))
    stockingValues_double['ID'] = np.repeat(IDs_3,66)
    stockingValues_double['time'] = all_times_stockingRate
    stockingValues_double = pd.concat([filtered_FinalRuns, stockingValues_double])
    stockingValues_double['accepted?'] = "stockingDensity_double"


    # EXPERIMENT 3: What if stocking densities were half what they are currently?
    all_runs_stockingRate_half = []
    all_times_stockingRate_half = []
    # loop through each row of accepted parameters
    for X0_stocking, r_5, A_5 in zip(X0_stockingRate,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        ABC_halfStockingRate_2009 = solve_ivp(ecoNetwork, (4,4.95), X0_stocking,  t_eval = t1_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2010 = ABC_halfStockingRate_2009.y[0:10, 2:3].flatten()
        # 2010: fallow deer reintroduced
        half_stocking_values_2010[0] =  0.65/2
        half_stocking_values_2010[1] =  5.88/2
        half_stocking_values_2010[3] =  1.53/2
        half_stocking_values_2010[7] =  0.95/2
        ABC_halfStockingRate_2010 = solve_ivp(ecoNetwork, (5,5.95), half_stocking_values_2010,  t_eval = t2_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2011
        half_stocking_values_2011 = ABC_halfStockingRate_2010.y[0:10, 2:3].flatten()
        half_stocking_values_2011[0] =  0.65/2
        half_stocking_values_2011[1] =  5.88/2
        half_stocking_values_2011[3] =  1.53/2
        half_stocking_values_2011[7] =  0.95/2
        ABC_halfStockingRate_2011 = solve_ivp(ecoNetwork, (6,6.95), half_stocking_values_2011,  t_eval = t3_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2012
        half_stocking_values_2012 = ABC_halfStockingRate_2011.y[0:10, 2:3].flatten()
        half_stocking_values_2012[0] =  0.65/2
        half_stocking_values_2012[1] =  5.88/2
        half_stocking_values_2012[3] =  1.53/2
        half_stocking_values_2012[7] =  0.95/2
        ABC_halfStockingRate_2012 = solve_ivp(ecoNetwork, (7,7.95), half_stocking_values_2012,  t_eval = t4_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2013: red deer reintroduced
        half_stocking_values_2013 = ABC_halfStockingRate_2012.y[0:10, 2:3].flatten()
        half_stocking_values_2013[0] =  0.65/2
        half_stocking_values_2013[1] =  5.88/2
        half_stocking_values_2013[3] =  1.53/2
        half_stocking_values_2013[5] =  2.69/2
        half_stocking_values_2013[7] =  0.95/2
        t5_stockingRate = np.linspace(8, 8.95, 3)
        ABC_halfStockingRate_2013 = solve_ivp(ecoNetwork, (8,8.95), half_stocking_values_2013,  t_eval = t5_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2014
        half_stocking_values_2014 = ABC_halfStockingRate_2013.y[0:10, 2:3].flatten()
        half_stocking_values_2014[0] =  0.65/2
        half_stocking_values_2014[1] =  5.88/2
        half_stocking_values_2014[3] =  1.53/2
        half_stocking_values_2014[5] =  2.69/2
        half_stocking_values_2014[7] =  0.95/2
        ABC_halfStockingRate_2014 = solve_ivp(ecoNetwork, (9,9.95), half_stocking_values_2014,  t_eval = t6_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2015
        half_stocking_values_2015 = ABC_halfStockingRate_2014.y[0:10, 2:3].flatten()
        half_stocking_values_2015[0] =  0.65/2
        half_stocking_values_2015[1] =  5.88/2
        half_stocking_values_2015[3] =  1.53/2
        half_stocking_values_2015[5] =  2.69/2
        half_stocking_values_2015[7] =  0.95/2
        ABC_halfStockingRate_2015 = solve_ivp(ecoNetwork, (10,10.95), half_stocking_values_2015,  t_eval = t7_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2016 = ABC_halfStockingRate_2015.y[0:10, 2:3].flatten()
        # 2016
        half_stocking_values_2016[0] =  0.65/2
        half_stocking_values_2016[1] =  5.88/2
        half_stocking_values_2016[3] =  1.53/2
        half_stocking_values_2016[5] =  2.69/2
        half_stocking_values_2016[7] =  0.95/2
        ABC_halfStockingRate_2016 = solve_ivp(ecoNetwork, (11,11.95), half_stocking_values_2016,  t_eval = t8_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2017 = ABC_halfStockingRate_2016.y[0:10, 2:3].flatten()
        # 2017
        half_stocking_values_2017[0] =  0.65/2
        half_stocking_values_2017[1] =  5.88/2
        half_stocking_values_2017[3] =  1.53/2
        half_stocking_values_2017[5] =  2.69/2
        half_stocking_values_2017[7] =  0.95/2
        ABC_halfStockingRate_2017 = solve_ivp(ecoNetwork, (12,12.95), half_stocking_values_2017,  t_eval = t9_stockingRate, args=(A_5, r_5), method = 'RK23')
        # 2018
        half_stocking_values_2018 = ABC_halfStockingRate_2017.y[0:10, 2:3].flatten()
        half_stocking_values_2018[0] =  0.65/2
        half_stocking_values_2018[1] =  5.88/2
        half_stocking_values_2018[3] =  1.53/2
        half_stocking_values_2018[5] =  2.69/2
        half_stocking_values_2018[7] =  0.95/2
        ABC_halfStockingRate_2018 = solve_ivp(ecoNetwork, (13,13.95), half_stocking_values_2018,  t_eval = t95_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2019 = ABC_halfStockingRate_2018.y[0:10, 2:3].flatten()
        # 2019
        half_stocking_values_2019[0] =  0.65/2
        half_stocking_values_2019[1] =  5.88/2
        half_stocking_values_2019[3] =  1.53/2
        half_stocking_values_2019[5] =  2.69/2
        half_stocking_values_2019[7] =  0.95/2
        ABC_halfStockingRate_2019 = solve_ivp(ecoNetwork, (14,14.95), half_stocking_values_2019,  t_eval = t10_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2020 = ABC_halfStockingRate_2019.y[0:10, 2:3].flatten()
        # 2020
        half_stocking_values_2020[0] =  0.65/2
        half_stocking_values_2020[1] =  5.88/2
        half_stocking_values_2020[3] =  1.53/2
        half_stocking_values_2020[5] =  2.69/2
        half_stocking_values_2020[7] =  0.95/2
        t11_stockingRate = np.linspace(15, 15.95, 3)
        ABC_halfStockingRate_2020 = solve_ivp(ecoNetwork, (15,15.95), half_stocking_values_2020,  t_eval = t11_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2021 = ABC_halfStockingRate_2020.y[0:10, 2:3].flatten()
        # 2021 - future projections
        half_stocking_values_2021[0] =  0.65/2
        half_stocking_values_2021[1] =  5.88/2
        half_stocking_values_2021[3] =  1.53/2
        half_stocking_values_2021[5] =  2.69/2
        half_stocking_values_2021[7] =  0.95/2
        t12_stockingRate = np.linspace(16, 16.95, 3)
        ABC_halfStockingRate_2021 = solve_ivp(ecoNetwork, (16,16.95), half_stocking_values_2021,  t_eval = t12_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2022 = ABC_halfStockingRate_2021.y[0:10, 2:3].flatten()
        # 2022
        half_stocking_values_2022[0] =  0.65/2
        half_stocking_values_2022[1] =  5.88/2
        half_stocking_values_2022[3] =  1.53/2
        half_stocking_values_2022[5] =  2.69/2
        half_stocking_values_2022[7] =  0.95/2
        ABC_halfStockingRate_2022 = solve_ivp(ecoNetwork, (17,17.95), half_stocking_values_2022,  t_eval = t13_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2023 = ABC_halfStockingRate_2022.y[0:10, 2:3].flatten()
        # 2023
        half_stocking_values_2023[0] =  0.65/2
        half_stocking_values_2023[1] =  5.88/2
        half_stocking_values_2023[3] =  1.53/2
        half_stocking_values_2023[5] =  2.69/2
        half_stocking_values_2023[7] =  0.95/2
        ABC_halfStockingRate_2023 = solve_ivp(ecoNetwork, (18,18.95), half_stocking_values_2023,  t_eval = t14_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2024 = ABC_halfStockingRate_2023.y[0:10, 2:3].flatten()
        # 2024
        half_stocking_values_2024[0] =  0.65/2
        half_stocking_values_2024[1] =  5.88/2
        half_stocking_values_2024[3] =  1.53/2
        half_stocking_values_2024[5] =  2.69/2
        half_stocking_values_2024[7] =  0.95/2
        ABC_halfStockingRate_2024 = solve_ivp(ecoNetwork, (19,19.95), half_stocking_values_2024,  t_eval = t15_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2025 = ABC_halfStockingRate_2024.y[0:10, 2:3].flatten()
        # 2025
        half_stocking_values_2025[0] =  0.65/2
        half_stocking_values_2025[1] =  5.88/2
        half_stocking_values_2025[3] =  1.53/2
        half_stocking_values_2025[5] =  2.69/2
        half_stocking_values_2025[7] =  0.95/2
        ABC_halfStockingRate_2025 = solve_ivp(ecoNetwork, (20,20.95), half_stocking_values_2025,  t_eval = t16_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2026 = ABC_halfStockingRate_2025.y[0:10, 2:3].flatten()
        # 2026
        half_stocking_values_2026[0] =  0.65/2
        half_stocking_values_2026[1] =  5.88/2
        half_stocking_values_2026[3] =  1.53/2
        half_stocking_values_2026[5] =  2.69/2
        half_stocking_values_2026[7] =  0.95/2
        ABC_halfStockingRate_2026 = solve_ivp(ecoNetwork, (21,21.95), half_stocking_values_2026,  t_eval = t17_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2027 = ABC_halfStockingRate_2026.y[0:10, 2:3].flatten()
        # 2027
        half_stocking_values_2027[0] =  0.65/2
        half_stocking_values_2027[1] =  5.88/2
        half_stocking_values_2027[3] =  1.53/2
        half_stocking_values_2027[5] =  2.69/2
        half_stocking_values_2027[7] =  0.95/2
        ABC_halfStockingRate_2027 = solve_ivp(ecoNetwork, (22,22.95), half_stocking_values_2027,  t_eval = t18_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2028 = ABC_halfStockingRate_2027.y[0:10, 2:3].flatten()
        # 2028
        half_stocking_values_2028[0] =  0.65/2
        half_stocking_values_2028[1] =  5.88/2
        half_stocking_values_2028[3] =  1.53/2
        half_stocking_values_2028[5] =  2.69/2
        half_stocking_values_2028[7] =  0.95/2
        ABC_halfStockingRate_2028 = solve_ivp(ecoNetwork, (23,23.95), half_stocking_values_2028,  t_eval = t19_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2029 = ABC_halfStockingRate_2028.y[0:10, 2:3].flatten()
        # 2029
        half_stocking_values_2029[0] =  0.65/2
        half_stocking_values_2029[1] =  5.88/2
        half_stocking_values_2029[3] =  1.53/2
        half_stocking_values_2029[5] =  2.69/2
        half_stocking_values_2029[7] =  0.95/2
        t20_stockingRate = np.linspace(24, 24.95, 3)
        ABC_halfStockingRate_2029 = solve_ivp(ecoNetwork, (24,24.95), half_stocking_values_2029,  t_eval = t20_stockingRate, args=(A_5, r_5), method = 'RK23')
        half_stocking_values_2030 = ABC_halfStockingRate_2029.y[0:10, 2:3].flatten()
        # 2030
        half_stocking_values_2030[0] =  0.65/2
        half_stocking_values_2030[1] =  5.88/2
        half_stocking_values_2030[3] =  1.53/2
        half_stocking_values_2030[5] =  2.69/2
        half_stocking_values_2030[7] =  0.95/2
        ABC_halfStockingRate_2030 = solve_ivp(ecoNetwork, (25, 26), half_stocking_values_2030,  t_eval = t21_stockingRate, args=(A_5, r_5), method = 'RK23')
        # append the runs
        combined_runs_stockingRate_half = np.hstack((ABC_halfStockingRate_2009.y, ABC_halfStockingRate_2010.y, ABC_halfStockingRate_2011.y, ABC_halfStockingRate_2012.y, ABC_halfStockingRate_2013.y, ABC_halfStockingRate_2014.y, ABC_halfStockingRate_2015.y, ABC_halfStockingRate_2016.y, ABC_halfStockingRate_2017.y, ABC_halfStockingRate_2018.y, ABC_halfStockingRate_2019.y, ABC_halfStockingRate_2020.y, ABC_halfStockingRate_2021.y, ABC_halfStockingRate_2022.y, ABC_halfStockingRate_2023.y, ABC_halfStockingRate_2024.y, ABC_halfStockingRate_2025.y, ABC_halfStockingRate_2026.y, ABC_halfStockingRate_2027.y, ABC_halfStockingRate_2028.y, ABC_halfStockingRate_2029.y, ABC_halfStockingRate_2030.y))
        combined_times_stockingRate_half = np.hstack((ABC_halfStockingRate_2009.t, ABC_halfStockingRate_2010.t, ABC_halfStockingRate_2011.t, ABC_halfStockingRate_2012.t, ABC_halfStockingRate_2013.t, ABC_halfStockingRate_2014.t, ABC_halfStockingRate_2015.t, ABC_halfStockingRate_2016.t, ABC_halfStockingRate_2017.t, ABC_halfStockingRate_2018.t, ABC_halfStockingRate_2019.t, ABC_halfStockingRate_2020.t, ABC_halfStockingRate_2021.t, ABC_halfStockingRate_2022.t, ABC_halfStockingRate_2023.t, ABC_halfStockingRate_2024.t, ABC_halfStockingRate_2025.t, ABC_halfStockingRate_2026.t, ABC_halfStockingRate_2027.t, ABC_halfStockingRate_2028.t, ABC_halfStockingRate_2029.t, ABC_halfStockingRate_2030.t))
        all_runs_stockingRate_half = np.append(all_runs_stockingRate_half, combined_runs_stockingRate_half)
        all_times_stockingRate_half = np.append(all_times_stockingRate_half, combined_times_stockingRate_half)
    stockingValues_half = (np.vstack(np.hsplit(all_runs_stockingRate_half.reshape(len(species)*len(accepted_simulations_2020), 66).transpose(),len(accepted_simulations_2020))))
    stockingValues_half = pd.DataFrame(data=stockingValues_half, columns=species)
    stockingValues_half['ID'] = np.repeat(IDs_3,66)
    stockingValues_half['time'] = all_times_stockingRate_half
    stockingValues_half = pd.concat([filtered_FinalRuns, stockingValues_half])
    stockingValues_half['accepted?'] = "stockingDensity_half"

    # # reality checks
    # all_runs_realityCheck = []
    # all_times_realityCheck = []
    # t_realityCheck = np.linspace(0, 40, 20)
    # # change X0 depending on what's needed for the reality check
    # X0_5 = [1, 1, 1, 1, 1, 1, 1]
    # for r_5, A_5 in zip(r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
    #     # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
    #     A_5 = pd.DataFrame(data = A_5, index = species, columns = species)
    #     A_5['largeHerb']['largeHerb'] = -0.01
    #     A_5['tamworthPig']['tamworthPig'] = -0.01
    #     A_5['roeDeer']['roeDeer'] = -0.01
    #     A_5 = A_5.to_numpy()
    #     realityCheck_ABC = solve_ivp(ecoNetwork, (0, 40), X0_5,  t_eval = t_realityCheck, args=(A_5, r_5), method = 'RK23') 
    #     all_runs_realityCheck = np.append(all_runs_realityCheck, realityCheck_ABC.y)
    #     all_times_realityCheck = np.append(all_times_realityCheck, realityCheck_ABC.t)
    # realityCheck = (np.vstack(np.hsplit(all_runs_realityCheck.reshape(len(species)*len(accepted_simulations_2020), 20).transpose(),len(accepted_simulations_2020))))
    # realityCheck = pd.DataFrame(data=realityCheck, columns=species)
    # IDs_reality = np.arange(1,1 + len(accepted_simulations_2020))
    # realityCheck['ID'] = np.repeat(IDs_reality,20)
    # realityCheck['time'] = all_times_realityCheck
    # # plot reality check
    # grouping1 = np.repeat(realityCheck['ID'], len(species))
    # # # extract the node values from all dataframes
    # final_runs1 = realityCheck.drop(['ID', 'time'], axis=1).values.flatten()
    # # we want species column to be spec1,spec2,spec3,spec4, etc.
    # species_realityCheck = np.tile(species, (20*len(accepted_simulations_2020)))
    # # time 
    # firstODEyears = np.repeat(realityCheck['time'],len(species))
    # # put it in a dataframe
    # final_df = pd.DataFrame(
    #     {'Abundance %': final_runs1, 'runNumber': grouping1, 'Ecosystem Element': species_realityCheck, 'Time': firstODEyears})
    # # calculate median 
    # m = final_df.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    # m.name = 'Median'
    # final_df = final_df.join(m, on=['Time','Ecosystem Element'])
    # # calculate quantiles
    # perc1 = final_df.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    # perc1.name = 'ninetyfivePerc'
    # final_df = final_df.join(perc1, on=['Time','Ecosystem Element'])
    # perc2 = final_df.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    # perc2.name = "fivePerc"
    # final_df = final_df.join(perc2, on=['Time','Ecosystem Element'])
    # # graph it
    # g = sns.FacetGrid(final_df, col="Ecosystem Element", col_wrap=4, sharey = False)
    # g.map(sns.lineplot, 'Time', 'Median')
    # g.map(sns.lineplot, 'Time', 'fivePerc')
    # g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    # for ax in g.axes.flat:
    #     ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    # plt.tight_layout()
    # plt.show()

    return final_runs_3, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, stockingValues_double, stockingValues_half




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    final_runs_3, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, stockingValues_double, stockingValues_half = runODE_3()
    # extract accepted nodes from all dataframes
    accepted_shape1 = np.repeat(final_runs['accepted?'], len(species))
    accepted_shape2 = np.repeat(final_runs_2['accepted?'], len(species))
    accepted_shape3 = np.repeat(final_runs_3['accepted?'], len(species))
    accepted_shape4 = np.repeat(no_reintro['accepted?'], len(species))
    accepted_shape5 = np.repeat(stockingValues_double['accepted?'], len(species))
    accepted_shape6 = np.repeat(stockingValues_half['accepted?'], len(species))
    # concatenate them
    accepted_shape = pd.concat([accepted_shape1, accepted_shape2, accepted_shape3, accepted_shape4, accepted_shape5, accepted_shape6], axis=0)
    # add a grouping variable to graph each run separately
    grouping1 = np.repeat(final_runs['ID'], len(species))
    grouping2 = np.repeat(final_runs_2['ID'], len(species))
    grouping3 = np.repeat(final_runs_3['ID'], len(species))
    grouping4 = np.repeat(no_reintro['ID'], len(species))
    grouping5 = np.repeat(stockingValues_double['ID'], len(species))
    grouping6 = np.repeat(stockingValues_half['ID'], len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping1, grouping2, grouping3, grouping4, grouping5, grouping6), axis=0)
    # # extract the node values from all dataframes
    final_runs1 = final_runs.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    final_runs2 = final_runs_2.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    final_runs3 = final_runs_3.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    y_noReintro = no_reintro.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    y_stocking_double = stockingValues_double.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    y_stocking_half = stockingValues_half.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    # concatenate them
    y_values = np.concatenate((final_runs1, final_runs2, final_runs3, y_noReintro, y_stocking_double, y_stocking_half), axis=0)
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_firstRun = np.tile(species, 12*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 36*len(accepted_simulations))
    species_thirdRun = np.tile(species, 30*len(accepted_simulations_2020))
    species_noReintro = np.tile(species, (30*len(accepted_simulations_2020)) + (12*len(accepted_simulations)))
    species_stocking_double = np.tile(species, (66*len(accepted_simulations_2020)) + (12*len(accepted_simulations)))
    species_stocking_half = np.tile(species, (66*len(accepted_simulations_2020)) + (12*len(accepted_simulations)))
    species_list = np.concatenate((species_firstRun, species_secondRun, species_thirdRun, species_noReintro, species_stocking_double, species_stocking_half), axis=0)
    # time 
    firstODEyears = np.repeat(final_runs['time'],len(species))
    secondODEyears = np.repeat(final_runs_2['time'],len(species))
    thirdODEyears = np.repeat(final_runs_3['time'],len(species))
    indices_noReintro = np.repeat(no_reintro['time'],len(species))
    indices_stocking_double = np.repeat(stockingValues_double['time'],len(species))
    indices_stocking_half = np.repeat(stockingValues_half['time'],len(species))
    indices = pd.concat([firstODEyears, secondODEyears, thirdODEyears, indices_noReintro, indices_stocking_double, indices_stocking_half], axis=0)
    
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
    # filter the accepted runs to graph: 1) reintro vs. no reintro; 2) accepted vs rejected runs; 3) current vs double vs half stocking rates - are there phase shifts?
    filtered_df = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "noReintro") ]
    filtered_rejectedAccepted = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "Rejected") ]
    filtered_stockingDensity = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "stockingDensity_double") | (final_df['runType'] == "stockingDensity_half") ]

    # Accepted vs. Counterfactual graph (no reintroductions vs. reintroductions)
    colors = ["#6788ee", "#e26952"]
    g = sns.FacetGrid(filtered_df, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=5, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
    g.set(xticks=[0, 4, 16, 27])
    # add subplot titles
    axes = g.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("Exmoor ponies")
    axes[1].set_title("Fallow deer")
    axes[2].set_title("Grassland & parkland")
    axes[3].set_title("Longhorn cattle")
    axes[4].set_title("Organic carbon")
    axes[5].set_title("Red deer")
    axes[6].set_title("Roe deer")
    axes[7].set_title("Tamworth pigs")
    axes[8].set_title("Thorny scrubland")
    axes[9].set_title("Woodland")
    # add filter lines
    g.axes[2].vlines(x=4,ymin=0.9,ymax=1, color='r')
    g.axes[4].vlines(x=4,ymin=0.95,ymax=1.9, color='r')
    g.axes[6].vlines(x=4,ymin=1,ymax=3.3, color='r')
    g.axes[8].vlines(x=4,ymin=1,ymax=19, color='r')
    g.axes[9].vlines(x=4,ymin=0.85,ymax=1.56, color='r')
    # plot next set of filter lines
    g.axes[2].vlines(x=16,ymin=0.67,ymax=0.79, color='r')
    g.axes[4].vlines(x=16,ymin=1.7,ymax=2.2, color='r')
    g.axes[6].vlines(x=16,ymin=1.7,ymax=6.7, color='r')
    g.axes[8].vlines(x=16,ymin=22.5,ymax=35.1, color='r')
    g.axes[9].vlines(x=16,ymin=0.98,ymax=1.7, color='r')
    # make sure they all start from 0 
    g.axes.set(ylim =(0,None))
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Reintroductions', 'No reintroductions'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    plt.savefig('reintroNoReintro_1mil_practice.png')
    plt.show()


    # Accepted vs. rejected runs graph
    r = sns.FacetGrid(filtered_rejectedAccepted, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=5, sharey = False)
    r.map(sns.lineplot, 'Time', 'Median')
    r.map(sns.lineplot, 'Time', 'fivePerc')
    r.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in r.axes.flat:
        ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
    r.set(xticks=[0, 4, 16, 27])
    # add subplot titles
    axes = r.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("Exmoor ponies")
    axes[1].set_title("Fallow deer")
    axes[2].set_title("Grassland & parkland")
    axes[3].set_title("Longhorn cattle")
    axes[4].set_title("Organic carbon")
    axes[5].set_title("Red deer")
    axes[6].set_title("Roe deer")
    axes[7].set_title("Tamworth pigs")
    axes[8].set_title("Thorny scrubland")
    axes[9].set_title("Woodland")
    # add filter lines
    r.axes[2].vlines(x=4,ymin=0.9,ymax=1, color='r')
    r.axes[4].vlines(x=4,ymin=0.95,ymax=1.9, color='r')
    r.axes[6].vlines(x=4,ymin=1,ymax=3.3, color='r')
    r.axes[8].vlines(x=4,ymin=1,ymax=19, color='r')
    r.axes[9].vlines(x=4,ymin=0.85,ymax=1.56, color='r')
    # plot next set of filter lines
    r.axes[2].vlines(x=16,ymin=0.67,ymax=0.79, color='r')
    r.axes[4].vlines(x=16,ymin=1.7,ymax=2.2, color='r')
    r.axes[6].vlines(x=16,ymin=1.7,ymax=6.7, color='r')
    r.axes[8].vlines(x=16,ymin=22.5,ymax=35.1, color='r')
    r.axes[9].vlines(x=16,ymin=0.98,ymax=1.7, color='r')
    # make sure they all start from 0 
    r.axes.set(ylim =(0,None))
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Rejected Runs', 'Accepted Runs'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    plt.savefig('acceptedRejected_1mil_practice.png')
    plt.show()


    # Different stocking densities graph
    n = sns.FacetGrid(filtered_stockingDensity, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=5, sharey = False)
    n.map(sns.lineplot, 'Time', 'Median')
    n.map(sns.lineplot, 'Time', 'fivePerc')
    n.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    # fill in lines between quantiles
    for ax in n.axes.flat:
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[6].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[4].get_xdata(),ax.lines[4].get_ydata(), ax.lines[7].get_ydata(), color = '#e26952', alpha=0.2)
        ax.fill_between(ax.lines[5].get_xdata(),ax.lines[5].get_ydata(), ax.lines[8].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
    n.set(xticks=[0, 4, 16, 27])
    # add subplot titles
    axes = n.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("Exmoor ponies")
    axes[1].set_title("Fallow deer")
    axes[2].set_title("Grassland & parkland")
    axes[3].set_title("Longhorn cattle")
    axes[4].set_title("Organic carbon")
    axes[5].set_title("Red deer")
    axes[6].set_title("Roe deer")
    axes[7].set_title("Tamworth pigs")
    axes[8].set_title("Thorny scrubland")
    axes[9].set_title("Woodland")
    # add filter lines
    n.axes[2].vlines(x=4,ymin=0.9,ymax=1, color='r')
    n.axes[4].vlines(x=4,ymin=0.95,ymax=1.9, color='r')
    n.axes[6].vlines(x=4,ymin=1,ymax=3.3, color='r')
    n.axes[8].vlines(x=4,ymin=1,ymax=19, color='r')
    n.axes[9].vlines(x=4,ymin=0.85,ymax=1.56, color='r')
    # plot next set of filter lines
    n.axes[2].vlines(x=16,ymin=0.67,ymax=0.79, color='r')
    n.axes[4].vlines(x=16,ymin=1.7,ymax=2.2, color='r')
    n.axes[6].vlines(x=16,ymin=1.7,ymax=6.7, color='r')
    n.axes[8].vlines(x=16,ymin=22.5,ymax=35.1, color='r')
    n.axes[9].vlines(x=16,ymin=0.98,ymax=1.7, color='r')
    # make sure they all start from 0 
    n.axes.set(ylim =(0,None))
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Normal stocking density', 'Double stocking density', 'Half stocking density'],bbox_to_anchor=(2, 0), loc='lower right', fontsize=12)
    # ten digit random number
    plt.show()


plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
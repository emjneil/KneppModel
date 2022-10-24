# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# things to change depending on the parameter set:
# 1. parameter set
# 2. filtering conditions
# 3. min/max euro bison negative interactions
# 4. experiment 2 parameters and 10%
# 5. name of graphs / txt files

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
totalSimulations = 25000

# store species in a list
species = ['europeanBison','exmoorPony','fallowDeer','grasslandParkland','longhornCattle','organicCarbon','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']
# define the Lotka-Volterra equation
def ecoNetwork(t, X, A, r):
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))


# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():
    # define the array

    interaction_matrix = [
                # european bison - pos interactions through optimizer; neg interactions random higher than others 
                [-14.1, 0, 0, 13.7, 0, 0, 0, 0, 0, 0.29, 4.1], #parameter set 1
                # exmoor pony - special case, no growth
                [0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # fallow deer 
                [0, 0, -0.07, 0.57, 0, 0, 0, 0, 0, 0.0026, 0.2],
                # [0, 0, -15.9, 85.5, 0, 0, 0, 0, 0, 0.48, 41.3], #parameter set 2
                # grassland parkland
                [0, -0.00041, -0.00039, -0.78, -0.00065, -0.00056, 0, -0.00014, -0.0036, -0.01, -0.15],
                # [0, -0.00069, -0.0004, -0.78, -0.0008, -0.00049, 0, -0.00029, -0.0034, -0.01, -0.14], #parameter set 2
                # longhorn cattle  
                [0, 0, 0, 0.34, -0.31, 0, 0, 0, 0, 0.0067, 0.37],
                # [0, 0, 0, 19.4, -15.0, 0, 0, 0, 0, 0.15, 12.4], #parameter set 2
                # organic carbon
                [0, 0.0074, 0.0046, 0.05, 0.0069, -0.082, 0.0053, 0.004, 0.0055, 0.0012, 0.048],  
                # [0, 0.0075, 0.0048, 0.044, 0.0065, -0.083, 0.0053, 0.0041, 0.005, 0.0011, 0.044],  
                # red deer  
                [0, 0, 0, 0.34, 0, 0, -0.3, 0, 0, 0.016, 0.35],
                # [0, 0, 0, 15.6, 0, 0, -15.1, 0, 0, 0.82, 13.4],
                # roe deer 
                [0, 0, 0, 11.9, 0, 0, 0, -14.5, 0, 0.57, 11.25],
                # tamworth pig 
                [0, 0, 0, 4, 0, 0, 0, 0, -14.4, 0.34, 3.1],  
                # thorny scrub
                [0, -0.034, -0.031, 0, -0.044, 0, -0.031, -0.011, -0.028, -0.0072, -0.13],
                # [0, -0.038, -0.02, 0, -0.042, 0, -0.031, -0.012, -0.013, -0.0072, -0.13],
                # woodland
                [0, -0.0057, -0.0054, 0, -0.0084, 0, -0.0062, -0.0013, -0.0026, 0.00035, -0.009]
                # [0, -0.0046, -0.0021, 0, -0.0058, 0, -0.0051, -0.0014, -0.0019, 0.00035, -0.009]
    ]

    # generate random uniform numbers
    variation = np.random.uniform(low = 0.97, high=1.03, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix


def generateGrowth():
    # PARAMETER SET 1: leaning on the data (some consumers have incorrect negative diagonals)
    growthRates = [0, 0, 0, 0.98, 0, 0, 0, 0, 0, 0.78, 0.066] 
    # PARAMETER SET 2: leaning on the methods (some consumers have incorrect yearly data)
    # growthRates = [0, 0, 0, 0.97, 0, 0, 0, 0, 0, 0.77, 0.061]

    # multiply by a range
    variation = np.random.uniform(low = 0.97, high=1.03, size = (len(species),))
    growth = growthRates * variation
    return growth
    

def generateX0():
    # scale everything to abundance of one (except species to be reintroduced)
    X0 = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
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
#     nx.draw(networkVisual.subgraph('European bison'), pos=pos, font_size=6, node_size = 4000, node_color='green')
#     nx.draw(networkVisual.subgraph('Exmoor pony'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('Tamworth pig'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('Fallow deer'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('Longhorn cattle'), pos=pos, font_size=6, node_size = 4000, node_color='red')
#     nx.draw(networkVisual.subgraph('Red deer'), pos=pos, font_size=6, node_size = 4000, node_color='red')

#     plt.show()
# networkVisual()


# # # # --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -------


# check for stability
def calcJacobian(A, r, n):
    # make an empty array to fill (with diagonals = 1, zeros elsewhere since we want eigenalue)
    i_matrix = np.eye(len(n))
    # put n into an array to multiply by A
    n_array = np.matlib.repmat(n, 1, len(n))
    n_array = np.reshape (n_array, (11,11))
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
    t = np.linspace(2005, 2008.95, 8)
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
        # check viability of the parameter set (is it stable?); pinv used bc inv had singular matrix errors
        ia = np.linalg.pinv(A)
        # n is the equilibrium state; calc as inverse of -A*r
        n = -np.matmul(ia, r)
        isStable = calcStability(A, r, n)
        # if all the values of n are above zero at equilibrium, & if the parameter set is viable (stable & all n > 0 at equilibrium); do the calculation
        if np.all(n > 0) &  isStable == True:
            NUMBER_STABLE += 1
            # check ecological parameters (primary producers shouldn't go negative when on their own)
            all_ecoCheck = []
            for i in range(len(species)):
                X0_ecoCheck = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                X0_ecoCheck[i] = 1
                ecoCheck_ABC = solve_ivp(ecoNetwork, (0, 1), X0_ecoCheck,  t_eval = t_eco, args=(A, r), method = 'RK23') 
                all_ecoCheck = np.append(all_ecoCheck, ecoCheck_ABC.y)
            all_ecoCheck_results = (np.vstack(np.hsplit(all_ecoCheck.reshape(len(species), 22).transpose(),1)))
            all_ecoCheck_results = pd.DataFrame(data=all_ecoCheck_results, columns=species)
            # ecological reality check: primary producers should not decline with no herbivores present
            
            if (all_ecoCheck_results.loc[7,'grasslandParkland'] >= 1) & (all_ecoCheck_results.loc[19,'thornyScrub'] >= 1) & (all_ecoCheck_results.loc[21,'woodland'] >= 1):
                # remember the parameters used
                X0_growth = pd.concat([pd.DataFrame(X0), pd.DataFrame(r)], axis = 1)
                X0_growth.columns = ['X0','growth']
                parameters_used = pd.concat([X0_growth, pd.DataFrame(A, index = species, columns = species)])
                all_parameters.append(parameters_used)
                # run the ODE
                first_ABC = solve_ivp(ecoNetwork, (2005, 2008.95), X0,  t_eval = t, args=(A, r), method = 'RK23')
                # append all the runs
                all_runs = np.append(all_runs, first_ABC.y)
                all_times = np.append(all_times, first_ABC.t)
                # add one to the counter (so we know how many simulations were run in the end)
                NUMBER_OF_SIMULATIONS += 1

    # check the final runs
    print("number of stable simulations", NUMBER_STABLE)
    print("number of stable & ecologically sound simulations", NUMBER_OF_SIMULATIONS)
    
    with open("ps1_stable_numberStable.txt", "w") as text_file:
        print("number of stable simulations: {}".format(NUMBER_STABLE), file=text_file)
        print("number of stable & ecologically sound simulations: {}".format(NUMBER_OF_SIMULATIONS), file=text_file)

    # combine the final runs into one dataframe
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 8).transpose(),NUMBER_OF_SIMULATIONS)))
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
    final_runs['ID'] = np.repeat(IDs,8)
    # select only the last year
    accepted_year = final_runs.loc[final_runs['time'] == 2008.95]
    # with pd.option_context('display.max_columns',None):
    #     print(accepted_year)
    # filter the runs through conditions
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= 3.3) & (accepted_year['roeDeer'] >= 1) &
    (accepted_year['grasslandParkland'] <= 1) & (accepted_year['grasslandParkland'] >= 0.68) &
    (accepted_year['woodland'] <=1.6) & (accepted_year['woodland'] >= 0.85) &
    (accepted_year['thornyScrub'] <= 19) & (accepted_year['thornyScrub'] >= 1) &
    (accepted_year['organicCarbon'] <= 2.1) & (accepted_year['organicCarbon'] >= 1) 
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
        t = np.linspace(2009, 2009.95, 2)
        second_ABC = solve_ivp(ecoNetwork, (2009, 2009.95), X0_3,  t_eval = t, args=(A_2, r_2), method = 'RK23')
        # 2010: fallow deer reintroduced
        starting_values_2010 = second_ABC.y[0:11, 1:2].flatten()
        starting_values_2010[1] = 0.57
        starting_values_2010[2] = 1
        starting_values_2010[4] = 1.45
        starting_values_2010[8] = 0.85
        t_1 = np.linspace(2010, 2010.95, 2)
        third_ABC = solve_ivp(ecoNetwork, (2010, 2010.95), starting_values_2010,  t_eval = t_1, args=(A_2, r_2), method = 'RK23')
        # 2011
        starting_values_2011 = third_ABC.y[0:11, 1:2].flatten()
        starting_values_2011[1] = 0.65
        starting_values_2011[2] = 1.93
        starting_values_2011[4] = 1.74
        starting_values_2011[8] = 1.1
        t_2 = np.linspace(2011, 2011.95, 2)
        fourth_ABC = solve_ivp(ecoNetwork, (2011, 2011.95), starting_values_2011,  t_eval = t_2, args=(A_2, r_2), method = 'RK23')
        # 2012
        starting_2012 = fourth_ABC.y[0:11, 1:2].flatten()
        starting_values_2012 = starting_2012.copy()
        starting_values_2012[1] = 0.74
        starting_values_2012[2] = 2.38
        starting_values_2012[4] = 2.19
        starting_values_2012[8] = 1.65
        t_3 = np.linspace(2012, 2012.95, 2)
        fifth_ABC = solve_ivp(ecoNetwork, (2012, 2012.95), starting_values_2012,  t_eval = t_3, args=(A_2, r_2), method = 'RK23')
        # 2013: red deer reintroduced
        starting_2013 = fifth_ABC.y[0:11, 1:2].flatten()
        starting_values_2013 = starting_2013.copy()
        starting_values_2013[1] = 0.43
        starting_values_2013[2] = 2.38
        starting_values_2013[4] = 2.43
        starting_values_2013[6] = 1
        starting_values_2013[8] = 0.3
        t_4 = np.linspace(2013, 2013.95, 2)
        sixth_ABC = solve_ivp(ecoNetwork, (2013, 2013.95), starting_values_2013,  t_eval = t_4, args=(A_2, r_2), method = 'RK23')
        # 2014
        starting_2014 = sixth_ABC.y[0:11, 1:2].flatten()
        starting_values_2014 = starting_2014.copy()
        starting_values_2014[1] = 0.43
        starting_values_2014[2] = 2.38
        starting_values_2014[4] = 4.98
        starting_values_2014[6] = 1
        starting_values_2014[8] = 0.9
        t_5 = np.linspace(2014, 2014.95, 2)
        seventh_ABC = solve_ivp(ecoNetwork, (2014, 2014.95), starting_values_2014,  t_eval = t_5, args=(A_2, r_2), method = 'RK23')
        # 2015
        starting_values_2015 = seventh_ABC.y[0:11, 1:2].flatten()
        starting_values_2015[1] = 0.43
        starting_values_2015[2] = 2.38
        starting_values_2015[4] = 2.01
        starting_values_2015[6] = 1
        starting_values_2015[8] = 0.9
        t_2015 = np.linspace(2015, 2015.95, 2)
        ABC_2015 = solve_ivp(ecoNetwork, (2015, 2015.95), starting_values_2015,  t_eval = t_2015, args=(A_2, r_2), method = 'RK23')
        last_values_2015 = ABC_2015.y[0:11, 1:2].flatten()
        # 2016
        starting_values_2016 = last_values_2015.copy()
        starting_values_2016[1] = 0.48
        starting_values_2016[2] = 3.33
        starting_values_2016[4] = 1.62
        starting_values_2016[6] = 2
        starting_values_2016[8] = 0.4
        t_2016 = np.linspace(2016, 2016.95, 2)
        ABC_2016 = solve_ivp(ecoNetwork, (2016, 2016.95), starting_values_2016,  t_eval = t_2016, args=(A_2, r_2), method = 'RK23')
        last_values_2016 = ABC_2016.y[0:11, 1:2].flatten()
        # 2017
        starting_values_2017 = last_values_2016.copy()
        starting_values_2017[1] = 0.43
        starting_values_2017[2] = 3.93
        starting_values_2017[4] = 1.49
        starting_values_2017[6] = 1.08
        starting_values_2017[8] = 0.35
        t_2017 = np.linspace(2017, 2017.95, 2)
        ABC_2017 = solve_ivp(ecoNetwork, (2017, 2017.95), starting_values_2017,  t_eval = t_2017, args=(A_2, r_2), method = 'RK23')
        last_values_2017 = ABC_2017.y[0:11, 1:2].flatten()
        # 2018
        starting_values_2018 = last_values_2017.copy()
        starting_values_2018[1] = 0.39
        starting_values_2018[2] = 5.98
        starting_values_2018[4] = 1.66
        starting_values_2018[6] = 1.85
        starting_values_2018[8] = 0.8
        t_2018 = np.linspace(2018, 2018.95, 2)
        ABC_2018 = solve_ivp(ecoNetwork, (2018, 2018.95), starting_values_2018,  t_eval = t_2018, args=(A_2, r_2), method = 'RK23')
        last_values_2018 = ABC_2018.y[0:11, 1:2].flatten()
        # 2019
        starting_values_2019 = last_values_2018.copy()
        starting_values_2019[1] = 0
        starting_values_2019[2] = 6.62
        starting_values_2019[4] = 1.64
        starting_values_2019[6] = 2.85
        starting_values_2019[8] = 0.45
        t_2019 = np.linspace(2019, 2019.95, 2)
        ABC_2019 = solve_ivp(ecoNetwork, (2019, 2019.95), starting_values_2019,  t_eval = t_2019, args=(A_2, r_2), method = 'RK23')
        last_values_2019 = ABC_2019.y[0:11, 1:2].flatten()
        # 2020
        starting_values_2020 = last_values_2019.copy()
        starting_values_2020[1] = 0.65
        starting_values_2020[2] = 5.88
        starting_values_2020[4] = 1.53
        starting_values_2020[6] = 2.7
        starting_values_2020[8] = 0.35
        t_2020 = np.linspace(2020, 2020.95, 2)
        ABC_2020 = solve_ivp(ecoNetwork, (2020,2020.95), starting_values_2020,  t_eval = t_2020, args=(A_2, r_2), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
        combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
        # append to dataframe
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)   
        all_times_2 = np.append(all_times_2, combined_times)
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 24).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    final_runs_2['time'] = all_times_2
    # append all the parameters to a dataframe
    all_parameters_2 = pd.concat(all_parameters_2)
    # add ID to the dataframe & parameters
    all_parameters_2['ID'] = ([(x+1) for x in range(len(accepted_simulations)) for _ in range(len(parameters_used_2))])
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,24)
    return final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = runODE_2()
    # filter 2015 values : ponies = the same, fallow deer started at 3.33 next year but 0.88 were culled; longhorn got to maximum 2.45; red got to 2 next year; pigs got to maximum 1.1
    accepted_simulations_2015 = final_runs_2[(final_runs_2['time'] == 2015.95) & 
    (final_runs_2['exmoorPony'] <= 0.48) & (final_runs_2['exmoorPony'] >= 0.39) & 
    (final_runs_2['fallowDeer'] <= 4.81) & (final_runs_2['fallowDeer'] >= 3.6) &  # leaning on data
    # (final_runs_2['fallowDeer'] <= 8.02) & (final_runs_2['fallowDeer'] >= 3.6) &  # leaning on methods
    (final_runs_2['longhornCattle'] <= 2.7) & (final_runs_2['longhornCattle'] >= 2.37) & # leaning on data
    # (final_runs_2['longhornCattle'] <= 2.7) & (final_runs_2['longhornCattle'] >= 2.21) & # leaning on methods
    (final_runs_2['tamworthPig'] <= 1.45) & (final_runs_2['tamworthPig'] >= 0.85)]
    filtered_2015 = final_runs_2[final_runs_2['ID'].isin(accepted_simulations_2015['ID'])]
    print("number passed 2015 filters:", filtered_2015.shape[0]/24)
    
    # filter 2016 values : ponies = the same, fallow deer = 3.9 but 25 were culled; longhorn got to maximum 2.03; pig got to maximum 0.85
    accepted_simulations_2016 = filtered_2015[(filtered_2015['time'] == 2016.95) & 
    (filtered_2015['exmoorPony'] <= 0.52) & (filtered_2015['exmoorPony'] >= 0.43) & 
    (filtered_2015['fallowDeer'] <= 5.12) & (filtered_2015['fallowDeer'] >= 3.93) & # leaning on data
    # (filtered_2015['fallowDeer'] <= 8) & (filtered_2015['fallowDeer'] >= 3.93) & # leaning on methods
    (filtered_2015['longhornCattle'] <= 2.34) & (filtered_2015['longhornCattle'] >= 2.04) & 
    (filtered_2015['tamworthPig'] <= 1.25) & (filtered_2015['tamworthPig'] >= 0.65)]
    filtered_2016 = filtered_2015[filtered_2015['ID'].isin(accepted_simulations_2016['ID'])]
    print("number passed 2016 filters:", filtered_2016.shape[0]/24)
    # filter 2017 values : ponies = the same; fallow = 7.34 + 1.36 culled (this was maybe supplemented so no filter), same with red; cows got to max 2.06; red deer got to 1.85 + 2 culled; pig got to 1.1
    accepted_simulations_2017 = filtered_2016[(filtered_2016['time'] == 2017.95) & 
    (filtered_2016['exmoorPony'] <= 0.48) & (filtered_2016['exmoorPony'] >= 0.39) & 
    (filtered_2016['longhornCattle'] <= 2.3) & (filtered_2016['longhornCattle'] >= 1.96) & 
    (filtered_2016['redDeer'] <= 2.31) & (filtered_2016['redDeer'] >= 1.8) & # leaning on data
    # (filtered_2016['redDeer'] <= 3.31) & (filtered_2016['redDeer'] >= 1.8) & # leaning on data
    (filtered_2016['tamworthPig'] <= 1.75) & (filtered_2016['tamworthPig'] >= 1.15)]
    # (filtered_2016['tamworthPig'] <= 1.75) & (filtered_2016['tamworthPig'] >= 1)]
    filtered_2017 = filtered_2016[filtered_2016['ID'].isin(accepted_simulations_2017['ID'])]
    print("number passed 2017 filters:", filtered_2017.shape[0]/24)
    # filter 2018 values : p ponies = same, fallow = 6.62 + 57 culled; cows got to max 2.21; reds got to 2.85 + 3 culled; pigs got to max 1.15
    accepted_simulations_2018 = filtered_2017[(filtered_2017['time'] == 2018.95) & 
    (filtered_2017['exmoorPony'] <= 0.43) & (filtered_2017['exmoorPony'] >= 0.35) & 
    (filtered_2017['fallowDeer'] <= 8.57) & (filtered_2017['fallowDeer'] >= 7.38) & 
    (filtered_2017['longhornCattle'] <= 2.45) & (filtered_2017['longhornCattle'] >= 2.15) & 
    (filtered_2017['redDeer'] <= 3.39) & (filtered_2017['redDeer'] >= 2.77) & 
    (filtered_2017['tamworthPig'] <= 1.45) & (filtered_2017['tamworthPig'] >= 0.85)]
    filtered_2018 = filtered_2017[filtered_2017['ID'].isin(accepted_simulations_2018['ID'])]
    print("number passed 2018 filters:", filtered_2018.shape[0]/24)
    # filter 2019 values : ponies = 0, fallow = 6.62 + 1.36 culled; longhorn maximum 2
    accepted_simulations_2019 = filtered_2018[(filtered_2018['time'] == 2019.95) & 
    (filtered_2018['fallowDeer'] <= 8.14) & (filtered_2018['fallowDeer'] >= 6.95) & 
    (filtered_2018['longhornCattle'] <= 2.36) & (filtered_2018['longhornCattle'] >= 2.06) & 
    (filtered_2018['redDeer'] <= 3.69) & (filtered_2018['redDeer'] >= 3.08)]
    filtered_2019 = filtered_2018[filtered_2018['ID'].isin(accepted_simulations_2019['ID'])]
    print("number passed 2019 filters:", filtered_2019.shape[0]/24)
    # now choose just the final years (these will become the starting conditions in the next model)
    filtered_2020 = filtered_2019.loc[filtered_2019['time'] == 2020.95]
    # filter the final 2020 runs
    accepted_simulations_2020 = filtered_2020.loc[
    # 2020  - no filtering for fallow or red deer bc we don't know what they've grown to yet (next survey March 2021)
    (filtered_2020['exmoorPony'] <= 0.7) & (filtered_2020['exmoorPony'] >= 0.61) &
    (filtered_2020['roeDeer'] <= 3.3) & (filtered_2020['roeDeer'] >= 1.7) &
    (filtered_2020['grasslandParkland'] <= 0.86) & (filtered_2020['grasslandParkland'] >= 0.61) &
    (filtered_2020['woodland'] <= 1.7) & (filtered_2020['woodland'] >= 1) &
    (filtered_2020['thornyScrub'] <= 31.9) & (filtered_2020['thornyScrub'] >= 19) & 
    (filtered_2020['organicCarbon'] <= 2.2) & (filtered_2020['organicCarbon'] >= 1.7)
    ]
    print("number passed 2020 filters:", accepted_simulations_2020.shape)


    with open("ps1_stable_numberAcceptedSims.txt", "w") as text_file:
        print("number accepted, first ODE: {}".format(accepted_simulations.shape), file=text_file)
        print("number passed 2015 filters: {}".format(filtered_2015.shape[0]/24), file=text_file)
        print("number passed 2016 filters: {}".format(filtered_2016.shape[0]/24), file=text_file)
        print("number passed 2017 filters: {}".format(filtered_2017.shape[0]/24), file=text_file)
        print("number passed 2018 filters: {}".format(filtered_2018.shape[0]/24), file=text_file)
        print("number passed 2019 filters: {}".format(filtered_2019.shape[0]/24), file=text_file)
        print("number passed 2020 filters: {}".format(accepted_simulations_2020.shape), file=text_file)

  

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
    plt.savefig('hist_ps1_stableOnly.png')

    # CORRELATION MATRIX
    growthRates_3.columns = ['euroBison_growth','exmoorPony_growth', 'fallowDeer_growth', 'grasslandParkland_growth', 'longhornCattle_growth','organicCarbon_growth', 'redDeer_growth', 'roeDeer_growth', 'tamworthPig_growth', 'thornyScrub_growth', 'woodland_growth']
    # reshape int matrix
    euroBisonInts = interaction_strength_3[interaction_strength_3.index=='europeanBison']
    euroBisonInts.columns = ['bison_bison', 'bison_pony','bison_fallow','bison_grass','bison_cattle','bison_carbon','bison_red','bison_roe','bison_pig','bison_scrub','bison_wood']
    euroBisonInts = euroBisonInts.reset_index(drop=True)
    exmoorInts = interaction_strength_3[interaction_strength_3.index=='exmoorPony']
    exmoorInts.columns = ['pony_bison','pony_pony', 'pony_fallow','pony_grass','pony_cattle','pony_carbon','pony_red','pony_roe','pony_pig','pony_scrub','pony_wood']
    exmoorInts = exmoorInts.reset_index(drop=True)
    fallowInts = interaction_strength_3[interaction_strength_3.index=='fallowDeer']
    fallowInts.columns = ['fallow_bison', 'fallow_pony', 'fallow_fallow', 'fallow_grass','fallow_cattle','fallow_carbon','fallow_red', 'fallow_roe', 'fallow_pig', 'fallow_scrub', 'fallow_wood']
    fallowInts = fallowInts.reset_index(drop=True)
    arableInts = interaction_strength_3[interaction_strength_3.index=='grasslandParkland']
    arableInts.columns = ['grass_bison','grass_pony', 'grass_fallow', 'grass_grass','grass_cattle','grass_carbon','grass_red', 'grass_roe', 'grass_pig', 'grass_scrub', 'grass_wood']
    arableInts = arableInts.reset_index(drop=True)
    longhornInts = interaction_strength_3[interaction_strength_3.index=='longhornCattle']
    longhornInts.columns = ['cattle_bison','cattle_pony', 'cattle_fallow', 'cattle_grass','cattle_cattle','cattle_carbon','cattle_red', 'cattle_roe', 'cattle_pig', 'cattle_scrub', 'cattle_wood']
    longhornInts = longhornInts.reset_index(drop=True)
    orgCarbInts = interaction_strength_3[interaction_strength_3.index=='organicCarbon']
    orgCarbInts.columns = ['carbon_bison','carbon_pony', 'carbon_fallow', 'carbon_grass','carbon_cattle','carbon_carbon','carbon_red', 'carbon_roe', 'carbon_pig', 'carbon_scrub', 'carbon_wood']
    orgCarbInts = orgCarbInts.reset_index(drop=True)
    redDeerInts = interaction_strength_3[interaction_strength_3.index=='redDeer']
    redDeerInts.columns = ['red_bison','red_pony', 'red_fallow', 'red_grass','red_cattle','red_carbon','red_red', 'red_roe', 'red_pig', 'red_scrub', 'red_wood']
    redDeerInts = redDeerInts.reset_index(drop=True)
    roeDeerInts = interaction_strength_3[interaction_strength_3.index=='roeDeer']
    roeDeerInts.columns = ['roe_bison','roe_pony', 'roe_fallow', 'roe_grass','roe_cattle','roe_carbon','roe_red', 'roe_roe', 'roe_pig', 'roe_scrub', 'roe_wood']
    roeDeerInts = roeDeerInts.reset_index(drop=True)
    tamworthPigInts = interaction_strength_3[interaction_strength_3.index=='tamworthPig']
    tamworthPigInts.columns = ['pig_bison','pig_pony', 'pig_fallow', 'pig_grass','pig_cattle','pig_carbon','pig_red', 'pig_roe', 'pig_pig', 'pig_scrub', 'pig_wood']
    tamworthPigInts = tamworthPigInts.reset_index(drop=True)
    thornyScrubInts = interaction_strength_3[interaction_strength_3.index=='thornyScrub']
    thornyScrubInts.columns = ['scrub_bison','scrub_pony', 'scrub_fallow', 'scrub_grass','scrub_cattle','scrub_carbon','scrub_red', 'scrub_roe', 'scrub_pig', 'scrub_scrub', 'scrub_wood']
    thornyScrubInts = thornyScrubInts.reset_index(drop=True)
    woodlandInts = interaction_strength_3[interaction_strength_3.index=='woodland']
    woodlandInts.columns = ['wood_bison','wood_pony', 'wood_fallow', 'wood_grass','wood_cattle','wood_carbon','wood_red', 'wood_roe', 'wood_pig', 'wood_scrub', 'wood_wood']
    woodlandInts = woodlandInts.reset_index(drop=True)
    combined = pd.concat([growthRates_3, euroBisonInts, exmoorInts, fallowInts, arableInts, longhornInts, orgCarbInts, redDeerInts, roeDeerInts, tamworthPigInts, thornyScrubInts, woodlandInts], axis=1)
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
    # generate mask for upper triangle
    mask = np.triu(np.ones_like(signif_Matrix, dtype=bool))
    # plot it
    plt.subplots(figsize=(11,11))
    ax = sns.heatmap(
    signif_Matrix, 
    vmin=-1, vmax=1, center=0,
    mask = mask,
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
    plt.savefig('corrMatrix_ps1_stableOnly.png')
    # plt.show()
    return r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# # # # # ------ SOLVE ODE #3: Projecting forwards 10 years (2020-2030) -------

def runODE_3():
    r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = generateParameters3()
    # project into the future
    X0_thirdRun = X0_3.to_numpy()
    all_runs_3 = []
    all_parameters_3 = []
    all_times_3 = []
    t = np.linspace(2021, 2021.95, 2)
    # loop through each row of accepted parameters
    for X0_4, r_4, A_4 in zip(X0_thirdRun,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # starting values for 2021 - the stocking densities
        X0_4[1] =  0.65
        X0_4[2] =  5.88
        X0_4[4] =  1.53
        X0_4[6] =  2.69
        X0_4[8] =  0.95
        # concantenate the parameters
        X0_growth_3 = pd.concat([pd.DataFrame(X0_4), pd.DataFrame(r_4)], axis = 1)
        X0_growth_3.columns = ['X0','growth']
        parameters_used_3 = pd.concat([X0_growth_3, pd.DataFrame(A_4, index = species, columns = species)])
        # 2021
        ABC_2021 = solve_ivp(ecoNetwork, (2021, 2021.95), X0_4,  t_eval = t, args=(A_4, r_4), method = 'RK23')        
        # ten percent above/below 2021 values
        starting_2022 = ABC_2021.y[0:11, 1:2].flatten()
        starting_2022[1] = np.random.uniform(low=0.61,high=0.7)  
        starting_2022[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2022[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2022[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2022[8] =  np.random.uniform(low=0.86,high=1.0)
        t_1 = np.linspace(2022, 2022.95, 2)
        # 2022
        ABC_2022 = solve_ivp(ecoNetwork, (2022, 2022.95), starting_2022,  t_eval = t_1, args=(A_4, r_4), method = 'RK23')
        starting_2023 = ABC_2022.y[0:11, 1:2].flatten()
        starting_2023[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2023[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2023[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2023[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2023[8] =  np.random.uniform(low=0.86,high=1.0)
        t_2 = np.linspace(2023, 2023.95, 2)
        # 2023
        ABC_2023 = solve_ivp(ecoNetwork, (2023, 2023.95), starting_2023,  t_eval = t_2, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2024 = ABC_2023.y[0:11, 1:2].flatten()
        starting_2024[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2024[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2024[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2024[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2024[8] =  np.random.uniform(low=0.86,high=1.0)
        t_3 = np.linspace(2024, 2024.95, 2)
        # run the model for 2024
        ABC_2024 = solve_ivp(ecoNetwork, (2024, 2024.95), starting_2024,  t_eval = t_3, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2025 = ABC_2024.y[0:11, 1:2].flatten()
        starting_2025[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2025[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2025[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2025[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2025[8] =  np.random.uniform(low=0.86,high=1.0)
        t_4 = np.linspace(2025, 2025.95, 2)
        # run the model for 2025
        ABC_2025 = solve_ivp(ecoNetwork, (2025, 2025.95), starting_2025,  t_eval = t_4, args=(A_4, r_4), method = 'RK23')
        starting_2026 = ABC_2025.y[0:11, 1:2].flatten()
        starting_2026[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2026[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2026[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2026[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2026[8] =  np.random.uniform(low=0.86,high=1.0)
        t_5 = np.linspace(2026, 2026.95, 2)
        # 2026
        ABC_2026 = solve_ivp(ecoNetwork, (2026, 2026.95), starting_2026,  t_eval = t_5, args=(A_4, r_4), method = 'RK23')
        starting_2027 = ABC_2026.y[0:11, 1:2].flatten()
        starting_2027[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2027[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2027[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2027[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2027[8] =  np.random.uniform(low=0.86,high=1.0)
        t_6 = np.linspace(2027, 2027.95, 2)
        # 2027
        ABC_2027 = solve_ivp(ecoNetwork, (2027, 2027.95), starting_2027,  t_eval = t_6, args=(A_4, r_4), method = 'RK23')
        starting_2028 = ABC_2027.y[0:11, 1:2].flatten()
        starting_2028[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2028[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2028[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2028[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2028[8] =  np.random.uniform(low=0.86,high=1.0)
        t_7 = np.linspace(2028, 2028.95, 2)
        # 2028
        ABC_2028 = solve_ivp(ecoNetwork, (2028, 2028.95), starting_2028,  t_eval = t_7, args=(A_4, r_4), method = 'RK23')
        starting_2029 = ABC_2028.y[0:11, 1:2].flatten()
        starting_2029[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2029[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2029[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2029[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2029[8] =  np.random.uniform(low=0.86,high=1.0)
        t_8 = np.linspace(2029, 2029.95, 2)
        # 2029
        ABC_2029 = solve_ivp(ecoNetwork, (2029, 2029.95), starting_2028,  t_eval = t_8, args=(A_4, r_4), method = 'RK23')
        starting_2030 = ABC_2029.y[0:11, 1:2].flatten()
        starting_2030[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2030[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2030[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2030[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2030[8] =  np.random.uniform(low=0.86,high=1.0)
        t_9 = np.linspace(2030, 2030.95, 2)
        # 2030
        ABC_2030 = solve_ivp(ecoNetwork, (2030, 2030.95), starting_2030,  t_eval = t_9, args=(A_4, r_4), method = 'RK23')
        starting_2031 = ABC_2030.y[0:11, 1:2].flatten()
        starting_2031[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2031[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2031[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2031[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2031[8] =  np.random.uniform(low=0.86,high=1.0)
        t_10 = np.linspace(2031, 2031.95, 2)
        # 2031
        ABC_2031 = solve_ivp(ecoNetwork, (2031, 2031.95), starting_2031,  t_eval = t_10, args=(A_4, r_4), method = 'RK23')
        starting_2032 = ABC_2031.y[0:11, 1:2].flatten()
        starting_2032[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2032[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2032[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2032[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2032[8] =  np.random.uniform(low=0.86,high=1.0)
        t_11 = np.linspace(2032, 2032.95, 2)
        # 2032
        ABC_2032 = solve_ivp(ecoNetwork, (2032, 2032.95), starting_2032,  t_eval = t_11, args=(A_4, r_4), method = 'RK23')
        starting_2033 = ABC_2032.y[0:11, 1:2].flatten()
        starting_2033[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2033[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2033[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2033[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2033[8] =  np.random.uniform(low=0.86,high=1.0)
        t_12 = np.linspace(2033, 2033.95, 2)
        # 2033
        ABC_2033 = solve_ivp(ecoNetwork, (2033, 2033.95), starting_2033,  t_eval = t_12, args=(A_4, r_4), method = 'RK23')
        starting_2034 = ABC_2033.y[0:11, 1:2].flatten()
        starting_2034[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2034[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2034[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2034[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2034[8] =  np.random.uniform(low=0.86,high=1.0)
        t_13 = np.linspace(2034, 2034.95, 2)
        # 2034
        ABC_2034 = solve_ivp(ecoNetwork, (2034, 2034.95), starting_2034,  t_eval = t_13, args=(A_4, r_4), method = 'RK23')
        starting_2035 = ABC_2034.y[0:11, 1:2].flatten()
        starting_2035[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2035[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2035[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2035[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2035[8] =  np.random.uniform(low=0.86,high=1.0)
        t_14 = np.linspace(2035, 2035.95, 2)
        # 2035
        ABC_2035 = solve_ivp(ecoNetwork, (2035,2035.95), starting_2035,  t_eval = t_14, args=(A_4, r_4), method = 'RK23')
        starting_2036 = ABC_2035.y[0:11, 1:2].flatten()
        starting_2036[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2036[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2036[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2036[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2036[8] =  np.random.uniform(low=0.86,high=1.0)
        t_15 = np.linspace(2036, 2036.95, 2)
        # 2036
        ABC_2036 = solve_ivp(ecoNetwork, (2036,2036.95), starting_2036,  t_eval = t_15, args=(A_4, r_4), method = 'RK23')
        starting_2037 = ABC_2036.y[0:11, 1:2].flatten()
        starting_2037[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2037[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2037[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2037[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2037[8] =  np.random.uniform(low=0.86,high=1.0)
        t_16 = np.linspace(2037, 2037.95, 2)
        # 2037
        ABC_2037 = solve_ivp(ecoNetwork, (2037,2037.95), starting_2037,  t_eval = t_16, args=(A_4, r_4), method = 'RK23')
        starting_2038 = ABC_2037.y[0:11, 1:2].flatten()
        starting_2038[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2038[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2038[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2038[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2038[8] =  np.random.uniform(low=0.86,high=1.0)
        t_17 = np.linspace(2038, 2038.95, 2)
        # 2038 
        ABC_2038 = solve_ivp(ecoNetwork, (2038,2038.95), starting_2038,  t_eval = t_17, args=(A_4, r_4), method = 'RK23')
        starting_2039 = ABC_2038.y[0:11, 1:2].flatten()
        starting_2039[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2039[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2039[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2039[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2039[8] =  np.random.uniform(low=0.86,high=1.0)
        t_18 = np.linspace(2039, 2039.95, 2)
        # 2039
        ABC_2039 = solve_ivp(ecoNetwork, (2039,2039.95), starting_2039,  t_eval = t_18, args=(A_4, r_4), method = 'RK23')
        starting_2040 = ABC_2039.y[0:11, 1:2].flatten()
        starting_2040[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2040[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2040[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2040[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2040[8] =  np.random.uniform(low=0.86,high=1.0)
        t_19 = np.linspace(2040, 2040.95, 2)
        # 2040
        ABC_2040 = solve_ivp(ecoNetwork, (2040,2040.95), starting_2040,  t_eval = t_19, args=(A_4, r_4), method = 'RK23')
        starting_2041 = ABC_2040.y[0:11, 1:2].flatten()
        starting_2041[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2041[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2041[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2041[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2041[8] =  np.random.uniform(low=0.86,high=1.0)
        t_20 = np.linspace(2041, 2041.95, 2)
        # 2041
        ABC_2041 = solve_ivp(ecoNetwork, (2041,2041.95), starting_2041,  t_eval = t_20, args=(A_4, r_4), method = 'RK23')
        starting_2042 = ABC_2041.y[0:11, 1:2].flatten()
        starting_2042[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2042[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2042[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2042[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2042[8] =  np.random.uniform(low=0.86,high=1.0)
        t_21 = np.linspace(2042, 2042.95, 2)        
        # 2042
        ABC_2042 = solve_ivp(ecoNetwork, (2042,2042.95), starting_2042,  t_eval = t_21, args=(A_4, r_4), method = 'RK23')
        starting_2043 = ABC_2042.y[0:11, 1:2].flatten()
        starting_2043[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2043[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2043[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2043[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2043[8] =  np.random.uniform(low=0.86,high=1.0)
        t_22 = np.linspace(2043, 2043.95, 2)    
        # 2043
        ABC_2043 = solve_ivp(ecoNetwork, (2043,2043.95), starting_2043,  t_eval = t_22, args=(A_4, r_4), method = 'RK23')
        starting_2044 = ABC_2043.y[0:11, 1:2].flatten()
        starting_2044[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2044[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2044[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2044[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2044[8] =  np.random.uniform(low=0.86,high=1.0)
        t_23 = np.linspace(2044, 2044.95, 2)   
        # 2044
        ABC_2044 = solve_ivp(ecoNetwork, (2044,2044.95), starting_2044,  t_eval = t_23, args=(A_4, r_4), method = 'RK23')
        starting_2045 = ABC_2044.y[0:11, 1:2].flatten()
        starting_2045[1] = np.random.uniform(low=0.61,high=0.7)
        starting_2045[2] =  np.random.uniform(low=5.3,high=6.5)
        starting_2045[4] =  np.random.uniform(low=1.38,high=1.7)
        starting_2045[6] =  np.random.uniform(low=2.42,high=3.0)
        starting_2045[8] =  np.random.uniform(low=0.86,high=1.0)
        t_24 = np.linspace(2045, 2045.95, 2)  
        # 2045
        ABC_2045 = solve_ivp(ecoNetwork, (2045,2046), starting_2045,  t_eval = t_24, args=(A_4, r_4), method = 'RK23')
        # concatenate & append all the runs
        combined_runs_2 = np.hstack((ABC_2021.y, ABC_2022.y, ABC_2023.y, ABC_2024.y, ABC_2025.y, ABC_2026.y, ABC_2027.y, ABC_2028.y, ABC_2029.y, ABC_2030.y, ABC_2031.y, ABC_2032.y, ABC_2033.y, ABC_2034.y, ABC_2035.y, ABC_2036.y, ABC_2037.y, ABC_2038.y, ABC_2039.y, ABC_2040.y, ABC_2041.y, ABC_2042.y, ABC_2043.y, ABC_2044.y, ABC_2045.y))
        combined_times_2 = np.hstack((ABC_2021.t, ABC_2022.t, ABC_2023.t, ABC_2024.t, ABC_2025.t, ABC_2026.t, ABC_2027.t, ABC_2028.t, ABC_2029.t, ABC_2030.t, ABC_2031.t, ABC_2032.t, ABC_2033.t, ABC_2034.t, ABC_2035.t, ABC_2036.t, ABC_2037.t, ABC_2038.t, ABC_2039.t, ABC_2040.t, ABC_2041.t, ABC_2042.t, ABC_2043.t, ABC_2044.t, ABC_2045.t))
        all_runs_3 = np.append(all_runs_3, combined_runs_2)
        # append all the parameters
        all_parameters_3.append(parameters_used_3)   
        # append the times
        all_times_3 = np.append(all_times_3, combined_times_2)
    # check the final runs
    final_runs_3 = (np.vstack(np.hsplit(all_runs_3.reshape(len(species)*len(accepted_simulations_2020), 50).transpose(),len(accepted_simulations_2020))))
    final_runs_3 = pd.DataFrame(data=final_runs_3, columns=species)
    # append all the parameters to a dataframe
    all_parameters_3 = pd.concat(all_parameters_3)
    # add ID to the dataframe & parameters 
    all_parameters_3['ID'] = ([(x+1) for x in range(len(accepted_simulations_2020)) for _ in range(len(parameters_used_3))])
    IDs = np.arange(1,1 + len(accepted_simulations_2020))
    final_runs_3['ID'] = np.repeat(IDs,50)
    final_runs_3['time'] = all_times_3
    final_runs_3['accepted?'] = np.repeat('Accepted', len(final_runs_3))



   # REALITY CHECK 1: What if there was no culling? Go one large herbivore at a time for: fallow deer, longhorn cattle, red deer, tamworth pig
    # no exmoor pony bc they're a special case (no growth/no pos interactions)
    # all_runs_noCulls = []
    # all_times_noCulls = []
    # X0_noCull = X0_secondRun.copy()
    # # loop through each row of accepted parameters
    # for X0_noCulling, r_7, A_7 in zip(X0_noCull,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
    #     # 2009
    #     t_noCulls = np.linspace(2009, 2009.95, 2)
    #     second_ABC = solve_ivp(ecoNetwork, (2009,2009.95), X0_noCulling,  t_eval = t_noCulls, args=(A_7, r_7), method = 'RK23')
    #     # 2010: fallow deer reintroduced
    #     starting_values_2010 = second_ABC.y[0:11, 1:2].flatten()
    #     starting_values_2010[1] = 0.57
    #     starting_values_2010[2] = 1
    #     starting_values_2010[4] = 1.45
    #     # starting_values_2010[8] = 0.85
    #     t_1 = np.linspace(2010, 2010.95, 2)
    #     third_ABC = solve_ivp(ecoNetwork, (2010,2010.95), starting_values_2010,  t_eval = t_1, args=(A_7, r_7), method = 'RK23')
    #     # 2011
    #     starting_values_2011 = third_ABC.y[0:11, 1:2].flatten()
    #     starting_values_2011[1] = 0.65
    #     starting_values_2011[2] = 1.93
    #     starting_values_2011[4] = 1.74
    #     # starting_values_2011[8] = 1.1
    #     t_2 = np.linspace(2011, 2011.95, 2)
    #     fourth_ABC = solve_ivp(ecoNetwork, (2011,2011.95), starting_values_2011,  t_eval = t_2, args=(A_7, r_7), method = 'RK23')
    #     # 2012
    #     starting_2012 = fourth_ABC.y[0:11, 1:2].flatten()
    #     starting_values_2012 = starting_2012.copy()
    #     starting_values_2012[1] = 0.74
    #     starting_values_2012[2] = 2.38
    #     starting_values_2012[4] = 2.19
    #     # starting_values_2012[8] = 1.65
    #     t_3 = np.linspace(2012, 2012.95, 2)
    #     fifth_ABC = solve_ivp(ecoNetwork, (2012,2012.95), starting_values_2012,  t_eval = t_3, args=(A_7, r_7), method = 'RK23')
    #     # 2013: red deer reintroduced
    #     starting_2013 = fifth_ABC.y[0:11, 1:2].flatten()
    #     starting_values_2013 = starting_2013.copy()
    #     starting_values_2013[1] = 0.43
    #     starting_values_2013[2] = 2.38
    #     starting_values_2013[4] = 2.43
    #     starting_values_2013[6] = 1
    #     # starting_values_2013[8] = 0.3
    #     t_4 = np.linspace(2013, 2013.95, 2)
    #     sixth_ABC = solve_ivp(ecoNetwork, (2013,2013.95), starting_values_2013,  t_eval = t_4, args=(A_7, r_7), method = 'RK23')
    #     # 2014
    #     starting_2014 = sixth_ABC.y[0:11, 1:2].flatten()
    #     starting_values_2014 = starting_2014.copy()
    #     starting_values_2014[1] = 0.43
    #     starting_values_2014[2] = 2.38
    #     starting_values_2014[4] = 4.98
    #     starting_values_2014[6] = 1
    #     # starting_values_2014[8] = 0.9
    #     t_5 = np.linspace(2014, 2014.95, 2)
    #     seventh_ABC = solve_ivp(ecoNetwork, (2014,2014.95), starting_values_2014,  t_eval = t_5, args=(A_7, r_7), method = 'RK23')
    #     # 2015
    #     starting_values_2015 = seventh_ABC.y[0:11, 1:2].flatten()
    #     starting_values_2015[1] = 0.43
    #     starting_values_2015[2] = 2.38
    #     starting_values_2015[4] = 2.01
    #     starting_values_2015[6] = 1
    #     # starting_values_2015[8] = 0.9
    #     t_2015 = np.linspace(2015, 2015.95, 2)
    #     ABC_2015 = solve_ivp(ecoNetwork, (2015,2015.95), starting_values_2015,  t_eval = t_2015, args=(A_7, r_7), method = 'RK23')
    #     last_values_2015 = ABC_2015.y[0:11, 1:2].flatten()
    #     # 2016
    #     starting_values_2016 = last_values_2015.copy()
    #     starting_values_2016[1] = 0.48
    #     starting_values_2016[2] = 3.33
    #     starting_values_2016[4] = 1.62
    #     starting_values_2016[6] = 2
    #     # starting_values_2016[8] = 0.4
    #     t_2016 = np.linspace(2016, 2016.95, 2)
    #     ABC_2016 = solve_ivp(ecoNetwork, (2016,2016.95), starting_values_2016,  t_eval = t_2016, args=(A_7, r_7), method = 'RK23')
    #     last_values_2016 = ABC_2016.y[0:11, 1:2].flatten()
    #     # 2017
    #     starting_values_2017 = last_values_2016.copy()
    #     starting_values_2017[1] = 0.43
    #     starting_values_2017[2] = 3.93
    #     starting_values_2017[4] = 1.49
    #     starting_values_2017[6] = 1.08
    #     # starting_values_2017[8] = 0.35
    #     t_2017 = np.linspace(2017, 2017.95, 2)
    #     ABC_2017 = solve_ivp(ecoNetwork, (2017,2017.95), starting_values_2017,  t_eval = t_2017, args=(A_7, r_7), method = 'RK23')
    #     last_values_2017 = ABC_2017.y[0:11, 1:2].flatten()
    #     # 2018
    #     starting_values_2018 = last_values_2017.copy()
    #     starting_values_2018[1] = 0.39
    #     starting_values_2018[2] = 5.98
    #     starting_values_2018[4] = 1.66
    #     starting_values_2018[6] = 1.85
    #     # starting_values_2018[8] = 0.8
    #     t_2018 = np.linspace(2018, 2018.95, 2)
    #     ABC_2018 = solve_ivp(ecoNetwork, (2018,2018.95), starting_values_2018,  t_eval = t_2018, args=(A_7, r_7), method = 'RK23')
    #     last_values_2018 = ABC_2018.y[0:11, 1:2].flatten()
    #     # 2019
    #     starting_values_2019 = last_values_2018.copy()
    #     starting_values_2019[1] = 0
    #     starting_values_2019[2] = 6.62
    #     starting_values_2019[4] = 1.64
    #     starting_values_2019[6] = 2.85
    #     # starting_values_2019[8] = 0.45
    #     t_2019 = np.linspace(2019, 2019.95, 2)
    #     ABC_2019 = solve_ivp(ecoNetwork, (2019,2019.95), starting_values_2019,  t_eval = t_2019, args=(A_7, r_7), method = 'RK23')
    #     last_values_2019 = ABC_2019.y[0:11, 1:2].flatten()
    #     # 2020
    #     starting_values_2020 = last_values_2019.copy()
    #     starting_values_2020[1] = 0.65
    #     starting_values_2020[2] = 5.88
    #     starting_values_2020[4] = 1.53
    #     starting_values_2020[6] = 2.7
    #     # starting_values_2020[8] = 0.35
    #     t_2020 = np.linspace(2020, 2021, 2)
    #     ABC_2020 = solve_ivp(ecoNetwork, (2020,2021), starting_values_2020,  t_eval = t_2020, args=(A_7, r_7), method = 'RK23')
    #     # concatenate & append all the runs
    #     combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
    #     combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
    #     # append to dataframe
    #     all_runs_noCulls = np.append(all_runs_noCulls, combined_runs)
    #     # append all the parameters
    #     all_times_noCulls = np.append(all_times_noCulls, combined_times)
    # no_Cull = (np.vstack(np.hsplit(all_runs_noCulls.reshape(len(species)*len(accepted_simulations_2020), 24).transpose(),len(accepted_simulations_2020))))
    # no_Cull = pd.DataFrame(data=no_Cull, columns=species)
    # IDs_3 = np.arange(1,1 + len(accepted_simulations_2020))
    # no_Cull['ID'] = np.repeat(IDs_3,24)
    # no_Cull['time'] = all_times_noCulls
    filtered_FinalRuns = final_runs.loc[(final_runs['accepted?'] == "Accepted") ]
    # no_Cull = pd.concat([filtered_FinalRuns, no_Cull])
    # no_Cull['accepted?'] = "noCulls"
    # # plot reality check #1
    # grouping_exp1 = np.repeat(no_Cull['ID'], len(species))
    # # extract the node values from all dataframes
    # finalRuns_noCull = no_Cull.drop(['ID', 'time','accepted?'], axis=1).values.flatten()
    # # we want species column to be spec1,spec2,spec3,spec4, etc.
    # species_realityCheck1 = np.tile(species, (24*len(accepted_simulations_2020)) + (8*len(accepted_simulations)))
    # # time 
    # firstODEyears_exp1 = np.repeat(no_Cull['time'],len(species))
    # # put it in a dataframe
    # final_df_exp1 = pd.DataFrame(
    #     {'Abundance %': finalRuns_noCull, 'runNumber': grouping_exp1, 'Ecosystem Element': species_realityCheck1, 'Time': firstODEyears_exp1})
    # # calculate median 
    # exp1 = final_df_exp1.groupby(['Time', 'Ecosystem Element'])[['Abundance %']].apply(np.median)
    # exp1.name = 'Median'
    # final_df_exp1 = final_df_exp1.join(exp1, on=['Time','Ecosystem Element'])
    # # calculate quantiles
    # perc_exp1_1 = final_df_exp1.groupby(['Time','Ecosystem Element'])['Abundance %'].quantile(.95)
    # perc_exp1_1.name = 'ninetyfivePerc'
    # final_df_exp1 = final_df_exp1.join(perc_exp1_1, on=['Time','Ecosystem Element'])
    # perc_exp1 = final_df_exp1.groupby(['Time', 'Ecosystem Element'])['Abundance %'].quantile(.05)
    # perc_exp1.name = "fivePerc"
    # final_df_exp1 = final_df_exp1.join(perc_exp1, on=['Time','Ecosystem Element'])
    # # # graph it
    # f = sns.FacetGrid(final_df_exp1, col="Ecosystem Element", col_wrap=4, sharey = False)
    # f.map(sns.lineplot, 'Time', 'Median')
    # f.map(sns.lineplot, 'Time', 'fivePerc')
    # f.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    # for ax in f.axes.flat:
    #     ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha =0.2)
    # plt.tight_layout()
    # plt.show()




    # # REALITY CHECK 2: do herbivores decline quickly with no habitat? do woodland & scrub increase with no herbivory? etc.
    # all_runs_realityCheck = []
    # all_times_realityCheck = []
    # t_realityCheck = np.linspace(2009, 2100, 20)
    # # change X0 depending on what's needed for the reality check
    # X0_5 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # for r_realityCheck2, A_realityCheck2 in zip(r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
    #     # A_realityCheck2 = pd.DataFrame(data = A_realityCheck2, index = species, columns = species)
    #     # A_realityCheck2['europeanBison']['europeanBison'] = -0.001
    #     # A_realityCheck2['fallowDeer']['fallowDeer'] = -0.001
    #     # A_realityCheck2['exmoorPony']['exmoorPony'] = -0.001
    #     # A_realityCheck2['longhornCattle']['longhornCattle'] = -0.001
    #     # A_realityCheck2['redDeer']['redDeer'] = -0.001
    #     # A_realityCheck2['tamworthPig']['tamworthPig'] = -0.001
    #     # A_realityCheck2['roeDeer']['roeDeer'] = -0.001
    #     # A_realityCheck2 = A_realityCheck2.to_numpy()
    #     realityCheck_ABC = solve_ivp(ecoNetwork, (2009, 2100), X0_5,  t_eval = t_realityCheck, args=(A_realityCheck2, r_realityCheck2), method = 'RK23') 
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
    t_noReintro = np.linspace(2009, 2046, 50)
    for X0_noReintro, r_noReintro, A_noReintro in zip(X0_3_noReintro,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        noReintro_ABC = solve_ivp(ecoNetwork, (2009, 2046), X0_noReintro,  t_eval = t_noReintro, args=(A_noReintro, r_noReintro), method = 'RK23') 
        all_runs_2 = np.append(all_runs_2, noReintro_ABC.y)
        all_times_2 = np.append(all_times_2, noReintro_ABC.t)
    no_reintro = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations_2020), 50).transpose(),len(accepted_simulations_2020))))
    no_reintro = pd.DataFrame(data=no_reintro, columns=species)
    IDs_2 = np.arange(1,1 + len(accepted_simulations_2020))
    no_reintro['ID'] = np.repeat(IDs_2,50)
    no_reintro['time'] = all_times_2
    # concantenate this will the accepted runs from years 1-5
    no_reintro = pd.concat([filtered_FinalRuns, no_reintro])
    no_reintro['accepted?'] = "noReintro"



    # EXPERIMENT 2: What is the range of parameters needed for grass to collapse?
    all_runs_experiment2 = []
    all_times_experiment2 = []
    combined_stockingDensity_ponies = []
    combined_stockingDensity_fallowDeer = []
    combined_stockingDensity_longhornCattle = []
    combined_stockingDensity_redDeer = []
    combined_stockingDensity_pigs = []
    X0_experiment2 = X0_thirdRun.copy()

    # loop through each row of accepted parameters
    for X0_exp2, r_9, A_9 in zip(X0_experiment2,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # take 10% above and below the optimizer outputs - parameter 1
        stocking_exmoorPony =  np.random.uniform(low=0.068, high=0.084)
        stocking_fallowDeer = np.random.uniform(low=0.014, high=0.017)
        stocking_longhornCattle = np.random.uniform(low=0.1, high=0.12)
        stocking_redDeer = np.random.uniform(low=0.015, high=0.019)
        stocking_tamworthPig = np.random.uniform(low=0.86, high=1.1)

        # parameter 2
        # stocking_exmoorPony =  0
        # stocking_fallowDeer = np.random.uniform(low=0.13, high=0.15)
        # stocking_longhornCattle = 0
        # stocking_redDeer = 0
        # stocking_tamworthPig = np.random.uniform(low=0.83, high=1)

        # 2021 - future projections
        X0_exp2[1] =  0.65 * stocking_exmoorPony
        X0_exp2[2] =  5.88*stocking_fallowDeer
        X0_exp2[4] =  1.53*stocking_longhornCattle
        X0_exp2[6] =  2.69*stocking_redDeer
        X0_exp2[8] =  0.95*stocking_tamworthPig
        t1_experiment2 = np.linspace(2021, 2021.95, 2)
        ABC_stockingRate_2021 = solve_ivp(ecoNetwork, (2021,2021.95), X0_exp2,  t_eval = t1_experiment2, args=(A_9, r_9), method = 'RK23')
        stocking_values_2022 = ABC_stockingRate_2021.y[0:11, 1:2].flatten()
        # 2022
        stocking_values_2022[1] =  0.65*stocking_exmoorPony
        stocking_values_2022[2] =  5.88*stocking_fallowDeer
        stocking_values_2022[4] =  1.53*stocking_longhornCattle
        stocking_values_2022[6] =  2.69*stocking_redDeer
        stocking_values_2022[8] =  0.95*stocking_tamworthPig
        t13_stockingRate = np.linspace(2022, 2022.95, 2)
        ABC_stockingRate_2022 = solve_ivp(ecoNetwork, (2022,2022.95), stocking_values_2022,  t_eval = t13_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2023 = ABC_stockingRate_2022.y[0:11, 1:2].flatten()
        # 2023
        stocking_values_2023[1] =  0.65 * stocking_exmoorPony
        stocking_values_2023[2] =  5.88*stocking_fallowDeer
        stocking_values_2023[4] =  1.53*stocking_longhornCattle
        stocking_values_2023[6] =  2.69*stocking_redDeer
        stocking_values_2023[8] =  0.95*stocking_tamworthPig
        t14_stockingRate = np.linspace(2023, 2023.95, 2)
        ABC_stockingRate_2023 = solve_ivp(ecoNetwork, (2023,2023.95), stocking_values_2023,  t_eval = t14_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2024 = ABC_stockingRate_2023.y[0:11, 1:2].flatten()
        # 2024
        stocking_values_2024[1] =  0.65 * stocking_exmoorPony
        stocking_values_2024[2] =  5.88*stocking_fallowDeer
        stocking_values_2024[4] =  1.53*stocking_longhornCattle
        stocking_values_2024[6] =  2.69*stocking_redDeer
        stocking_values_2024[8] =  0.95*stocking_tamworthPig
        t15_stockingRate = np.linspace(2024, 2024.95, 2)
        ABC_stockingRate_2024 = solve_ivp(ecoNetwork, (2024,2024.95), stocking_values_2024,  t_eval = t15_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2025 = ABC_stockingRate_2024.y[0:11, 1:2].flatten()
        # 2025
        stocking_values_2025[1] =  0.65 * stocking_exmoorPony
        stocking_values_2025[2] =  5.88*stocking_fallowDeer
        stocking_values_2025[4] =  1.53*stocking_longhornCattle
        stocking_values_2025[6] =  2.69*stocking_redDeer
        stocking_values_2025[8] =  0.95*stocking_tamworthPig
        t16_stockingRate = np.linspace(2025, 2025.95, 2)
        ABC_stockingRate_2025 = solve_ivp(ecoNetwork, (2025,2025.95), stocking_values_2025,  t_eval = t16_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2026 = ABC_stockingRate_2025.y[0:11, 1:2].flatten()
        # 2026
        stocking_values_2026[1] =  0.65 * stocking_exmoorPony
        stocking_values_2026[2] =  5.88*stocking_fallowDeer
        stocking_values_2026[4] =  1.53*stocking_longhornCattle
        stocking_values_2026[6] =  2.69*stocking_redDeer
        stocking_values_2026[8] =  0.95*stocking_tamworthPig
        t17_stockingRate = np.linspace(2026, 2026.95, 2)
        ABC_stockingRate_2026 = solve_ivp(ecoNetwork, (2026,2026.95), stocking_values_2026,  t_eval = t17_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2027 = ABC_stockingRate_2026.y[0:11, 1:2].flatten()
        # 2027
        stocking_values_2027[1] =  0.65 * stocking_exmoorPony
        stocking_values_2027[2] =  5.88*stocking_fallowDeer
        stocking_values_2027[4] =  1.53*stocking_longhornCattle
        stocking_values_2027[6] =  2.69*stocking_redDeer
        stocking_values_2027[8] =  0.95*stocking_tamworthPig
        t18_stockingRate = np.linspace(2027, 2027.95, 2)
        ABC_stockingRate_2027 = solve_ivp(ecoNetwork, (2027,2027.95), stocking_values_2027,  t_eval = t18_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2028 = ABC_stockingRate_2027.y[0:11, 1:2].flatten()
        # 2028
        stocking_values_2028[1] =  0.65 * stocking_exmoorPony
        stocking_values_2028[2] =  5.88*stocking_fallowDeer
        stocking_values_2028[4] =  1.53*stocking_longhornCattle
        stocking_values_2028[6] =  2.69*stocking_redDeer
        stocking_values_2028[8] =  0.95*stocking_tamworthPig
        t19_stockingRate = np.linspace(2028, 2028.95, 2)
        ABC_stockingRate_2028 = solve_ivp(ecoNetwork, (2028,2028.95), stocking_values_2028,  t_eval = t19_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2029 = ABC_stockingRate_2028.y[0:11, 1:2].flatten()
        # 2029
        stocking_values_2029[1] =  0.65 * stocking_exmoorPony
        stocking_values_2029[2] =  5.88*stocking_fallowDeer
        stocking_values_2029[4] =  1.53*stocking_longhornCattle
        stocking_values_2029[6] =  2.69*stocking_redDeer
        stocking_values_2029[8] =  0.95*stocking_tamworthPig
        t20_stockingRate = np.linspace(2029, 2029.95, 2)
        ABC_stockingRate_2029 = solve_ivp(ecoNetwork, (2029,2029.95), stocking_values_2029,  t_eval = t20_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2030 = ABC_stockingRate_2029.y[0:11, 1:2].flatten()
        # 2030
        stocking_values_2030[1] =  0.65 * stocking_exmoorPony
        stocking_values_2030[2] =  5.88*stocking_fallowDeer
        stocking_values_2030[4] =  1.53*stocking_longhornCattle
        stocking_values_2030[6] =  2.69*stocking_redDeer
        stocking_values_2030[8] =  0.95*stocking_tamworthPig
        t21_stockingRate = np.linspace(2030, 2030.95, 2)
        ABC_stockingRate_2030 = solve_ivp(ecoNetwork, (2030, 2030.95), stocking_values_2030,  t_eval = t21_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2031 = ABC_stockingRate_2030.y[0:11, 1:2].flatten()
        # 2031
        stocking_values_2031[1] =  0.65 * stocking_exmoorPony
        stocking_values_2031[2] =  5.88*stocking_fallowDeer
        stocking_values_2031[4] =  1.53*stocking_longhornCattle
        stocking_values_2031[6] =  2.69*stocking_redDeer
        stocking_values_2031[8] =  0.95*stocking_tamworthPig
        t22_stockingRate = np.linspace(2031, 2031.95, 2)
        ABC_stockingRate_2031 = solve_ivp(ecoNetwork, (2031, 2031.95), stocking_values_2031,  t_eval = t22_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2032 = ABC_stockingRate_2031.y[0:11, 1:2].flatten()
        # 2032
        stocking_values_2032[1] =  0.65 * stocking_exmoorPony
        stocking_values_2032[2] =  5.88*stocking_fallowDeer
        stocking_values_2032[4] =  1.53*stocking_longhornCattle
        stocking_values_2032[6] =  2.69*stocking_redDeer
        stocking_values_2032[8] =  0.95*stocking_tamworthPig
        t23_stockingRate = np.linspace(2032, 2032.95, 2)
        ABC_stockingRate_2032 = solve_ivp(ecoNetwork, (2032, 2032.95), stocking_values_2032,  t_eval = t23_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2033 = ABC_stockingRate_2032.y[0:11, 1:2].flatten()
        # 2033
        stocking_values_2033[1] =  0.65 * stocking_exmoorPony
        stocking_values_2033[2] =  5.88*stocking_fallowDeer
        stocking_values_2033[4] =  1.53*stocking_longhornCattle
        stocking_values_2033[6] =  2.69*stocking_redDeer
        stocking_values_2033[8] =  0.95*stocking_tamworthPig
        t24_stockingRate = np.linspace(2033, 2033.95, 2)
        ABC_stockingRate_2033 = solve_ivp(ecoNetwork, (2033, 2033.95), stocking_values_2033,  t_eval = t24_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2034 = ABC_stockingRate_2033.y[0:11, 1:2].flatten()
        # 2034
        stocking_values_2034[1] =  0.65 * stocking_exmoorPony
        stocking_values_2034[2] =  5.88*stocking_fallowDeer
        stocking_values_2034[4] =  1.53*stocking_longhornCattle
        stocking_values_2034[6] =  2.69*stocking_redDeer
        stocking_values_2034[8] =  0.95*stocking_tamworthPig
        t25_stockingRate = np.linspace(2034, 2034.95, 2)
        ABC_stockingRate_2034 = solve_ivp(ecoNetwork, (2034, 2034.95), stocking_values_2034,  t_eval = t25_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2035 = ABC_stockingRate_2034.y[0:11, 1:2].flatten()
        # 2035
        stocking_values_2035[1] =  0.65 * stocking_exmoorPony
        stocking_values_2035[2] =  5.88*stocking_fallowDeer
        stocking_values_2035[4] =  1.53*stocking_longhornCattle
        stocking_values_2035[6] =  2.69*stocking_redDeer
        stocking_values_2035[8] =  0.95*stocking_tamworthPig
        t26_stockingRate = np.linspace(2035, 2035.95, 2)
        ABC_stockingRate_2035 = solve_ivp(ecoNetwork, (2035, 2035.95), stocking_values_2035,  t_eval = t26_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2036 = ABC_stockingRate_2035.y[0:11, 1:2].flatten()
        # 2036
        stocking_values_2036[1] =  0.65 * stocking_exmoorPony
        stocking_values_2036[2] =  5.88*stocking_fallowDeer
        stocking_values_2036[4] =  1.53*stocking_longhornCattle
        stocking_values_2036[6] =  2.69*stocking_redDeer
        stocking_values_2036[8] =  0.95*stocking_tamworthPig
        t27_stockingRate = np.linspace(2036, 2036.95, 2)
        ABC_stockingRate_2036 = solve_ivp(ecoNetwork, (2036, 2036.95), stocking_values_2036,  t_eval = t27_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2037 = ABC_stockingRate_2036.y[0:11, 1:2].flatten()
        # 2037
        stocking_values_2037[1] =  0.65 * stocking_exmoorPony
        stocking_values_2037[2] =  5.88*stocking_fallowDeer
        stocking_values_2037[4] =  1.53*stocking_longhornCattle
        stocking_values_2037[6] =  2.69*stocking_redDeer
        stocking_values_2037[8] =  0.95*stocking_tamworthPig
        t28_stockingRate = np.linspace(2037, 2037.95, 2)
        ABC_stockingRate_2037 = solve_ivp(ecoNetwork, (2037, 2037.95), stocking_values_2037,  t_eval = t28_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2038 = ABC_stockingRate_2037.y[0:11, 1:2].flatten()
        # 2038
        stocking_values_2038[1] =  0.65 * stocking_exmoorPony
        stocking_values_2038[2] =  5.88*stocking_fallowDeer
        stocking_values_2038[4] =  1.53*stocking_longhornCattle
        stocking_values_2038[6] =  2.69*stocking_redDeer
        stocking_values_2038[8] =  0.95*stocking_tamworthPig
        t29_stockingRate = np.linspace(2038, 2038.95, 2)
        ABC_stockingRate_2038 = solve_ivp(ecoNetwork, (2038, 2038.95), stocking_values_2038,  t_eval = t29_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2039 = ABC_stockingRate_2038.y[0:11, 1:2].flatten()
        # 2039
        stocking_values_2039[1] =  0.65 * stocking_exmoorPony
        stocking_values_2039[2] =  5.88*stocking_fallowDeer
        stocking_values_2039[4] =  1.53*stocking_longhornCattle
        stocking_values_2039[6] =  2.69*stocking_redDeer
        stocking_values_2039[8] =  0.95*stocking_tamworthPig
        t30_stockingRate = np.linspace(2039, 2039.95, 2)
        ABC_stockingRate_2039 = solve_ivp(ecoNetwork, (2039, 2039.95), stocking_values_2039,  t_eval = t30_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2040 = ABC_stockingRate_2039.y[0:11, 1:2].flatten()
        # 2040
        stocking_values_2040[1] =  0.65 * stocking_exmoorPony
        stocking_values_2040[2] =  5.88*stocking_fallowDeer
        stocking_values_2040[4] =  1.53*stocking_longhornCattle
        stocking_values_2040[6] =  2.69*stocking_redDeer
        stocking_values_2040[8] =  0.95*stocking_tamworthPig
        t31_stockingRate = np.linspace(2040, 2040.95, 2)
        ABC_stockingRate_2040 = solve_ivp(ecoNetwork, (2040, 2040.95), stocking_values_2040,  t_eval = t31_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2041 = ABC_stockingRate_2040.y[0:11, 1:2].flatten()
        # 2041
        stocking_values_2041[1] =  0.65 * stocking_exmoorPony
        stocking_values_2041[2] =  5.88*stocking_fallowDeer
        stocking_values_2041[4] =  1.53*stocking_longhornCattle
        stocking_values_2041[6] =  2.69*stocking_redDeer
        stocking_values_2041[8] =  0.95*stocking_tamworthPig
        t32_stockingRate = np.linspace(2041, 2041.95, 2)
        ABC_stockingRate_2041 = solve_ivp(ecoNetwork, (2041, 2041.95), stocking_values_2041,  t_eval = t32_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2042 = ABC_stockingRate_2041.y[0:11, 1:2].flatten()
        # 2042
        stocking_values_2042[1] =  0.65 * stocking_exmoorPony
        stocking_values_2042[2] =  5.88*stocking_fallowDeer
        stocking_values_2042[4] =  1.53*stocking_longhornCattle
        stocking_values_2042[6] =  2.69*stocking_redDeer
        stocking_values_2042[8] =  0.95*stocking_tamworthPig
        t33_stockingRate = np.linspace(2042, 2042.95, 2)
        ABC_stockingRate_2042 = solve_ivp(ecoNetwork, (2042, 2042.95), stocking_values_2042,  t_eval = t33_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2043 = ABC_stockingRate_2042.y[0:11, 1:2].flatten()
        # 2043
        stocking_values_2043[1] =  0.65 * stocking_exmoorPony
        stocking_values_2043[2] =  5.88*stocking_fallowDeer
        stocking_values_2043[4] =  1.53*stocking_longhornCattle
        stocking_values_2043[6] =  2.69*stocking_redDeer
        stocking_values_2043[8] =  0.95*stocking_tamworthPig
        t34_stockingRate = np.linspace(2043, 2043.95, 2)
        ABC_stockingRate_2043 = solve_ivp(ecoNetwork, (2043, 2043.95), stocking_values_2043,  t_eval = t34_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2044 = ABC_stockingRate_2043.y[0:11, 1:2].flatten()
        # 2044
        stocking_values_2044[1] =  0.65 * stocking_exmoorPony
        stocking_values_2044[2] =  5.88*stocking_fallowDeer
        stocking_values_2044[4] =  1.53*stocking_longhornCattle
        stocking_values_2044[6] =  2.69*stocking_redDeer
        stocking_values_2044[8] =  0.95*stocking_tamworthPig
        t35_stockingRate = np.linspace(2044, 2044.95, 2)
        ABC_stockingRate_2044 = solve_ivp(ecoNetwork, (2044, 2044.95), stocking_values_2044,  t_eval = t35_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2045 = ABC_stockingRate_2044.y[0:11, 1:2].flatten()
        # 2045
        stocking_values_2045[1] =  0.65*stocking_exmoorPony
        stocking_values_2045[2] =  5.88*stocking_fallowDeer
        stocking_values_2045[4] =  1.53*stocking_longhornCattle
        stocking_values_2045[6] =  2.69*stocking_redDeer
        stocking_values_2045[8] =  0.95*stocking_tamworthPig
        t36_stockingRate = np.linspace(2045, 2046, 2)
        ABC_stockingRate_2045 = solve_ivp(ecoNetwork, (2045, 2046), stocking_values_2045,  t_eval = t36_stockingRate, args=(A_9, r_9), method = 'RK23')
        # concantenate the runs
        combined_runs_stockingRate = np.hstack((ABC_stockingRate_2021.y, ABC_stockingRate_2022.y, ABC_stockingRate_2023.y, ABC_stockingRate_2024.y, ABC_stockingRate_2025.y, ABC_stockingRate_2026.y, ABC_stockingRate_2027.y, ABC_stockingRate_2028.y, ABC_stockingRate_2029.y, ABC_stockingRate_2030.y, ABC_stockingRate_2031.y, ABC_stockingRate_2032.y, ABC_stockingRate_2033.y, ABC_stockingRate_2034.y, ABC_stockingRate_2035.y, ABC_stockingRate_2036.y, ABC_stockingRate_2037.y, ABC_stockingRate_2038.y, ABC_stockingRate_2039.y, ABC_stockingRate_2040.y, ABC_stockingRate_2041.y, ABC_stockingRate_2042.y, ABC_stockingRate_2043.y, ABC_stockingRate_2044.y, ABC_stockingRate_2045.y))
        combined_times_stockingRate = np.hstack((ABC_stockingRate_2021.t, ABC_stockingRate_2022.t, ABC_stockingRate_2023.t, ABC_stockingRate_2024.t, ABC_stockingRate_2025.t, ABC_stockingRate_2026.t, ABC_stockingRate_2027.t, ABC_stockingRate_2028.t, ABC_stockingRate_2029.t, ABC_stockingRate_2030.t, ABC_stockingRate_2031.t, ABC_stockingRate_2032.t, ABC_stockingRate_2033.t, ABC_stockingRate_2034.t, ABC_stockingRate_2035.t, ABC_stockingRate_2036.t, ABC_stockingRate_2037.t, ABC_stockingRate_2038.t, ABC_stockingRate_2039.t, ABC_stockingRate_2040.t, ABC_stockingRate_2041.t, ABC_stockingRate_2042.t, ABC_stockingRate_2043.t, ABC_stockingRate_2044.t, ABC_stockingRate_2045.t))
        all_runs_experiment2 = np.append(all_runs_experiment2, combined_runs_stockingRate)
        # add stocking densities
        combined_stockingDensity_ponies = np.append(combined_stockingDensity_ponies, stocking_exmoorPony)
        combined_stockingDensity_fallowDeer = np.append(combined_stockingDensity_fallowDeer, stocking_fallowDeer)
        combined_stockingDensity_longhornCattle = np.append(combined_stockingDensity_longhornCattle, stocking_longhornCattle)
        combined_stockingDensity_redDeer = np.append(combined_stockingDensity_redDeer, stocking_redDeer)
        combined_stockingDensity_pigs = np.append(combined_stockingDensity_pigs, stocking_tamworthPig)
        # add times
        all_times_experiment2 = np.append(all_times_experiment2, combined_times_stockingRate)
    experiment2 = (np.vstack(np.hsplit(all_runs_experiment2.reshape(len(species)*len(accepted_simulations_2020), 50).transpose(),len(accepted_simulations_2020))))
    experiment2 = pd.DataFrame(data=experiment2, columns=species)
    IDs_3 = np.arange(1,1 + len(accepted_simulations_2020))
    experiment2['ID'] = np.repeat(IDs_3,50)
    experiment2['time'] = all_times_experiment2
    # add stocking densities, repeat them so that there's one stocking density per herbivore species per run
    experiment2['stocking_exmoorPony'] =  np.repeat(combined_stockingDensity_ponies,50)
    experiment2['stocking_fallowDeer'] =  np.repeat(combined_stockingDensity_fallowDeer,50)
    experiment2['stocking_longhornCattle'] =  np.repeat(combined_stockingDensity_longhornCattle,50)
    experiment2['stocking_redDeer'] =  np.repeat(combined_stockingDensity_redDeer,50)
    experiment2['stocking_tamworthPig'] =  np.repeat(combined_stockingDensity_pigs,50)
    # select only the last year 
    accepted_year_exp2 = experiment2.loc[experiment2['time'] == 2046]
    # select the runs where grassland declines by >= 90% (0.13) or 0.36 if parameter set 2
    accepted_simulations_exp2 = accepted_year_exp2[(accepted_year_exp2['grasslandParkland'] <= 0.13)]
    print("number accepted, experiment 2:", accepted_simulations_exp2.shape)

    with open("ps1_stable_numberAcceptedExp2.txt", "w") as text_file:
        print("number accepted, experiment 2: {}".format(accepted_simulations_exp2.shape), file=text_file)

    # match the IDs
    accepted_simulations_exp2 = experiment2[experiment2['ID'].isin(accepted_simulations_exp2['ID'])]
    filtered_FinalRuns_2 = final_runs_2.loc[(final_runs_2['accepted?'] == "Accepted")]
    finalResults_experiment2 = pd.concat([filtered_FinalRuns, filtered_FinalRuns_2, accepted_simulations_exp2])
    finalResults_experiment2['accepted?'] = "noGrass"

    

    # EXPERIMENT 2: What is the range of parameters needed for grass to take over?
    all_runs_experiment25 = []
    all_times_experiment25 = []

    # loop through each row of accepted parameters
    for X0_exp25, r_9, A_9 in zip(X0_experiment2,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # take 10% above and below the optimizer outputs; parameter set 1
        stocking_exmoorPony_25 =  np.random.uniform(low=1.66, high=2)
        stocking_fallowDeer_25 = np.random.uniform(low=1.67, high=2)
        stocking_longhornCattle_25 = np.random.uniform(low=1.65, high=2)
        stocking_redDeer_25 = np.random.uniform(low=4.1, high=5)
        stocking_tamworthPig_25 = np.random.uniform(low=1.89, high=2.3)

        # # parameter set 2
        # stocking_exmoorPony_25 =  np.random.uniform(low=17.2, high=21.1)
        # stocking_fallowDeer_25 = np.random.uniform(low=20.7, high=25.3)
        # stocking_longhornCattle_25 = np.random.uniform(low=12.4, high=15.2)
        # stocking_redDeer_25 = np.random.uniform(low=14.3, high=17.5)
        # stocking_tamworthPig_25 = np.random.uniform(low=1.4, high=1.7)

        # 2021 - future projections
        X0_exp25[1] =  0.65*stocking_exmoorPony_25
        X0_exp25[2] =  5.88*stocking_fallowDeer_25
        X0_exp25[4] =  1.53*stocking_longhornCattle_25
        X0_exp25[6] =  2.69*stocking_redDeer_25
        X0_exp25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2021_25 = solve_ivp(ecoNetwork, (2021,2021.95), X0_exp25,  t_eval = t1_experiment2, args=(A_9, r_9), method = 'RK23')
        stocking_values_2022_25 = ABC_stockingRate_2021_25.y[0:11, 1:2].flatten()
        # 2022
        stocking_values_2022_25[1] =  0.65*stocking_exmoorPony_25
        stocking_values_2022_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2022_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2022_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2022_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2022_25 = solve_ivp(ecoNetwork, (2022,2022.95), stocking_values_2022_25,  t_eval = t13_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2023_25 = ABC_stockingRate_2022_25.y[0:11, 1:2].flatten()
        # 2023
        stocking_values_2023_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2023_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2023_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2023_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2023_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2023_25 = solve_ivp(ecoNetwork, (2023,2023.95), stocking_values_2023_25,  t_eval = t14_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2024_25 = ABC_stockingRate_2023_25.y[0:11, 1:2].flatten()
        # 2024
        stocking_values_2024_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2024_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2024_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2024_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2024_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2024_25 = solve_ivp(ecoNetwork, (2024,2024.95), stocking_values_2024_25,  t_eval = t15_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2025_25 = ABC_stockingRate_2024_25.y[0:11, 1:2].flatten()
        # 2025
        stocking_values_2025_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2025_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2025_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2025_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2025_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2025_25 = solve_ivp(ecoNetwork, (2025,2025.95), stocking_values_2025_25,  t_eval = t16_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2026_25 = ABC_stockingRate_2025_25.y[0:11, 1:2].flatten()
        # 2026
        stocking_values_2026_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2026_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2026_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2026_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2026_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2026_25 = solve_ivp(ecoNetwork, (2026,2026.95), stocking_values_2026_25,  t_eval = t17_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2027_25 = ABC_stockingRate_2026_25.y[0:11, 1:2].flatten()
        # 2027
        stocking_values_2027_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2027_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2027_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2027_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2027_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2027_25 = solve_ivp(ecoNetwork, (2027,2027.95), stocking_values_2027_25,  t_eval = t18_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2028_25 = ABC_stockingRate_2027_25.y[0:11, 1:2].flatten()
        # 2028
        stocking_values_2028_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2028_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2028_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2028_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2028_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2028_25 = solve_ivp(ecoNetwork, (2028,2028.95), stocking_values_2028_25,  t_eval = t19_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2029_25 = ABC_stockingRate_2028_25.y[0:11, 1:2].flatten()
        # 2029
        stocking_values_2029_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2029_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2029_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2029_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2029_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2029_25 = solve_ivp(ecoNetwork, (2029,2029.95), stocking_values_2029_25,  t_eval = t20_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2030_25 = ABC_stockingRate_2029_25.y[0:11, 1:2].flatten()
        # 2030
        stocking_values_2030_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2030_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2030_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2030_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2030_25[8] =  0.95*stocking_tamworthPig_25
        t21_stockingRate = np.linspace(2030, 2030.95, 2)
        ABC_stockingRate_2030_25 = solve_ivp(ecoNetwork, (2030, 2030.95), stocking_values_2030_25,  t_eval = t21_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2031_25 = ABC_stockingRate_2030_25.y[0:11, 1:2].flatten()
        # 2031
        stocking_values_2031_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2031_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2031_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2031_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2031_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2031_25 = solve_ivp(ecoNetwork, (2031, 2031.95), stocking_values_2031_25,  t_eval = t22_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2032_25 = ABC_stockingRate_2031_25.y[0:11, 1:2].flatten()
        # 2032
        stocking_values_2032_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2032_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2032_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2032_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2032_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2032_25 = solve_ivp(ecoNetwork, (2032, 2032.95), stocking_values_2032_25,  t_eval = t23_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2033_25 = ABC_stockingRate_2032_25.y[0:11, 1:2].flatten()
        # 2033
        stocking_values_2033_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2033_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2033_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2033_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2033_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2033_25 = solve_ivp(ecoNetwork, (2033, 2033.95), stocking_values_2033_25,  t_eval = t24_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2034_25 = ABC_stockingRate_2033_25.y[0:11, 1:2].flatten()
        # 2034
        stocking_values_2034_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2034_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2034_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2034_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2034_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2034_25 = solve_ivp(ecoNetwork, (2034, 2034.95), stocking_values_2034_25,  t_eval = t25_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2035_25 = ABC_stockingRate_2034_25.y[0:11, 1:2].flatten()
        # 2035
        stocking_values_2035_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2035_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2035_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2035_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2035_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2035_25 = solve_ivp(ecoNetwork, (2035, 2035.95), stocking_values_2035_25,  t_eval = t26_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2036_25 = ABC_stockingRate_2035_25.y[0:11, 1:2].flatten()
        # 2036
        stocking_values_2036_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2036_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2036_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2036_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2036_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2036_25 = solve_ivp(ecoNetwork, (2036, 2036.95), stocking_values_2036_25,  t_eval = t27_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2037_25 = ABC_stockingRate_2036_25.y[0:11, 1:2].flatten()
        # 2037
        stocking_values_2037_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2037_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2037_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2037_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2037_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2037_25 = solve_ivp(ecoNetwork, (2037, 2037.95), stocking_values_2037_25,  t_eval = t28_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2038_25 = ABC_stockingRate_2037_25.y[0:11, 1:2].flatten()
        # 2038
        stocking_values_2038_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2038_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2038_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2038_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2038_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2038_25 = solve_ivp(ecoNetwork, (2038, 2038.95), stocking_values_2038_25,  t_eval = t29_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2039_25 = ABC_stockingRate_2038_25.y[0:11, 1:2].flatten()
        # 2039
        stocking_values_2039_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2039_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2039_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2039_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2039_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2039_25 = solve_ivp(ecoNetwork, (2039, 2039.95), stocking_values_2039_25,  t_eval = t30_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2040_25 = ABC_stockingRate_2039_25.y[0:11, 1:2].flatten()
        # 2040
        stocking_values_2040_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2040_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2040_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2040_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2040_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2040_25 = solve_ivp(ecoNetwork, (2040, 2040.95), stocking_values_2040_25,  t_eval = t31_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2041_25 = ABC_stockingRate_2040_25.y[0:11, 1:2].flatten()
        # 2041
        stocking_values_2041_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2041_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2041_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2041_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2041_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2041_25 = solve_ivp(ecoNetwork, (2041, 2041.95), stocking_values_2041_25,  t_eval = t32_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2042_25 = ABC_stockingRate_2041_25.y[0:11, 1:2].flatten()
        # 2042
        stocking_values_2042_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2042_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2042_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2042_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2042_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2042_25 = solve_ivp(ecoNetwork, (2042, 2042.95), stocking_values_2042_25,  t_eval = t33_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2043_25 = ABC_stockingRate_2042_25.y[0:11, 1:2].flatten()
        # 2043
        stocking_values_2043_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2043_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2043_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2043_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2043_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2043_25 = solve_ivp(ecoNetwork, (2043, 2043.95), stocking_values_2043_25,  t_eval = t34_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2044_25 = ABC_stockingRate_2043_25.y[0:11, 1:2].flatten()
        # 2044
        stocking_values_2044_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2044_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2044_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2044_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2044_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2044_25 = solve_ivp(ecoNetwork, (2044, 2044.95), stocking_values_2044_25,  t_eval = t35_stockingRate, args=(A_9, r_9), method = 'RK23')
        stocking_values_2045_25 = ABC_stockingRate_2044_25.y[0:11, 1:2].flatten()
        # 2045
        stocking_values_2045_25[1] =  0.65 * stocking_exmoorPony_25
        stocking_values_2045_25[2] =  5.88*stocking_fallowDeer_25
        stocking_values_2045_25[4] =  1.53*stocking_longhornCattle_25
        stocking_values_2045_25[6] =  2.69*stocking_redDeer_25
        stocking_values_2045_25[8] =  0.95*stocking_tamworthPig_25
        ABC_stockingRate_2045_25 = solve_ivp(ecoNetwork, (2045, 2046), stocking_values_2045_25,  t_eval = t36_stockingRate, args=(A_9, r_9), method = 'RK23')
        # concantenate the runs
        combined_runs_stockingRate_25 = np.hstack((ABC_stockingRate_2021_25.y, ABC_stockingRate_2022_25.y, ABC_stockingRate_2023_25.y, ABC_stockingRate_2024_25.y, ABC_stockingRate_2025_25.y, ABC_stockingRate_2026_25.y, ABC_stockingRate_2027_25.y, ABC_stockingRate_2028_25.y, ABC_stockingRate_2029_25.y, ABC_stockingRate_2030_25.y, ABC_stockingRate_2031_25.y, ABC_stockingRate_2032_25.y, ABC_stockingRate_2033_25.y, ABC_stockingRate_2034_25.y, ABC_stockingRate_2035_25.y, ABC_stockingRate_2036_25.y, ABC_stockingRate_2037_25.y, ABC_stockingRate_2038_25.y, ABC_stockingRate_2039_25.y, ABC_stockingRate_2040_25.y, ABC_stockingRate_2041_25.y, ABC_stockingRate_2042_25.y, ABC_stockingRate_2043_25.y, ABC_stockingRate_2044_25.y, ABC_stockingRate_2045_25.y))
        combined_times_stockingRate_25 = np.hstack((ABC_stockingRate_2021_25.t, ABC_stockingRate_2022_25.t, ABC_stockingRate_2023_25.t, ABC_stockingRate_2024_25.t, ABC_stockingRate_2025_25.t, ABC_stockingRate_2026_25.t, ABC_stockingRate_2027_25.t, ABC_stockingRate_2028_25.t, ABC_stockingRate_2029_25.t, ABC_stockingRate_2030_25.t, ABC_stockingRate_2031_25.t, ABC_stockingRate_2032_25.t, ABC_stockingRate_2033_25.t, ABC_stockingRate_2034_25.t, ABC_stockingRate_2035_25.t, ABC_stockingRate_2036_25.t, ABC_stockingRate_2037_25.t, ABC_stockingRate_2038_25.t, ABC_stockingRate_2039_25.t, ABC_stockingRate_2040_25.t, ABC_stockingRate_2041_25.t, ABC_stockingRate_2042_25.t, ABC_stockingRate_2043_25.t, ABC_stockingRate_2044_25.t, ABC_stockingRate_2045_25.t))
        all_runs_experiment25 = np.append(all_runs_experiment25, combined_runs_stockingRate_25)
        # add times
        all_times_experiment25 = np.append(all_times_experiment25, combined_times_stockingRate_25)
    experiment25 = (np.vstack(np.hsplit(all_runs_experiment25.reshape(len(species)*len(accepted_simulations_2020), 50).transpose(),len(accepted_simulations_2020))))
    experiment25 = pd.DataFrame(data=experiment25, columns=species)
    experiment25['ID'] = np.repeat(IDs_3,50)
    experiment25['time'] = all_times_experiment25
    # select only the last year 
    accepted_year_exp25 = experiment25.loc[experiment25['time'] == 2046]
    # select the runs where grassland declines by >= 90% 
    accepted_simulations_exp25 = accepted_year_exp25[(accepted_year_exp25['grasslandParkland'] >= 1.13)]
    print("number accepted, experiment 2.5:", accepted_simulations_exp25.shape)

    with open("ps1_stable_numberAcceptedExp25.txt", "w") as text_file:
        print("number accepted, experiment 2.5: {}".format(accepted_simulations_exp25.shape), file=text_file)


    # match the IDs
    accepted_simulations_exp25 = experiment25[experiment25['ID'].isin(accepted_simulations_exp25['ID'])]
    # corr matrix between stocking densities and grassland
    finalResults_experiment25 = pd.concat([filtered_FinalRuns, filtered_FinalRuns_2, accepted_simulations_exp25])
    finalResults_experiment25['accepted?'] = "allGrass"



  
    # EXPERIMENT 3: What if we reintroduce European bison?
    all_runs_euroBison = []
    all_times_euroBison = []
    X0_euroBison = X0_3.to_numpy()
    t_bison = np.linspace(2021, 2021.95, 2)
    # loop through each row of accepted parameters
    for XO_bisonReintro, r_5, A_5 in zip(X0_euroBison,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # starting values for 2021 - the stocking densities
        XO_bisonReintro[1] =  0.65
        XO_bisonReintro[2] =  5.88
        XO_bisonReintro[4] =  1.53
        XO_bisonReintro[6] =  2.69
        XO_bisonReintro[8] =  0.95
        # add bison
        XO_bisonReintro[0] = 1
        # and their interactions (primary producers & carbon) - parameter set 1
        A_5[3][0] = np.random.uniform(low=-0.001, high=-0.00065)
        A_5[5][0] = np.random.uniform(low=0.0074, high=0.01)
        A_5[9][0] = np.random.uniform(low=-0.1, high=-0.044)
        A_5[10][0] = np.random.uniform(low=-0.01, high=-0.0084)

        # and their interactions (primary producers & carbon) - parameter set 2
        # A_5[3][0] = np.random.uniform(low=-0.001, high=-0.0008)
        # A_5[5][0] = np.random.uniform(low=0.0075, high=0.01)
        # A_5[9][0] = np.random.uniform(low=-0.1, high=-0.042)
        # A_5[10][0] = np.random.uniform(low=-0.01, high=-0.0058)
 
        # 2021 - future projections
        euroBison_ABC_2021 = solve_ivp(ecoNetwork, (2021,2021.95), XO_bisonReintro,  t_eval = t_bison, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2022 = euroBison_ABC_2021.y[0:11, 1:2].flatten()
        # 2022
        bisonReintro_2022[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2022[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2022[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2022[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2022[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2022[8] =  np.random.uniform(low=0.86,high=1.0)
        t_1 = np.linspace(2022, 2022.95, 2)
        euroBison_ABC_2022 = solve_ivp(ecoNetwork, (2022,2022.95), bisonReintro_2022,  t_eval = t_1, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2023 = euroBison_ABC_2022.y[0:11, 1:2].flatten()
        # 2023
        bisonReintro_2023[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2023[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2023[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2023[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2023[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2023[8] =  np.random.uniform(low=0.86,high=1.0)
        t_2 = np.linspace(2023, 2023.95, 2)
        euroBison_ABC_2023 = solve_ivp(ecoNetwork, (2023,2023.95), bisonReintro_2023,  t_eval = t_2, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2024 = euroBison_ABC_2023.y[0:11, 1:2].flatten()
        # 2024
        bisonReintro_2024[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2024[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2024[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2024[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2024[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2024[8] =  np.random.uniform(low=0.86,high=1.0)
        t_3 = np.linspace(2024, 2024.95, 2)
        euroBison_ABC_2024 = solve_ivp(ecoNetwork, (2024,2024.95), bisonReintro_2024,  t_eval = t_3, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2025 = euroBison_ABC_2024.y[0:11, 1:2].flatten()
        # 2025
        bisonReintro_2025[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2025[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2025[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2025[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2025[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2025[8] =  np.random.uniform(low=0.86,high=1.0)
        t_4 = np.linspace(2025, 2025.95, 2)
        euroBison_ABC_2025 = solve_ivp(ecoNetwork, (2025,2025.95), bisonReintro_2025,  t_eval = t_4, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2026 = euroBison_ABC_2025.y[0:11, 1:2].flatten()
        # 2026
        bisonReintro_2026[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2026[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2026[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2026[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2026[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2026[8] =  np.random.uniform(low=0.86,high=1.0)
        t_5 = np.linspace(2026, 2026.95, 2)
        euroBison_ABC_2026 = solve_ivp(ecoNetwork, (2026,2026.95), bisonReintro_2026,  t_eval = t_5, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2027 = euroBison_ABC_2026.y[0:11, 1:2].flatten()
        # 2027
        bisonReintro_2027[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2027[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2027[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2027[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2027[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2027[8] =  np.random.uniform(low=0.86,high=1.0)
        t_6 = np.linspace(2027, 2027.95, 2)
        euroBison_ABC_2027 = solve_ivp(ecoNetwork, (2027,2027.95), bisonReintro_2027,  t_eval = t_6, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2028 = euroBison_ABC_2027.y[0:11, 1:2].flatten()
        # 2028
        bisonReintro_2028[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2028[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2028[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2028[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2028[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2028[8] =  np.random.uniform(low=0.86,high=1.0)
        t_7 = np.linspace(2028, 2028.95, 2)
        euroBison_ABC_2028 = solve_ivp(ecoNetwork, (2028,2028.95), bisonReintro_2028,  t_eval = t_7, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2029 = euroBison_ABC_2028.y[0:11, 1:2].flatten()
        # 2029
        bisonReintro_2029[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2029[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2029[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2029[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2029[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2029[8] =  np.random.uniform(low=0.86,high=1.0)
        t_8 = np.linspace(2029, 2029.95, 2)
        euroBison_ABC_2029 = solve_ivp(ecoNetwork, (2029,2029.95), bisonReintro_2029,  t_eval = t_8, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2030 = euroBison_ABC_2029.y[0:11,1:2].flatten()
        # 2030
        bisonReintro_2030[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2030[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2030[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2030[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2030[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2030[8] =  np.random.uniform(low=0.86,high=1.0)
        t_9 = np.linspace(2030, 2030.95, 2)
        euroBison_ABC_2030 = solve_ivp(ecoNetwork, (2030, 2030.95), bisonReintro_2030,  t_eval = t_9, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2031 = euroBison_ABC_2030.y[0:11,1:2].flatten()
        # 2031
        bisonReintro_2031[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2031[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2031[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2031[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2031[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2031[8] =  np.random.uniform(low=0.86,high=1.0)
        t_10 = np.linspace(2031, 2031.95, 2)
        euroBison_ABC_2031 = solve_ivp(ecoNetwork, (2031, 2031.95), bisonReintro_2031,  t_eval = t_10, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2032 = euroBison_ABC_2031.y[0:11,1:2].flatten()
        # 2032
        bisonReintro_2032[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2032[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2032[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2032[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2032[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2032[8] =  np.random.uniform(low=0.86,high=1.0)
        t_11 = np.linspace(2032, 2032.95, 2)
        euroBison_ABC_2032 = solve_ivp(ecoNetwork, (2032, 2032.95), bisonReintro_2032,  t_eval = t_11, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2033 = euroBison_ABC_2032.y[0:11,1:2].flatten()
        # 2033
        bisonReintro_2033[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2033[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2033[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2033[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2033[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2033[8] =  np.random.uniform(low=0.86,high=1.0)
        t_12 = np.linspace(2033, 2033.95, 2)
        euroBison_ABC_2033 = solve_ivp(ecoNetwork, (2033, 2033.95), bisonReintro_2033,  t_eval = t_12, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2034 = euroBison_ABC_2033.y[0:11,1:2].flatten()
        # 2034
        bisonReintro_2034[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2034[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2034[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2034[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2034[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2034[8] =  np.random.uniform(low=0.86,high=1.0)
        t_13 = np.linspace(2034, 2034.95, 2)
        euroBison_ABC_2034 = solve_ivp(ecoNetwork, (2034, 2034.95), bisonReintro_2034,  t_eval = t_13, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2035 = euroBison_ABC_2034.y[0:11,1:2].flatten()
        # 2035
        bisonReintro_2035[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2035[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2035[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2035[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2035[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2035[8] =  np.random.uniform(low=0.86,high=1.0)
        t_14 = np.linspace(2035,2035.95, 2)
        euroBison_ABC_2035 = solve_ivp(ecoNetwork, (2035,2035.95), bisonReintro_2035,  t_eval = t_14, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2036 = euroBison_ABC_2035.y[0:11,1:2].flatten()
        # 2036
        bisonReintro_2036[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2036[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2036[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2036[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2036[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2036[8] =  np.random.uniform(low=0.86,high=1.0)
        t_15 = np.linspace(2036,2036.95, 2)
        euroBison_ABC_2036 = solve_ivp(ecoNetwork, (2036,2036.95), bisonReintro_2036,  t_eval = t_15, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2037 = euroBison_ABC_2036.y[0:11,1:2].flatten()
        # 2037
        bisonReintro_2037[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2037[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2037[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2037[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2037[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2037[8] =  np.random.uniform(low=0.86,high=1.0)
        t_16 = np.linspace(2037,2037.95, 2)
        euroBison_ABC_2037 = solve_ivp(ecoNetwork, (2037,2037.95), bisonReintro_2037,  t_eval = t_16, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2038 = euroBison_ABC_2037.y[0:11,1:2].flatten()
        # 2038
        bisonReintro_2038[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2038[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2038[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2038[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2038[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2038[8] =  np.random.uniform(low=0.86,high=1.0)
        t_17 = np.linspace(2038,2038.95, 2)
        euroBison_ABC_2038 = solve_ivp(ecoNetwork, (2038,2038.95), bisonReintro_2038,  t_eval = t_17, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2039 = euroBison_ABC_2038.y[0:11,1:2].flatten()
        # 2039
        bisonReintro_2039[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2039[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2039[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2039[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2039[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2039[8] =  np.random.uniform(low=0.86,high=1.0)
        t_18 = np.linspace(2039,2039.95, 2)
        euroBison_ABC_2039 = solve_ivp(ecoNetwork, (2039,2039.95), bisonReintro_2039,  t_eval = t_18, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2040 = euroBison_ABC_2039.y[0:11,1:2].flatten()
        # 2040
        bisonReintro_2040[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2040[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2040[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2040[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2040[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2040[8] =  np.random.uniform(low=0.86,high=1.0)
        t_19 = np.linspace(2040,2040.95, 2)
        euroBison_ABC_2040 = solve_ivp(ecoNetwork, (2040,2040.95), bisonReintro_2040,  t_eval = t_19, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2041 = euroBison_ABC_2040.y[0:11,1:2].flatten()
        # 2041
        bisonReintro_2041[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2041[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2041[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2041[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2041[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2041[8] =  np.random.uniform(low=0.86,high=1.0)
        t_20 = np.linspace(2041,2041.95, 2)
        euroBison_ABC_2041 = solve_ivp(ecoNetwork, (2041,2041.95), bisonReintro_2041,  t_eval = t_20, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2042 = euroBison_ABC_2041.y[0:11,1:2].flatten()
        # 2042
        bisonReintro_2042[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2042[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2042[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2042[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2042[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2042[8] =  np.random.uniform(low=0.86,high=1.0)
        t_21 = np.linspace(2042,2042.95, 2)
        euroBison_ABC_2042 = solve_ivp(ecoNetwork, (2042,2042.95), bisonReintro_2042,  t_eval = t_21, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2043 = euroBison_ABC_2042.y[0:11,1:2].flatten()
        # 2043
        bisonReintro_2043[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2043[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2043[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2043[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2043[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2043[8] =  np.random.uniform(low=0.86,high=1.0)
        t_22 = np.linspace(2043,2043.95, 2)
        euroBison_ABC_2043 = solve_ivp(ecoNetwork, (2043,2043.95), bisonReintro_2043,  t_eval = t_22, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2044 = euroBison_ABC_2043.y[0:11,1:2].flatten()
        # 2044
        bisonReintro_2044[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2044[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2044[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2044[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2044[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2044[8] =  np.random.uniform(low=0.86,high=1.0)
        t_23 = np.linspace(2044,2044.95, 2)
        euroBison_ABC_2044 = solve_ivp(ecoNetwork, (2044,2044.95), bisonReintro_2044,  t_eval = t_23, args=(A_5, r_5), method = 'RK23')
        bisonReintro_2045 = euroBison_ABC_2044.y[0:11,1:2].flatten()
        # 2045
        bisonReintro_2045[0] =  np.random.uniform(low=0.9,high=1.1)
        bisonReintro_2045[1] =  np.random.uniform(low=0.61,high=0.7)
        bisonReintro_2045[2] =  np.random.uniform(low=5.3,high=6.5)
        bisonReintro_2045[4] =  np.random.uniform(low=1.38,high=1.7)
        bisonReintro_2045[6] =  np.random.uniform(low=2.42,high=3.0)
        bisonReintro_2045[8] =  np.random.uniform(low=0.86,high=1.0)
        t_24 = np.linspace(2045, 2046, 2)
        euroBison_ABC_2045 = solve_ivp(ecoNetwork, (2045, 2046), bisonReintro_2045,  t_eval = t_24, args=(A_5, r_5), method = 'RK23')
        # append the runs
        combined_runs_euroBison = np.hstack((euroBison_ABC_2021.y, euroBison_ABC_2022.y, euroBison_ABC_2023.y, euroBison_ABC_2024.y, euroBison_ABC_2025.y, euroBison_ABC_2026.y, euroBison_ABC_2027.y, euroBison_ABC_2028.y, euroBison_ABC_2029.y, euroBison_ABC_2030.y, euroBison_ABC_2031.y, euroBison_ABC_2032.y, euroBison_ABC_2033.y, euroBison_ABC_2034.y, euroBison_ABC_2035.y, euroBison_ABC_2036.y, euroBison_ABC_2037.y, euroBison_ABC_2038.y, euroBison_ABC_2039.y, euroBison_ABC_2040.y, euroBison_ABC_2041.y, euroBison_ABC_2042.y, euroBison_ABC_2043.y, euroBison_ABC_2044.y, euroBison_ABC_2045.y))
        combined_times_euroBison = np.hstack((euroBison_ABC_2021.t, euroBison_ABC_2022.t, euroBison_ABC_2023.t, euroBison_ABC_2024.t, euroBison_ABC_2025.t, euroBison_ABC_2026.t, euroBison_ABC_2027.t, euroBison_ABC_2028.t, euroBison_ABC_2029.t, euroBison_ABC_2030.t, euroBison_ABC_2031.t, euroBison_ABC_2032.t, euroBison_ABC_2033.t, euroBison_ABC_2034.t, euroBison_ABC_2035.t, euroBison_ABC_2036.t, euroBison_ABC_2037.t, euroBison_ABC_2038.t, euroBison_ABC_2039.t, euroBison_ABC_2040.t, euroBison_ABC_2041.t, euroBison_ABC_2042.t, euroBison_ABC_2043.t, euroBison_ABC_2044.t, euroBison_ABC_2045.t))
        all_runs_euroBison = np.append(all_runs_euroBison, combined_runs_euroBison)
        all_times_euroBison = np.append(all_times_euroBison, combined_times_euroBison)
    euroBison = (np.vstack(np.hsplit(all_runs_euroBison.reshape(len(species)*len(accepted_simulations_2020), 50).transpose(),len(accepted_simulations_2020))))
    euroBison = pd.DataFrame(data=euroBison, columns=species)
    IDs_4 = np.arange(1,1 + len(accepted_simulations_2020))
    euroBison['ID'] = np.repeat(IDs_4, 50)
    euroBison['time'] = all_times_euroBison
    # concantenate this will the accepted runs from years 1-5
    euroBison = pd.concat([filtered_FinalRuns, filtered_FinalRuns_2, euroBison])
    euroBison['accepted?'] = "euroBison"

    return final_runs_3, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, finalResults_experiment2, euroBison, accepted_simulations_exp2, finalResults_experiment25, accepted_simulations_exp25




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    final_runs_3, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, finalResults_experiment2, euroBison, accepted_simulations_exp2, finalResults_experiment25, accepted_simulations_exp25 = runODE_3()
    # extract accepted nodes from all dataframes
    accepted_shape1 = np.repeat(final_runs['accepted?'], len(species))
    accepted_shape2 = np.repeat(final_runs_2['accepted?'], len(species))
    accepted_shape3 = np.repeat(final_runs_3['accepted?'], len(species))
    accepted_shape4 = np.repeat(no_reintro['accepted?'], len(species))
    accepted_shape5 = np.repeat(finalResults_experiment2['accepted?'], len(species))
    accepted_shape6 = np.repeat(finalResults_experiment25['accepted?'], len(species))
    accepted_shape7 = np.repeat(euroBison['accepted?'], len(species))
    # concatenate them
    accepted_shape = pd.concat([accepted_shape1, accepted_shape2, accepted_shape3, accepted_shape4, accepted_shape5, accepted_shape6, accepted_shape7], axis=0)
    # add a grouping variable to graph each run separately
    grouping1 = np.repeat(final_runs['ID'], len(species))
    grouping2 = np.repeat(final_runs_2['ID'], len(species))
    grouping3 = np.repeat(final_runs_3['ID'], len(species))
    grouping4 = np.repeat(no_reintro['ID'], len(species))
    grouping5 = np.repeat(finalResults_experiment2['ID'], len(species))
    grouping6 = np.repeat(finalResults_experiment25['ID'], len(species))
    grouping7 = np.repeat(euroBison['ID'], len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping1, grouping2, grouping3, grouping4, grouping5, grouping6, grouping7), axis=0)
    # extract the node values from all dataframes
    final_runs1 = final_runs.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    final_runs2 = final_runs_2.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    final_runs3 = final_runs_3.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    y_noReintro = no_reintro.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    y_experiment2 = finalResults_experiment2.drop(['ID', 'accepted?','time', 'stocking_exmoorPony', 'stocking_fallowDeer', 'stocking_longhornCattle', 'stocking_redDeer', 'stocking_tamworthPig'], axis=1).values.flatten()
    y_experiment25 = finalResults_experiment25.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    y_euroBison = euroBison.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    # concatenate them
    y_values = np.concatenate((final_runs1, final_runs2, final_runs3, y_noReintro, y_experiment2, y_experiment25, y_euroBison), axis=0)   
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_firstRun = np.tile(species, 8*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 24*len(accepted_simulations))
    species_thirdRun = np.tile(species, 50*len(accepted_simulations_2020))
    species_noReintro = np.tile(species, (50*len(accepted_simulations_2020)) + (8*len(accepted_simulations)))
    species_experiment2 = np.tile(species, (8*len(accepted_simulations)) + len(accepted_simulations_exp2)+ (24*len(accepted_simulations_2020)))
    species_experiment3 = np.tile(species, (8*len(accepted_simulations)) + len(accepted_simulations_exp25) + (24*len(accepted_simulations_2020)))
    species_euroBison = np.tile(species, (50*len(accepted_simulations_2020)) + (8*len(accepted_simulations)) + (24*len(accepted_simulations_2020)))
    species_list = np.concatenate((species_firstRun, species_secondRun, species_thirdRun, species_noReintro, species_experiment2, species_experiment3, species_euroBison), axis=0)
    # time 
    firstODEyears = np.repeat(final_runs['time'],len(species))
    secondODEyears = np.repeat(final_runs_2['time'],len(species))
    thirdODEyears = np.repeat(final_runs_3['time'],len(species))
    indices_noReintro = np.repeat(no_reintro['time'],len(species))
    indices_experiment2 = np.repeat(finalResults_experiment2['time'],len(species))
    indices_experiment3 = np.repeat(finalResults_experiment25['time'],len(species))
    indices_euroBison = np.repeat(euroBison['time'],len(species))
    indices = pd.concat([firstODEyears, secondODEyears, thirdODEyears, indices_noReintro, indices_experiment2, indices_experiment3, indices_euroBison], axis=0)
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
    filtered_df = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "noReintro") | (final_df['runType'] == "euroBison")]
    filtered_rejectedAccepted = final_df.loc[(final_df['runType'] == "Accepted") | (final_df['runType'] == "Rejected") ]
    filtered_noGrass = final_df.loc[(final_df['runType'] == "noGrass")]
    filtered_allGrass = final_df.loc[(final_df['runType'] == "allGrass")]

    # Accepted vs. Counterfactual graph (no reintroductions vs. reintroductions) vs. Euro Bison reintro
    colors = ["#6788ee", "#e26952", "#3F9E4D"]
    g = sns.FacetGrid(filtered_df, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=4, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[6].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[4].get_xdata(),ax.lines[4].get_ydata(), ax.lines[7].get_ydata(), color = '#e26952', alpha=0.2)
        ax.fill_between(ax.lines[5].get_xdata(),ax.lines[5].get_ydata(), ax.lines[8].get_ydata(), color = "#3F9E4D", alpha=0.2)
        ax.set_ylabel('Abundance')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    # add subplot titles
    axes = g.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("European bison")
    axes[1].set_title("Exmoor ponies")
    axes[2].set_title("Fallow deer")
    axes[3].set_title("Grassland & parkland")
    axes[4].set_title("Longhorn cattle")
    axes[5].set_title("Organic carbon")
    axes[6].set_title("Red deer")
    axes[7].set_title("Roe deer")
    axes[8].set_title("Tamworth pigs")
    axes[9].set_title("Thorny scrubland")
    axes[10].set_title("Woodland")
    # add filter lines
    g.axes[3].vlines(x=2009,ymin=0.68,ymax=1, color='r')
    g.axes[5].vlines(x=2009,ymin=1,ymax=2.1, color='r')
    g.axes[7].vlines(x=2009,ymin=1,ymax=3.3, color='r')
    g.axes[9].vlines(x=2009,ymin=1,ymax=19, color='r')
    g.axes[10].vlines(x=2009,ymin=0.85,ymax=1.6, color='r')
    # plot next set of filter lines
    g.axes[3].vlines(x=2021,ymin=0.61,ymax=0.86, color='r')
    g.axes[5].vlines(x=2021,ymin=1.7,ymax=2.2, color='r')
    g.axes[7].vlines(x=2021,ymin=1.7,ymax=3.3, color='r')
    g.axes[9].vlines(x=2021,ymin=19,ymax=31.9, color='r')
    g.axes[10].vlines(x=2021,ymin=1,ymax=1.7, color='r')
    # make sure they all start from 0
    g.axes[4].set(ylim =(0,None))
    g.axes[6].set(ylim =(0,None))
    g.axes[9].set(ylim =(0,None))
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Reintroductions', 'No reintroductions', 'European bison \n reintroduction'],bbox_to_anchor=(2.2, 0),loc='lower right', fontsize=12)
    plt.savefig('reintroNoReintro_ps1_stableOnly.png')
    # plt.show()


    # Accepted vs. rejected runs graph
    r = sns.FacetGrid(filtered_rejectedAccepted, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=4, sharey = False)
    r.map(sns.lineplot, 'Time', 'Median')
    r.map(sns.lineplot, 'Time', 'fivePerc')
    r.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in r.axes.flat:
        ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    # add subplot titles
    axes = r.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("European bison")
    axes[1].set_title("Exmoor ponies")
    axes[2].set_title("Fallow deer")
    axes[3].set_title("Grassland & parkland")
    axes[4].set_title("Longhorn cattle")
    axes[5].set_title("Organic carbon")
    axes[6].set_title("Red deer")
    axes[7].set_title("Roe deer")
    axes[8].set_title("Tamworth pigs")
    axes[9].set_title("Thorny scrubland")
    axes[10].set_title("Woodland")
    # add filter lines
    r.axes[3].vlines(x=2009,ymin=0.68,ymax=1, color='r')
    r.axes[5].vlines(x=2009,ymin=1,ymax=2.1, color='r')
    r.axes[7].vlines(x=2009,ymin=1,ymax=3.3, color='r')
    r.axes[9].vlines(x=2009,ymin=1,ymax=19, color='r')
    r.axes[10].vlines(x=2009,ymin=0.85,ymax=1.6, color='r')
    # plot next set of filter lines
    r.axes[3].vlines(x=2021,ymin=0.61,ymax=0.86, color='r')
    r.axes[5].vlines(x=2021,ymin=1.7,ymax=2.2, color='r')
    r.axes[7].vlines(x=2021,ymin=1.7,ymax=3.3, color='r')
    r.axes[9].vlines(x=2021,ymin=19,ymax=31.9, color='r')
    r.axes[10].vlines(x=2021,ymin=1,ymax=1.7, color='r')
    # stop the plots from overlapping
    plt.tight_layout()
    plt.legend(labels=['Rejected Runs', 'Accepted Runs'],bbox_to_anchor=(2.2, 0), loc='lower right', fontsize=12)
    plt.savefig('acceptedRejected_ps1_stableOnly.png')
    # plt.show()


    # Pushing grassland to zero
    colors_2 = ["#6788ee"]
    n = sns.FacetGrid(filtered_noGrass, col="Ecosystem Element", hue = "runType", palette = colors_2, col_wrap=4, sharey = False)
    n.map(sns.lineplot, 'Time', 'Median')
    n.map(sns.lineplot, 'Time', 'fivePerc')
    n.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    # fill in lines between quantiles
    # median - 0 1 2 3
    # upper - 4 5 6 7
    # lower - 8 9 10 11
    for ax in n.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha=0.2)
        ax.set_ylabel('Abundance')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    # add subplot titles
    axes = n.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("European bison")
    axes[1].set_title("Exmoor ponies")
    axes[2].set_title("Fallow deer")
    axes[3].set_title("Grassland & parkland")
    axes[4].set_title("Longhorn cattle")
    axes[5].set_title("Organic carbon")
    axes[6].set_title("Red deer")
    axes[7].set_title("Roe deer")
    axes[8].set_title("Tamworth pigs")
    axes[9].set_title("Thorny scrubland")
    axes[10].set_title("Woodland")
    # add filter lines
    n.axes[3].vlines(x=2009,ymin=0.68,ymax=1, color='r')
    n.axes[5].vlines(x=2009,ymin=1,ymax=2.1, color='r')
    n.axes[7].vlines(x=2009,ymin=1,ymax=3.3, color='r')
    n.axes[9].vlines(x=2009,ymin=1,ymax=19, color='r')
    n.axes[10].vlines(x=2009,ymin=0.85,ymax=1.6, color='r')
    # plot next set of filter lines
    n.axes[3].vlines(x=2021,ymin=0.61,ymax=0.86, color='r')
    n.axes[5].vlines(x=2021,ymin=1.7,ymax=2.2, color='r')
    n.axes[7].vlines(x=2021,ymin=1.7,ymax=3.3, color='r')
    n.axes[9].vlines(x=2021,ymin=19,ymax=31.9, color='r')
    n.axes[10].vlines(x=2021,ymin=1,ymax=1.7, color='r')
    # stop the plots from overlapping
    n.fig.suptitle('Scenarios in which grassland collapses (< 10%)', fontsize = 14) 
    plt.tight_layout()
    plt.savefig('zeroGrass_ps1_stableOnly.png')
    # plt.show()


    # Phase shift to grassland
    colors_2 = ["#6788ee"]
    e = sns.FacetGrid(filtered_allGrass, col="Ecosystem Element", hue = "runType", palette = colors_2, col_wrap=4, sharey = False)
    e.map(sns.lineplot, 'Time', 'Median')
    e.map(sns.lineplot, 'Time', 'fivePerc')
    e.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in e.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha=0.2)
        ax.set_ylabel('Abundance')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
    # add subplot titles
    axes = e.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("European bison")
    axes[1].set_title("Exmoor ponies")
    axes[2].set_title("Fallow deer")
    axes[3].set_title("Grassland & parkland")
    axes[4].set_title("Longhorn cattle")
    axes[5].set_title("Organic carbon")
    axes[6].set_title("Red deer")
    axes[7].set_title("Roe deer")
    axes[8].set_title("Tamworth pigs")
    axes[9].set_title("Thorny scrubland")
    axes[10].set_title("Woodland")
    # add filter lines
    e.axes[3].vlines(x=2009,ymin=0.68,ymax=1, color='r')
    e.axes[5].vlines(x=2009,ymin=1,ymax=2.1, color='r')
    e.axes[7].vlines(x=2009,ymin=1,ymax=3.3, color='r')
    e.axes[9].vlines(x=2009,ymin=1,ymax=19, color='r')
    e.axes[10].vlines(x=2009,ymin=0.85,ymax=1.6, color='r')
    # plot next set of filter lines
    e.axes[3].vlines(x=2021,ymin=0.61,ymax=0.86, color='r')
    e.axes[5].vlines(x=2021,ymin=1.7,ymax=2.2, color='r')
    e.axes[7].vlines(x=2021,ymin=1.7,ymax=3.3, color='r')
    e.axes[9].vlines(x=2021,ymin=19,ymax=31.9, color='r')
    e.axes[10].vlines(x=2021,ymin=1,ymax=1.7, color='r')
    # stop the plots from overlapping
    e.fig.suptitle('Scenarios in which grassland takes over (> 90%)', fontsize = 14) 
    plt.tight_layout()
    plt.savefig('allGrass_ps1_stableOnly.png')
    # plt.show()

plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
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
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # fallow deer
                [0, -1, 1, 0, 0, 0, 0, 0, 1, 1],
                # grassland parkland
                [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1],
                # longhorn cattle
                [0, 0, 1, -1, 0, 0, 0, 0, 1, 1],
                # organic carbon
                [1, 1, 1, 1, -1, 1, 1, 1, 1, 1],
                # red deer
                [0, 0, 1, 0, 0, -1, 0, 0, 1, 1],
                # roe deer
                [0, 0, 1, 0, 0, 0, -1, 0, 1, 1],
                # tamworth pig
                [0, 0, 1, 0, 0, 0, 0, -1, 1, 1],
                # thorny scrub
                [-1, -1, 0, -1, 0, -1, -1, -1, -1, -1],
                # woodland
                [-1, -1, 0, -1, 0, -1, -1, -1, 1, -1]
                ]
    # generate random uniform numbers
    variation = np.random.uniform(low = 0.9, high=1.1, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix


def generateGrowth():
    growthRates = [0, 0, 0.93, 0, 0, 0, 0, 0, 0.67, 0.027] 
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
    t = np.linspace(0, 48, 144)
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
                first_ABC = solve_ivp(ecoNetwork, (0, 48), X0,  t_eval = t, args=(A, r), method = 'RK23')
                # append all the runs
                all_runs = np.append(all_runs, first_ABC.y)
                all_times = np.append(all_times, first_ABC.t)
                # add one to the counter (so we know how many simulations were run in the end)
                NUMBER_OF_SIMULATIONS += 1

    # check the final runs
    print("number of stable simulations", NUMBER_STABLE)
    print("number of stable & ecologically sound simulations", NUMBER_OF_SIMULATIONS)
    
    # put together the final runs
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 144).transpose(),NUMBER_OF_SIMULATIONS)))
    final_runs = pd.DataFrame(data=final_runs, columns=species)
    final_runs['time'] = all_times

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
    final_runs['ID'] = np.repeat(IDs,144)
    # select only the last year
    accepted_year = final_runs.loc[final_runs['time'] == 48]
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
    # # add reintroduced species; ponies, longhorn cattle, and tamworth were reintroduced in 2009
    X0_2.loc[:, 'exmoorPony'] = [1 for i in X0_2.index]
    X0_2.loc[:, 'longhornCattle'] = [1 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [1 for i in X0_2.index]
    X0_secondRun = X0_2.to_numpy()
    # # select interaction matrices part of the dataframes 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    # turn to array
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
        # run the model for one year (2009), (to account for herbivore numbers being manually controlled every year)
        t = np.linspace(48, 59.95, 36)
        second_ABC = solve_ivp(ecoNetwork, (48,59.95), X0_3,  t_eval = t, args=(A_2, r_2), method = 'RK23')
        # identify last month (year running March-March like at Knepp)
        starting_values_2010 = second_ABC.y[0:10, 35:36].flatten()
        # force 2010's values; fallow deer reintroduced this year
        starting_values_2010[0] = 0.57
        starting_values_2010[1] = 1
        starting_values_2010[3] = 1.45
        starting_values_2010[7] = 0.85
        t_1 = np.linspace(60, 71.95, 36)
        third_ABC = solve_ivp(ecoNetwork, (60,71.95), starting_values_2010,  t_eval = t_1, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year (2011), adding forcings
        starting_values_2011 = third_ABC.y[0:10, 35:36].flatten()
        starting_values_2011[0] = 0.65
        starting_values_2011[1] = 1.93
        starting_values_2011[3] = 1.74
        starting_values_2011[7] = 1.1
        t_2 = np.linspace(72, 83.95, 36)
        fourth_ABC = solve_ivp(ecoNetwork, (72,83.95), starting_values_2011,  t_eval = t_2, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year (2012), adding forcings
        starting_values_2012 = fourth_ABC.y[0:10, 35:36].flatten()
        starting_values_2012[0] = 0.74
        starting_values_2012[1] = 2.38
        starting_values_2012[3] = 2.19
        starting_values_2012[7] = 1.65
        t_3 = np.linspace(84, 95.95, 36)
        fifth_ABC = solve_ivp(ecoNetwork, (84,95.95), starting_values_2012,  t_eval = t_3, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year (2013), adding forcings
        starting_values_2013 = fifth_ABC.y[0:10, 35:36].flatten()
        starting_values_2013[0] = 0.43
        starting_values_2013[1] = 2.38
        starting_values_2013[3] = 2.43
        # red deer reintroduced
        starting_values_2013[5] = 1
        starting_values_2013[7] = 0.3
        t_4 = np.linspace(96, 107.95, 36)
        sixth_ABC = solve_ivp(ecoNetwork, (96,107.95), starting_values_2013,  t_eval = t_4, args=(A_2, r_2), method = 'RK23')
        # take those values and re-run for another year (2014), adding forcings
        starting_values_2014 = sixth_ABC.y[0:10, 35:36].flatten()
        starting_values_2014[0] = 0.43
        starting_values_2014[1] = 2.38
        starting_values_2014[3] = 4.98
        starting_values_2014[5] = 1
        starting_values_2014[7] = 0.9
        t_5 = np.linspace(108, 119.95, 36)
        seventh_ABC = solve_ivp(ecoNetwork, (108,119.95), starting_values_2014,  t_eval = t_5, args=(A_2, r_2), method = 'RK23')
        # set numbers for 2015
        starting_values_2015 = seventh_ABC.y[0:10, 35:36].flatten()
        starting_values_2015[0] = 0.43
        starting_values_2015[1] = 2.38
        starting_values_2015[3] = 2.01
        starting_values_2015[5] = 1
        starting_values_2015[7] = 0.9


        ## 2015 ## 
        # now go month by month: March & April 2015
        t_March2015 = np.linspace(120, 121.95, 6)
        March2015_ABC = solve_ivp(ecoNetwork, (120,121.95), starting_values_2015,  t_eval = t_March2015, args=(A_2, r_2), method = 'RK23')
        starting_values_May2015 = March2015_ABC.y[0:10, 5:6].flatten()
        # pigs had one death in April
        starting_values_May2015[7] = 1.1
        # May 2015
        t_May2015 = np.linspace(122, 122.95, 3)
        May2015_ABC = solve_ivp(ecoNetwork, (122,122.95), starting_values_May2015,  t_eval = t_May2015, args=(A_2, r_2), method = 'RK23')
        starting_values_June2015 = May2015_ABC.y[0:10, 2:3].flatten()
        # pigs had 8 deaths in May
        starting_values_June2015[7] = 0.7
        # June 2015
        t_June2015 = np.linspace(123, 123.95, 3)
        June2015_ABC = solve_ivp(ecoNetwork, (123,123.95), starting_values_June2015,  t_eval = t_June2015, args=(A_2, r_2), method = 'RK23')
        starting_values_July2015 = June2015_ABC.y[0:10, 2:3].flatten()
        # 5 cow deaths in June 2015 (net 5 cows)
        starting_values_July2015[3] = 2.43
        # July & Aug 2015
        t_July2015 = np.linspace(124, 125.95, 6)
        July2015_ABC = solve_ivp(ecoNetwork, (124,125.95), starting_values_July2015,  t_eval = t_July2015, args=(A_2, r_2), method = 'RK23')
        starting_values_Sept2015 = July2015_ABC.y[0:10, 5:6].flatten()
        # 2 fallow deer deaths in August
        starting_values_Sept2015[1] = starting_values_Sept2015[1] - 0.048
        # Sept 2015
        t_Sept2015 = np.linspace(126, 126.95, 3)
        Sept2015_ABC = solve_ivp(ecoNetwork, (126,126.95), starting_values_Sept2015,  t_eval = t_Sept2015, args=(A_2, r_2), method = 'RK23')
        starting_values_Oct2015 = Sept2015_ABC.y[0:10, 2:3].flatten()
        # 2 fallow deer killed in September
        starting_values_Oct2015[1] = starting_values_Oct2015[1]-0.048
        # Oct 2015
        t_Oct2015 = np.linspace(127, 127.95, 3)
        Oct2015_ABC = solve_ivp(ecoNetwork, (127,127.95), starting_values_Oct2015,  t_eval = t_Oct2015, args=(A_2, r_2), method = 'RK23')
        starting_values_Nov2015 = Oct2015_ABC.y[0:10, 2:3].flatten()
        # 3 fallow deer culled, 39 cows culled
        starting_values_Nov2015[1] = starting_values_Nov2015[1] - 0.072
        starting_values_Nov2015[3] = 1.72
        # Nov 2015
        t_Nov2015 = np.linspace(128, 128.95, 3)
        Nov2015_ABC = solve_ivp(ecoNetwork, (128,128.95), starting_values_Nov2015,  t_eval = t_Nov2015, args=(A_2, r_2), method = 'RK23')
        starting_values_Dec2015 = Nov2015_ABC.y[0:10, 2:3].flatten()
        # 7 fallow deer culled, 1 pig culled in November
        starting_values_Dec2015[1] = starting_values_Dec2015[1] - 0.17
        starting_values_Dec2015[7] = 0.65
        # Dec 2015
        t_Dec2015 = np.linspace(129, 129.95, 3)
        Dec2015_ABC = solve_ivp(ecoNetwork, (129,129.95), starting_values_Dec2015,  t_eval = t_Dec2015, args=(A_2, r_2), method = 'RK23')
        starting_values_Jan2016 = Dec2015_ABC.y[0:10, 2:3].flatten()
        # 6 fallow deer culled, 5 pigs moved offsite in Dec 2015
        starting_values_Jan2016[1] = starting_values_Jan2016[1] - 0.14
        starting_values_Jan2016[3] = 1.6
        # Jan 2016
        t_Jan2016 = np.linspace(130, 130.95, 3)
        Jan2016_ABC = solve_ivp(ecoNetwork, (130,130.95), starting_values_Jan2016,  t_eval = t_Jan2016, args=(A_2, r_2), method = 'RK23')
        starting_values_Feb2016 = Jan2016_ABC.y[0:10, 2:3].flatten()
        # 7 fallow deer culled, 1 pig added, 4 pigs culled in Jan 2016
        starting_values_Feb2016[1] = starting_values_Feb2016[1] - 0.17
        starting_values_Feb2016[7] = 0.50
        # Feb 2016
        t_Feb2016 = np.linspace(131, 131.95, 3)
        Feb2016_ABC = solve_ivp(ecoNetwork, (131,131.95), starting_values_Feb2016,  t_eval = t_Feb2016, args=(A_2, r_2), method = 'RK23')
        starting_values_March2016 = Feb2016_ABC.y[0:10, 2:3].flatten()
        # 10 fallow deer culled, 2 pigs killed in Feb 2016
        starting_values_March2016[1] = starting_values_March2016[1] - 0.24
        starting_values_March2016[7] = 0.35


        ## 2016 ##
        # March 2016
        t_March2016 = np.linspace(132, 132.95, 3)
        March2016_ABC = solve_ivp(ecoNetwork, (132,132.95), starting_values_March2016,  t_eval = t_March2016, args=(A_2, r_2), method = 'RK23')
        starting_values_April2016 = March2016_ABC.y[0:10, 2:3].flatten()
        # change values: 1 pony added, 3 pigs moved in and 4 moved out in March 2016
        starting_values_April2016[0] = 0.48
        starting_values_April2016[7] = 0.45
        # April 2016
        t_Apr2016 = np.linspace(133, 133.95, 3)
        April2016_ABC = solve_ivp(ecoNetwork, (133,133.95), starting_values_April2016,  t_eval = t_Apr2016, args=(A_2, r_2), method = 'RK23')
        starting_values_May2016 = April2016_ABC.y[0:10, 2:3].flatten()
        # 1 cattle moved on-site
        starting_values_May2016[3] = 1.94
        # May 2016
        t_May2016 = np.linspace(134, 134.95, 3)
        May2016_ABC = solve_ivp(ecoNetwork, (134,134.95), starting_values_May2016,  t_eval = t_May2016, args=(A_2, r_2), method = 'RK23')
        starting_values_June2016 = May2016_ABC.y[0:10, 2:3].flatten()
        # 2 cow deaths
        starting_values_June2016[3] = 2.04
        # June 2016
        t_June2016 = np.linspace(135, 135.95, 3)
        June2016_ABC = solve_ivp(ecoNetwork, (135,135.95), starting_values_June2016,  t_eval = t_June2016, args=(A_2, r_2), method = 'RK23')
        starting_values_July2016 = June2016_ABC.y[0:10, 2:3].flatten()
        # 30 cows sold/moved off site, and 4 moved on-site
        starting_values_July2016[3] = 1.68
        # July 2016
        t_July2016 = np.linspace(136, 136.95, 3)
        July2016_ABC = solve_ivp(ecoNetwork, (136,136.95), starting_values_July2016,  t_eval = t_July2016, args=(A_2, r_2), method = 'RK23')
        starting_values_Aug2016 = July2016_ABC.y[0:10, 2:3].flatten()
        # 2 cows sold
        starting_values_Aug2016[3] = 1.64
        # Aug-Sept 2016
        t_Aug2016 = np.linspace(137, 137.95, 3)
        Aug2016_ABC = solve_ivp(ecoNetwork, (137,137.95), starting_values_Aug2016,  t_eval = t_Aug2016, args=(A_2, r_2), method = 'RK23')
        starting_values_Sep2016 = Aug2016_ABC.y[0:10, 2:3].flatten()
        # 5 fallow deer deaths
        starting_values_Sep2016[1] = starting_values_Sep2016[1] - 0.12
        # Sept-Oct 2016
        t_Sept2016 = np.linspace(138, 138.95, 3)
        Sept2016_ABC = solve_ivp(ecoNetwork, (138,138.95), starting_values_Sep2016,  t_eval = t_Sept2016, args=(A_2, r_2), method = 'RK23')
        starting_values_Oct2016 = Sept2016_ABC.y[0:10, 2:3].flatten()
        # 9 cows sold, 19 born or moved onto site (unclear)
        starting_values_Oct2016[3] = 1.83
        # Oct & Nov 2016
        t_Oct2016 = np.linspace(139, 140.95, 6)
        Oct2016_ABC = solve_ivp(ecoNetwork, (139,140.95), starting_values_Oct2016,  t_eval = t_Oct2016, args=(A_2, r_2), method = 'RK23')
        starting_values_Dec2016 = Oct2016_ABC.y[0:10, 5:6].flatten()
        # 3 fallow deaths, 5 cow sales
        starting_values_Dec2016[1] = starting_values_Dec2016[1] - 0.072
        starting_values_Dec2016[3] = 1.74
        # Dec 2016
        t_Dec2016 = np.linspace(141, 141.95, 3)
        Dec2016_ABC = solve_ivp(ecoNetwork, (141,141.95), starting_values_Dec2016,  t_eval = t_Dec2016, args=(A_2, r_2), method = 'RK23')
        starting_values_Jan2017 = Dec2016_ABC.y[0:10, 2:3].flatten()
        # 9 fallow deaths, 4 pig sales, 13 cow sales
        starting_values_Jan2017[1] = starting_values_Jan2017[1] - 0.22
        starting_values_Jan2017[3] = 1.49
        starting_values_Jan2017[7] = 0.65
        # Jan 2017
        t_Jan2017 = np.linspace(142, 142.95, 3)
        Jan2017_ABC = solve_ivp(ecoNetwork, (142,142.95), starting_values_Jan2017,  t_eval = t_Jan2017, args=(A_2, r_2), method = 'RK23')
        starting_values_Feb2017 = Jan2017_ABC.y[0:10, 2:3].flatten()
        # 4 pigs sold
        starting_values_Feb2017[7] = 0.45
        # Feb 2017
        t_Feb2017 = np.linspace(143, 143.95, 3)
        Feb2017_ABC = solve_ivp(ecoNetwork, (143,143.95), starting_values_Feb2017,  t_eval = t_Feb2017, args=(A_2, r_2), method = 'RK23')
        starting_values_March2017 = Feb2017_ABC.y[0:10, 2:3].flatten()
        # 10 fallow deer, 2 pigs killed, and in 2017 they start with 14 red deer (not clear if culled) and one less pony
        starting_values_March2017[0] = 0.43
        starting_values_March2017[1] = starting_values_March2017[1] - 0.24
        starting_values_March2017[5] = 1.08
        starting_values_March2017[7] = 0.35


        ## 2017 ##
        # March & April 2017
        t_March2017 = np.linspace(144, 145.95, 6)
        March2017_ABC = solve_ivp(ecoNetwork, (144,145.95), starting_values_March2017,  t_eval = t_March2017, args=(A_2, r_2), method = 'RK23')
        starting_values_May2017 = March2017_ABC.y[0:10, 5:6].flatten()
        # 3 cows added moved on-site
        starting_values_May2017[3] = 1.89
        # May & June 2017
        t_May2017 = np.linspace(146, 147.95, 6)
        May2017_ABC = solve_ivp(ecoNetwork, (146,147.95), starting_values_May2017,  t_eval = t_May2017, args=(A_2, r_2), method = 'RK23')
        starting_values_July2017 = May2017_ABC.y[0:10, 5:6].flatten()
        # 24 cows moved off-site and 3 moved on-site
        starting_values_July2017[3] = 1.77
        # July & Aug 2017
        t_July2017 = np.linspace(148, 149.95, 6)
        July2017_ABC = solve_ivp(ecoNetwork, (148,149.95), starting_values_July2017,  t_eval = t_July2017, args=(A_2, r_2), method = 'RK23')
        starting_values_Sept2017 = July2017_ABC.y[0:10, 5:6].flatten()
        # 16 fallow deer deaths
        starting_values_Sept2017[1] = starting_values_Sept2017[1] - 0.38
        # Sept 2017
        t_Sept2017 = np.linspace(150, 150.95, 3)
        Sept2017_ABC = solve_ivp(ecoNetwork, (150, 150.95), starting_values_Sept2017,  t_eval = t_Sept2017, args=(A_2, r_2), method = 'RK23')
        starting_values_Oct2017 = Sept2017_ABC.y[0:10, 2:3].flatten()
        # 5 fallow deaths, 24 cows sold and 3 moved off-site, and 23 moved on-site
        starting_values_Oct2017[1] = starting_values_Oct2017[1] - 0.12
        starting_values_Oct2017[3] = 1.70
        # Oct 2017
        t_Oct2017 = np.linspace(151, 151.95, 3)
        Oct2017_ABC = solve_ivp(ecoNetwork, (151,151.95), starting_values_Oct2017,  t_eval = t_Oct2017, args=(A_2, r_2), method = 'RK23')
        starting_values_Nov2017 = Oct2017_ABC.y[0:10, 2:3].flatten()
        # 4 fallow deaths, 2 cows moved off-site
        starting_values_Nov2017[1] = starting_values_Nov2017[1] - 0.096
        starting_values_Nov2017[3] = 1.66
        # Nov 2017
        t_Nov2017 = np.linspace(152, 152.95, 3)
        Nov2017_ABC = solve_ivp(ecoNetwork, (152,152.95), starting_values_Nov2017,  t_eval = t_Nov2017, args=(A_2, r_2), method = 'RK23')
        starting_values_Dec2017 = Nov2017_ABC.y[0:10, 2:3].flatten()
        # 2 fallow deer deaths
        starting_values_Dec2017[1] = starting_values_Dec2017[1] - 0.024
        # Dec 2018
        t_Dec2017 = np.linspace(153, 153.95, 3)
        Dec2017_ABC = solve_ivp(ecoNetwork, (153,153.95), starting_values_Dec2017,  t_eval = t_Dec2017, args=(A_2, r_2), method = 'RK23')
        starting_values_Jan2018 = Dec2017_ABC.y[0:10, 2:3].flatten()
        # 46 fallow deaths, 1 red deer death, 4 pig sales
        starting_values_Jan2018[1] = starting_values_Jan2018[1] - 1.1
        starting_values_Jan2018[5] = starting_values_Jan2018[5] - 0.08
        starting_values_Jan2018[7] = 0.9
        # Jan 2018
        t_Jan2018 = np.linspace(154, 154.95, 3)
        Jan2018_ABC = solve_ivp(ecoNetwork, (154,154.95), starting_values_Jan2018,  t_eval = t_Jan2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Feb2018 = Jan2018_ABC.y[0:10, 2:3].flatten()
        # 9 pigs sold
        starting_values_Feb2018[7] = 0.55
        # Feb 2018
        t_Feb2018 = np.linspace(155, 155.95, 3)
        Feb2018_ABC = solve_ivp(ecoNetwork, (155,155.95), starting_values_Feb2018,  t_eval = t_Feb2018, args=(A_2, r_2), method = 'RK23')
        starting_values_March2018 = Feb2018_ABC.y[0:10, 2:3].flatten()
        # 14 fallow deaths, 1 red deer death, ponies back to 9
        starting_values_March2018[0] = 0.39
        starting_values_March2018[1] = starting_values_March2018[1] - 0.33
        starting_values_March2018[5] = starting_values_March2018[5] - 0.08


        ## 2018 ##
        
        # March & April 2018
        t_March2018 = np.linspace(156, 157.95, 6)
        March2018_ABC = solve_ivp(ecoNetwork, (156,157.95), starting_values_March2018,  t_eval = t_March2018, args=(A_2, r_2), method = 'RK23')
        starting_values_May2018 = March2018_ABC.y[0:10, 5:6].flatten()
        # 1 cow moved on-site
        starting_values_May2018[3] = 1.91
        # May & June 2018
        t_May2018 = np.linspace(158, 159.95, 6)
        May2018_ABC = solve_ivp(ecoNetwork, (158,159.95), starting_values_May2018,  t_eval = t_May2018, args=(A_2, r_2), method = 'RK23')
        starting_values_July2018 = May2018_ABC.y[0:10, 5:6].flatten()
        # 2 cows moved on-site, 22 cow deaths/moved off-site
        starting_values_July2018[3] = 1.94
        # July 2018
        t_July2018 = np.linspace(160, 160.95, 3)
        July2018_ABC = solve_ivp(ecoNetwork, (160,160.95), starting_values_July2018,  t_eval = t_July2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Aug2018 = July2018_ABC.y[0:10, 2:3].flatten()
        # 1 red deer death, 1 pig death
        starting_values_Aug2018[5] = starting_values_Aug2018[5] - 0.077
        starting_values_Aug2018[7] = 1.1
        # Aug 2018
        t_Aug2018 = np.linspace(161, 161.95, 3)
        Aug2018_ABC = solve_ivp(ecoNetwork, (161,161.95), starting_values_Aug2018,  t_eval = t_Aug2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Sept2018 = Aug2018_ABC.y[0:10, 2:3].flatten()
        # 1 red deer death, 15 fallow deer deaths, 9 pony transfers, 1 longhorn transfer
        starting_values_Sept2018[0] = 0
        starting_values_Sept2018[1] = starting_values_Sept2018[1] - 0.36
        starting_values_Sept2018[3] = 1.92
        starting_values_Sept2018[5] = starting_values_Sept2018[5] - 0.077
        # Sept 2018
        t_Sept2018 = np.linspace(162, 162.95, 3)
        Sept2018_ABC = solve_ivp(ecoNetwork, (162,162.95), starting_values_Sept2018,  t_eval = t_Sept2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Oct2018 = Sept2018_ABC.y[0:10, 2:3].flatten()
        # 19 fallow deer deaths, 14 longhorns sold and 2 moved off-site, 20 longhorns moved on-site
        starting_values_Oct2018[1] = starting_values_Oct2018[1] - 0.45
        starting_values_Oct2018[3] = 2.00
        # Oct 2018
        t_Oct2018 = np.linspace(163, 163.9, 3)
        Oct2018_ABC = solve_ivp(ecoNetwork, (163,163.9), starting_values_Oct2018,  t_eval = t_Oct2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Nov2018 = Oct2018_ABC.y[0:10, 2:3].flatten()
        # 4 fallow deaths, 1 tamworth death, 5 longhorn sales
        starting_values_Nov2018[1] = starting_values_Nov2018[1] - 0.096
        starting_values_Nov2018[3] = 1.91
        starting_values_Nov2018[7] = 1.05
        # Nov 2018
        t_Nov2018 = np.linspace(164, 164.9, 3)
        Nov2018_ABC = solve_ivp(ecoNetwork, (164,164.9), starting_values_Nov2018,  t_eval = t_Nov2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Dec2018 = Nov2018_ABC.y[0:10, 2:3].flatten()
        # 8 longhorn sales, 12 pig sales
        starting_values_Dec2018[3] = 1.75
        starting_values_Dec2018[7] = 0.45
        # Dec 2018
        t_Dec2018 = np.linspace(165, 165.9, 3)
        Dec2018_ABC = solve_ivp(ecoNetwork, (165,165.9), starting_values_Dec2018,  t_eval = t_Dec2018, args=(A_2, r_2), method = 'RK23')
        starting_values_Jan2019 = Dec2018_ABC.y[0:10, 2:3].flatten()
        # 1 red deer death, 19 fallow deaths, 5 cows sold and 1 cow moved on-site
        starting_values_Jan2019[1] = starting_values_Jan2019[1] - 0.45
        starting_values_Jan2019[5] = starting_values_Jan2019[5] - 0.077
        starting_values_Jan2019[3] = 1.68
        # Jan & Feb 2019
        t_Jan2019 = np.linspace(166, 167.9, 6)
        Jan2019_ABC = solve_ivp(ecoNetwork, (166,167.9), starting_values_Jan2019,  t_eval = t_Jan2019, args=(A_2, r_2), method = 'RK23')
        starting_values_March2019 = Jan2019_ABC.y[0:10, 5:6].flatten()
        # 1 cow sold
        starting_values_March2019[3] = 1.64

        ## 2019 ##
        # March 2019
        t_March2019 = np.linspace(168, 168.9, 3)
        March2019_ABC = solve_ivp(ecoNetwork, (168,168.9), starting_values_March2019,  t_eval = t_March2019, args=(A_2, r_2), method = 'RK23')
        starting_values_April2019 = March2019_ABC.y[0:10, 2:3].flatten()
        # 7 red deer and 7 fallow deer culled
        starting_values_April2019[3] = starting_values_April2019[3] - 0.17
        starting_values_April2019[5] = starting_values_April2019[5] - 0.54
        # April 2019
        t_April2019 = np.linspace(169, 169.9, 3)
        April2019_ABC = solve_ivp(ecoNetwork, (169,169.9), starting_values_April2019,  t_eval = t_April2019, args=(A_2, r_2), method = 'RK23')
        starting_values_May2019 = April2019_ABC.y[0:10, 2:3].flatten()
        # 1 pig sold
        starting_values_May2019[7] = 0.4
        # May & June 2019
        t_May2019 = np.linspace(170, 171.9, 6)
        May2019_ABC = solve_ivp(ecoNetwork, (170,171.9), starting_values_May2019,  t_eval = t_May2019, args=(A_2, r_2), method = 'RK23')
        starting_values_July2019 = May2019_ABC.y[0:10, 5:6].flatten()
        # 28 longhorns moved off-sites
        starting_values_July2019[3] = 1.68
        # July 2019
        t_July2019 = np.linspace(172, 172.9, 3)
        July2019_ABC = solve_ivp(ecoNetwork, (172,172.9), starting_values_July2019,  t_eval = t_July2019, args=(A_2, r_2), method = 'RK23')
        starting_values_Aug2019 = July2019_ABC.y[0:10, 2:3].flatten()
        # 26 pigs sold, 3 longhorns sold, 5 longhorns moved off-site
        starting_values_Aug2019[3] = 1.72
        starting_values_Aug2019[7] = 0.45
        # Aug & Sept 2019
        t_Aug2019 = np.linspace(173, 174.9, 6)
        Aug2019_ABC = solve_ivp(ecoNetwork, (173,174.9), starting_values_Aug2019,  t_eval = t_Aug2019, args=(A_2, r_2), method = 'RK23')
        starting_values_Oct2019 = Aug2019_ABC.y[0:10, 5:6].flatten()
        # 15 fallow deaths, 19 cows sold and 4 moved off-site, 25 moved on-site
        starting_values_Oct2019[1] = starting_values_Oct2019[1] - 0.36
        starting_values_Oct2019[3] = 1.75
        # Oct 2019
        t_Oct2019 = np.linspace(175, 175.9, 3)
        Oct2019_ABC = solve_ivp(ecoNetwork, (175,175.9), starting_values_Oct2019,  t_eval = t_Oct2019, args=(A_2, r_2), method = 'RK23')
        starting_values_Nov2019 = Oct2019_ABC.y[0:10, 2:3].flatten()
        # 5 cows moved off-site
        starting_values_Nov2019[3] = 1.66
        # Nov 2019
        t_Nov2019 = np.linspace(176, 176.9, 3)
        Nov2019_ABC = solve_ivp(ecoNetwork, (176,176.9), starting_values_Nov2019,  t_eval = t_Nov2019, args=(A_2, r_2), method = 'RK23')
        starting_values_Dec2019 = Nov2019_ABC.y[0:10, 2:3].flatten()
        # 1 cow death, 7 fallow deaths, 3 red deer deaths
        starting_values_Dec2019[1] = starting_values_Dec2019[1] - 0.17
        starting_values_Dec2019[3] = 1.64
        starting_values_Dec2019[5] = starting_values_Dec2019[5] - 0.23
        # Dec 2019
        t_Dec2019 = np.linspace(177, 177.9, 3)
        Dec2019_ABC = solve_ivp(ecoNetwork, (177,177.9), starting_values_Dec2019,  t_eval = t_Dec2019, args=(A_2, r_2), method = 'RK23')
        starting_values_Jan2020 = Dec2019_ABC.y[0:10, 2:3].flatten()
        # 7 cow sales, 1 pig added, 4 red deer deaths, 12 fallow deer deaths
        starting_values_Jan2020[1] = starting_values_Jan2020[1] - 0.29
        starting_values_Jan2020[5] = starting_values_Jan2020[5] - 0.31
        starting_values_Jan2020[3] = 1.51
        starting_values_Jan2020[7] = 0.5
        # Jan 2020
        t_Jan2020 = np.linspace(178, 178.9, 3)
        Jan2020_ABC = solve_ivp(ecoNetwork, (178,178.9), starting_values_Jan2020,  t_eval = t_Jan2020, args=(A, r), method = 'RK23')
        starting_values_Feb2020 = Jan2020_ABC.y[0:10, 2:3].flatten()
        # 24 fallow deer deaths
        starting_values_Feb2020[1] = starting_values_Feb2020[1] - 0.57
        # Feb 2020
        t_Feb2020 = np.linspace(179, 179.9, 3)
        Feb2020_ABC = solve_ivp(ecoNetwork, (179,179.9), starting_values_Feb2020,  t_eval = t_Feb2020, args=(A_2, r_2), method = 'RK23')
        starting_values_March2020 = Feb2020_ABC.y[0:10, 2:3].flatten()
        # 2 pigs sold, 12 fallow deers killed, 2 reds killed, 1 cow moved off-site
        starting_values_March2020[1] = starting_values_March2020[1] - 0.29
        starting_values_March2020[3] = 1.49
        starting_values_March2020[5] = starting_values_March2020[5] - 0.15
        starting_values_March2020[7] = 0.4

        ## 2020 ##
        # March 2020
        t_March2020 = np.linspace(180, 180.9, 3)
        March2020_ABC = solve_ivp(ecoNetwork, (180,180.9), starting_values_March2020,  t_eval = t_March2020, args=(A_2, r_2), method = 'RK23')
        starting_values_April2020 = March2020_ABC.y[0:10, 2:3].flatten()
        # 1 pig death, 15 ponies added, 1 cow sold, 3 cows moved on-site
        starting_values_April2020[0] = 0.65
        starting_values_April2020[3] = 1.53
        starting_values_April2020[7] = 0.35
        # April - Feb 2020 (most recent data)
        t_April2020 = np.linspace(181, 192, 6)
        April2020_ABC = solve_ivp(ecoNetwork, (181, 192), starting_values_April2020,  t_eval = t_April2020, args=(A_2, r_2), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, March2015_ABC.y, May2015_ABC.y, June2015_ABC.y,  July2015_ABC.y, Sept2015_ABC.y, Oct2015_ABC.y, Nov2015_ABC.y, Dec2015_ABC.y, Jan2016_ABC.y, Feb2016_ABC.y, March2016_ABC.y, April2016_ABC.y, May2016_ABC.y, June2016_ABC.y, July2016_ABC.y, Aug2016_ABC.y, Sept2016_ABC.y, Oct2016_ABC.y, Dec2016_ABC.y, Jan2017_ABC.y, Feb2017_ABC.y, March2017_ABC.y, May2017_ABC.y, July2017_ABC.y, Sept2017_ABC.y, Oct2017_ABC.y, Nov2017_ABC.y, Dec2017_ABC.y, Jan2018_ABC.y, Feb2018_ABC.y, March2018_ABC.y, May2018_ABC.y, July2018_ABC.y, Aug2018_ABC.y, Sept2018_ABC.y, Oct2018_ABC.y, Nov2018_ABC.y, Dec2018_ABC.y, Jan2019_ABC.y, March2019_ABC.y, April2019_ABC.y, May2019_ABC.y, July2019_ABC.y, Aug2019_ABC.y, Oct2019_ABC.y, Nov2019_ABC.y, Dec2019_ABC.y, Jan2020_ABC.y, Feb2020_ABC.y, March2020_ABC.y, April2020_ABC.y))
        combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, March2015_ABC.t, May2015_ABC.t, June2015_ABC.t,  July2015_ABC.t, Sept2015_ABC.t, Oct2015_ABC.t, Nov2015_ABC.t, Dec2015_ABC.t, Jan2016_ABC.t, Feb2016_ABC.t, March2016_ABC.t, April2016_ABC.t, May2016_ABC.t, June2016_ABC.t, July2016_ABC.t, Aug2016_ABC.t, Sept2016_ABC.t, Oct2016_ABC.t, Dec2016_ABC.t, Jan2017_ABC.t, Feb2017_ABC.t, March2017_ABC.t, May2017_ABC.t, July2017_ABC.t, Sept2017_ABC.t, Oct2017_ABC.t, Nov2017_ABC.t, Dec2017_ABC.t, Jan2018_ABC.t, Feb2018_ABC.t, March2018_ABC.t, May2018_ABC.t, July2018_ABC.t, Aug2018_ABC.t, Sept2018_ABC.t, Oct2018_ABC.t, Nov2018_ABC.t, Dec2018_ABC.t, Jan2019_ABC.t, March2019_ABC.t, April2019_ABC.t, May2019_ABC.t, July2019_ABC.t, Aug2019_ABC.t, Oct2019_ABC.t, Nov2019_ABC.t, Dec2019_ABC.t, Jan2020_ABC.t, Feb2020_ABC.t, March2020_ABC.t, April2020_ABC.t))
        # append to dataframe
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)   
        all_times_2 = np.append(all_times_2, combined_times)
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 405).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    final_runs_2['time'] = all_times_2
    # append all the parameters to a dataframe
    all_parameters_2 = pd.concat(all_parameters_2)
    # add ID to the dataframe & parameters
    all_parameters_2['ID'] = ([(x+1) for x in range(len(accepted_simulations)) for _ in range(len(parameters_used_2))])
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,405)
    return final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = runODE_2()
    
    with pd.option_context('display.max_columns',None):
        print(final_runs_2)

    # filter the runs
    accepted_simulations_2020 = final_runs_2[
    # April 2015: 8 cow births, 5 pig births (10% above/below value), ponies the same
    ((final_runs_2['time']== 121.9) & (final_runs_2['longhornCattle'] >= 1.95)) &
    ((final_runs_2['time'] == 121.9) & (final_runs_2['tamworthPig'] <= 1.27) & (final_runs_2['tamworthPig'] >= 1.04)) &
    ((final_runs_2['time'] == 121.9) & (final_runs_2['exmoorPony'] <= 0.47) & (final_runs_2['exmoorPony'] >= 0.39)) &
    # May 2015, 14 cow births
    ((final_runs_2['time'] == 122.9) & (final_runs_2['longhornCattle'] <= 2.67) & (final_runs_2['longhornCattle'] >= 2.19)) &
    # June 2015: 5 cow births
    ((final_runs_2['time'] == 123.9) & (final_runs_2['longhornCattle'] <= 2.78) & (final_runs_2['longhornCattle'] >= 2.28)) &
    # Feb 2016, still 10 ponies
    ((final_runs_2['time'] == 131.9) & (final_runs_2['exmoorPony'] <= 0.47) & (final_runs_2['exmoorPony'] >= 0.39)) &
    # March 2016: 140 fallow deer, 26 red deer, 2 pig births
    ((final_runs_2['time'] == 132.9) & (final_runs_2['fallowDeer'] <= 3.63) & (final_runs_2['fallowDeer'] >= 2.97)) &
    ((final_runs_2['time'] == 132.9) & (final_runs_2['redDeer'] <= 2.2) & (final_runs_2['redDeer'] >= 1.8)) &
    ((final_runs_2['time'] == 132.9) & (final_runs_2['tamworthPig'] <= 0.55) & (final_runs_2['tamworthPig'] >= 0.45)) &
    # April 2016: 16 cattle births
    ((final_runs_2['time'] == 133.9) & (final_runs_2['longhornCattle'] <= 2.11) & (final_runs_2['longhornCattle'] >= 1.73)) &
    # May 2016: 8 pig births, 7 cow births
    ((final_runs_2['time'] == 134.9) & (final_runs_2['longhornCattle'] <= 2.29) & (final_runs_2['longhornCattle'] >= 1.87)) &
    ((final_runs_2['time'] == 134.9) & (final_runs_2['tamworthPig'] <= 0.94) & (final_runs_2['tamworthPig'] >= 0.77)) &
    # June 2016: 7 cow births
    ((final_runs_2['time'] == 135.9) & (final_runs_2['longhornCattle'] <= 2.39) & (final_runs_2['longhornCattle'] >= 1.95)) &
    # Sept 2016: 19 cow births
    ((final_runs_2['time'] == 138.9) & (final_runs_2['longhornCattle'] <= 2.41) & (final_runs_2['longhornCattle'] >= 1.97)) &
    # Feb 2017: still 11 ponies
    ((final_runs_2['time'] == 143.9) & (final_runs_2['exmoorPony'] <= 0.54) & (final_runs_2['exmoorPony'] >= 0.44)) &
    # March 2017: 165 fallow deer
    ((final_runs_2['time'] == 144.9) & (final_runs_2['fallowDeer'] <= 4.32) & (final_runs_2['fallowDeer'] >= 3.54)) &
    # April 2017: 15 pig births, 18 cow births
    ((final_runs_2['time'] == 145.9) & (final_runs_2['longhornCattle'] <= 2.01) & (final_runs_2['longhornCattle'] >= 1.65)) &
    ((final_runs_2['time'] == 145.9) & (final_runs_2['tamworthPig'] <= 1.21) & (final_runs_2['tamworthPig'] >= 0.99)) &
    # May 2017: 9 cow births
    ((final_runs_2['time'] == 146.9) & (final_runs_2['longhornCattle'] <= 2.27) & (final_runs_2['longhornCattle'] >= 1.85)) &
    # June 2017: 6 cow births
    ((final_runs_2['time'] == 147.9) & (final_runs_2['longhornCattle'] <= 2.39) & (final_runs_2['longhornCattle'] >= 1.95)) &
    # Jan 2018: 2 pig births, ponies still the same
    ((final_runs_2['time'] == 154.9) & (final_runs_2['tamworthPig'] <= 1.1) & (final_runs_2['tamworthPig'] >= 0.9) & (final_runs_2['exmoorPony'] >= 0.47) & (final_runs_2['exmoorPony'] >= 0.39)) &
    # March 2018: 24 red deer, 251 fallow deer, 5 pig births
    ((final_runs_2['time'] == 156.9) & (final_runs_2['redDeer'] <= 2.04) & (final_runs_2['redDeer'] >= 1.67)) &
    ((final_runs_2['time'] == 156.9) & (final_runs_2['fallowDeer'] <= 6.58) & (final_runs_2['fallowDeer'] >= 5.38)) &
    ((final_runs_2['time'] == 156.9) & (final_runs_2['tamworthPig'] <= 0.88) & (final_runs_2['tamworthPig'] >= 0.72)) &
    # April 2018: 12 cow births
    ((final_runs_2['time'] == 157.9) & (final_runs_2['longhornCattle'] <= 2.08) & (final_runs_2['longhornCattle'] >= 1.70)) &
    # May 2018: 16 cow births, 7 pig births
    ((final_runs_2['time'] == 158.9) & (final_runs_2['longhornCattle'] <= 2.43) & (final_runs_2['longhornCattle'] >= 1.99)) &
    ((final_runs_2['time'] == 158.9) & (final_runs_2['tamworthPig'] <= 1.27) & (final_runs_2['tamworthPig'] >= 1.04)) &
    # June 2018: 6 cow births
    ((final_runs_2['time']  == 159.9) & (final_runs_2['longhornCattle'] <= 2.55) & (final_runs_2['longhornCattle'] >= 2.09)) &
    # March 2019: 278 fallow deer, 37 red deer
    ((final_runs_2['time'] == 168.9) & (final_runs_2['fallowDeer'] <= 7.47) & (final_runs_2['fallowDeer'] >= 6.1)) &
    ((final_runs_2['time'] == 168.9) & (final_runs_2['redDeer'] <= 3.14) & (final_runs_2['redDeer'] >= 2.57)) &
    # April 2019: 14 longhorn births
    ((final_runs_2['time'] == 169.9) & (final_runs_2['longhornCattle'] <= 2.10) & (final_runs_2['longhornCattle'] >= 1.72)) &
    # May 2019: 9 longhorn births
    ((final_runs_2['time'] == 170.9) & (final_runs_2['longhornCattle'] <= 2.31) & (final_runs_2['longhornCattle'] >= 1.89)) &
    # June 2019: 7 longhorn births
    ((final_runs_2['time'] == 171.9) & (final_runs_2['longhornCattle'] <= 2.43) & (final_runs_2['longhornCattle'] >= 1.99)) &
    # July 2019: 28 pig births (pre-26 being sold)
    ((final_runs_2['time'] == 172.9) & (final_runs_2['tamworthPig'] <= 1.98) & (final_runs_2['tamworthPig'] >= 1.62)) &
    # 2020  - no filtering for fallow or red deer bc we don't know what they've grown to yet (next survey March 2021)
    ((final_runs_2['time'] == 192) & (final_runs_2['exmoorPony'] <= 0.72) & (final_runs_2['exmoorPony'] >= 0.59)) &
    ((final_runs_2['time'] == 192) & (final_runs_2['longhornCattle'] <= 1.68) & (final_runs_2['longhornCattle'] >= 1.38)) &
    ((final_runs_2['time'] == 192) & (final_runs_2['tamworthPigs'] <= 1.05) & (final_runs_2['tamworthPigs'] >= 0.86)) &
    ((final_runs_2['time'] == 192) & (final_runs_2['roeDeer'] <= 6.7) & (final_runs_2['roeDeer'] >= 1.7)) &
    ((final_runs_2['time'] == 192) & (final_runs_2['grasslandParkland'] <= 0.79) & (final_runs_2['grasslandParkland'] >= 0.67)) &
    ((final_runs_2['time'] == 192) & (final_runs_2['woodland'] <= 1.73) & (final_runs_2['woodland'] >= 0.98)) &
    ((final_runs_2['time'] == 192) & (final_runs_2['thornyScrub'] <= 35.1) & (final_runs_2['thornyScrub'] >= 22.5)) & 
    ((final_runs_2['time'] == 192) & (final_runs_2['organicCarbon'] <= 2.2) & (final_runs_2['organicCarbon'] >= 1.7))
    ]
    
    # now choose just the final years (these will become the starting conditions in the next model)
    accepted_simulations_2020 = accepted_simulations_2020.loc[accepted_simulations_2020['time'] == 192]

    with pd.option_context('display.max_columns',None):
        print(accepted_simulations_2020, accepted_simulations_2020.shape)

    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2020 = all_parameters_2[all_parameters_2['ID'].isin(accepted_simulations_2020['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2020['ID']), 'Accepted', 'Rejected')
    return accepted_simulations_2020, accepted_parameters_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# # # # ---------------------- ODE #3: projecting 10 years (2018-2028) -------------------------

def generateParameters3():
    accepted_simulations_2020, accepted_parameters_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun = filterRuns_2()
    # # select growth rates 
    growthRates_3 = accepted_parameters_2020.loc[accepted_parameters_2020['growth'].notnull(), ['growth']]
    growthRates_3 = pd.DataFrame(growthRates_3.values.reshape(len(accepted_simulations_2020), len(species)), columns = species)
    r_thirdRun = growthRates_3.to_numpy()
    # make the final runs of ODE #2 the initial conditions
    accepted_simulations_2 = accepted_simulations_2020.drop(['ID', 'time'], axis=1)
    accepted_parameters_2020.loc[accepted_parameters_2020['X0'].notnull(), ['X0']] = accepted_simulations_2.values.flatten()
    # select X0 
    X0_3 = accepted_parameters_2020.loc[accepted_parameters_2020['X0'].notnull(), ['X0']]
    X0_3 = pd.DataFrame(X0_3.values.reshape(len(accepted_simulations_2), len(species)), columns = species)
    # # select interaction matrices part of the dataframes 
    interaction_strength_3 = accepted_parameters_2020.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_3 = interaction_strength_3.dropna()
    A_thirdRun = interaction_strength_3.to_numpy()
    # check histograms for growth
    growth_filtered = growthRates_3[["grasslandParkland","thornyScrub","woodland"]]
    fig, axes = plt.subplots(len(growth_filtered.columns)//3,3, figsize=(25, 10))
    for col, axis in zip(growth_filtered.columns, axes):
        growth_filtered.hist(column = col, ax = axis, bins = 25)
    # check correlation matrix
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
    # combine dataframes
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
    # plt.savefig('corrMatrix_5mil_practice.png')
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
    t = np.linspace(192, 193, 36)

    # loop through each row of accepted parameters
    for X0_4, r_4, A_4 in zip(X0_thirdRun,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # starting values for 2021
        X0_4[0] = 0.65
        X0_4[1] =  5.88
        X0_4[3] =  1.53
        X0_4[5] =  2.69
        X0_4[7] =  0.95
        # concantenate the parameters
        X0_growth_3 = pd.concat([pd.DataFrame(X0_4), pd.DataFrame(r_4)], axis = 1)
        X0_growth_3.columns = ['X0','growth']
        parameters_used_3 = pd.concat([X0_growth_3, pd.DataFrame(A_4, index = species, columns = species)])
        # 2021
        ABC_2021 = solve_ivp(ecoNetwork, (192, 193), X0_4,  t_eval = t, args=(A_4, r_4), method = 'RK23')        
        # ten percent above/below 2021 values
        starting_2022 = ABC_2021.y[0:10, 35:36].flatten()
        starting_2022[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2022[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2022[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2022[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2022[7] =  np.random.uniform(low=0.86,high=1.0)
        t_1 = np.linspace(193, 194, 36)
        # 2022
        ABC_2022 = solve_ivp(ecoNetwork, (193, 194), starting_2022,  t_eval = t_1, args=(A_4, r_4), method = 'RK23')
        starting_2023 = ABC_2022.y[0:10, 35:36].flatten()
        starting_2023[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2023[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2023[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2023[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2023[7] =  np.random.uniform(low=0.86,high=1.0)
        t_2 = np.linspace(194, 195, 36)
        # 2023
        ABC_2023 = solve_ivp(ecoNetwork, (194, 195), starting_2023,  t_eval = t_2, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2024 = ABC_2023.y[0:10, 35:36].flatten()
        starting_2024[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2024[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2024[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2024[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2024[7] =  np.random.uniform(low=0.86,high=1.0)
        t_3 = np.linspace(195, 196, 36)
        # run the model for 2024
        ABC_2024 = solve_ivp(ecoNetwork, (195, 196), starting_2024,  t_eval = t_3, args=(A_4, r_4), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2025 = ABC_2024.y[0:10, 35:36].flatten()
        starting_2025[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2025[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2025[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2025[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2025[7] =  np.random.uniform(low=0.86,high=1.0)
        t_4 = np.linspace(196, 197, 36)
        # run the model for 2025
        ABC_2025 = solve_ivp(ecoNetwork, (196, 197), starting_2025,  t_eval = t_4, args=(A_4, r_4), method = 'RK23')
        starting_2026 = ABC_2025.y[0:10, 35:36].flatten()
        starting_2026[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2026[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2026[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2026[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2026[7] =  np.random.uniform(low=0.86,high=1.0)
        t_5 = np.linspace(197, 198, 36)
        # 2026
        ABC_2026 = solve_ivp(ecoNetwork, (197, 198), starting_2026,  t_eval = t_5, args=(A_4, r_4), method = 'RK23')
        starting_2027 = ABC_2026.y[0:10, 35:36].flatten()
        starting_2027[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2027[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2027[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2027[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2027[7] =  np.random.uniform(low=0.86,high=1.0)
        t_6 = np.linspace(198, 199, 36)
        # 2027
        ABC_2027 = solve_ivp(ecoNetwork, (198, 199), starting_2027,  t_eval = t_6, args=(A_4, r_4), method = 'RK23')
        starting_2028 = ABC_2027.y[0:10, 35:36].flatten()
        starting_2028[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2028[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2028[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2028[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2028[7] =  np.random.uniform(low=0.86,high=1.0)
        t_7 = np.linspace(199, 200, 36)
        # 2028
        ABC_2028 = solve_ivp(ecoNetwork, (199, 200), starting_2028,  t_eval = t_7, args=(A_4, r_4), method = 'RK23')
        starting_2029 = ABC_2028.y[0:10, 35:36].flatten()
        starting_2029[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2029[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2029[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2029[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2029[7] =  np.random.uniform(low=0.86,high=1.0)
        t_8 = np.linspace(200, 201, 36)
        # 2029
        ABC_2029 = solve_ivp(ecoNetwork, (200, 201), starting_2028,  t_eval = t_8, args=(A_4, r_4), method = 'RK23')
        starting_2030 = ABC_2029.y[0:10, 35:36].flatten()
        starting_2030[0] = np.random.uniform(low=0.59,high=0.72)
        starting_2030[1] =  np.random.uniform(low=5.3,high=6.5)
        starting_2030[3] =  np.random.uniform(low=1.38,high=1.7)
        starting_2030[5] =  np.random.uniform(low=2.42,high=3.0)
        starting_2030[7] =  np.random.uniform(low=0.86,high=1.0)
        t_9 = np.linspace(201, 202, 36)
        # 2030
        ABC_2030 = solve_ivp(ecoNetwork, (201, 202), starting_2030,  t_eval = t_9, args=(A_4, r_4), method = 'RK23')
        # concatenate & append all the runs
        combined_runs_2 = np.hstack((ABC_2021.y, ABC_2022.y, ABC_2023.y, ABC_2024.y, ABC_2025.y, ABC_2026.y, ABC_2027.y, ABC_2028.y, ABC_2029.y, ABC_2030.y))
        combined_times_2 = np.hstack((ABC_2021.t, ABC_2022.t, ABC_2023.t, ABC_2024.t, ABC_2025.t, ABC_2026.t, ABC_2027.t, ABC_2028.t, ABC_2029.t, ABC_2030.t))
        all_runs_3 = np.append(all_runs_3, combined_runs_2)
        # append all the parameters
        all_parameters_3.append(parameters_used_3)   
        # append the times
        all_times_3 = np.append(all_times_3, combined_times_2)
    # check the final runs
    final_runs_3 = (np.vstack(np.hsplit(all_runs_3.reshape(len(species)*len(accepted_simulations_2020), 360).transpose(),len(accepted_simulations_2020))))
    final_runs_3 = pd.DataFrame(data=final_runs_3, columns=species)
    # append all the parameters to a dataframe
    all_parameters_3 = pd.concat(all_parameters_3)
    # add ID to the dataframe & parameters
    all_parameters_3['ID'] = ([(x+1) for x in range(len(accepted_simulations_2020)) for _ in range(len(parameters_used_3))])
    IDs = np.arange(1,1 + len(accepted_simulations_2020))
    final_runs_3['ID'] = np.repeat(IDs,360)
    final_runs_3['time'] = all_times_3
    final_runs_3['accepted?'] = np.repeat('Accepted', len(final_runs_3))


    # set reintroduced species to zero to see what would've happened without reintroductions
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
    t_noReintro = np.linspace(4, 21, 792)
    for X0_noReintro, r_4, A_4 in zip(X0_3_noReintro,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        noReintro_ABC = solve_ivp(ecoNetwork, (4, 21), X0_noReintro,  t_eval = t_noReintro, args=(A_4, r_4), method = 'RK23') 
        all_runs_2 = np.append(all_runs_2, noReintro_ABC.y)
        all_times_2 = np.append(all_times_2, noReintro_ABC.t)
    no_reintro = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations_2020), 792).transpose(),len(accepted_simulations_2020))))
    no_reintro = pd.DataFrame(data=no_reintro, columns=species)
    IDs_2 = np.arange(1,1 + len(accepted_simulations_2020))
    no_reintro['ID'] = np.repeat(IDs_2,792)
    no_reintro['time'] = all_times_2
    # concantenate this will the accepted runs from years 1-5
    filtered_FinalRuns = final_runs.loc[(final_runs['accepted?'] == "Accepted") ]
    no_reintro = pd.concat([filtered_FinalRuns, no_reintro])
    no_reintro['accepted?'] = "noReintro"


    # # what if there had been no culling?
    # all_runs_noCulls = []
    # all_times_noCulls = []
    # t_noCulls = np.linspace(4, 40, 105)
    # X0_noCull = X0_secondRun
    # # loop through each row of accepted parameters
    # for X0_noCulling, r_5, A_5 in zip(X0_noCull,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
    #     # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
    #     noCull_ABC = solve_ivp(ecoNetwork, (4, 40), X0_noCulling,  t_eval = t_noCulls, args=(A_5, r_5), method = 'RK23') 
    #     all_runs_noCulls = np.append(all_runs_noCulls, noCull_ABC.y)
    #     all_times_noCulls = np.append(all_times_noCulls, noCull_ABC.t)
    # no_Cull = (np.vstack(np.hsplit(all_runs_noCulls.reshape(len(species)*len(accepted_simulations_2020), 105).transpose(),len(accepted_simulations_2020))))
    # no_Cull = pd.DataFrame(data=no_Cull, columns=species)
    # IDs_3 = np.arange(1,1 + len(accepted_simulations_2020))
    # no_Cull['ID'] = np.repeat(IDs_3,105)
    # no_Cull['time'] = all_times_noCulls
    # no_Cull = pd.concat([filtered_FinalRuns, no_Cull])
    # no_Cull['accepted?'] = "noCulls"



    # # how many herbivores are needed to change scrubland to grassland? 
    # all_runs_howManyHerbs = []
    # all_times_howManyHerbs = []
    # t_howManyHerbs = np.linspace(4, 39, 105)
    # X0_howManyHerbs = [0.1, 1, 1, 1, 1, 1, 0.1]
    # # loop through each row of accepted parameters
    # for X0_6, r_6, A_6 in zip(X0_howManyHerbs,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
    #     # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
    #     howManyHerbs_ABC = solve_ivp(ecoNetwork, (4, 39), X0_6,  t_eval = t_howManyHerbs, args=(A_6, r_6), method = 'RK23') 
    #     all_runs_howManyHerbs = np.append(all_runs_howManyHerbs, howManyHerbs_ABC.y)
    #     all_times_howManyHerbs = np.append(all_times_howManyHerbs, howManyHerbs_ABC.t)
    # howManyHerbs = (np.vstack(np.hsplit(all_runs_howManyHerbs.reshape(len(species)*len(accepted_simulations_2020), 105).transpose(),len(accepted_simulations_2020))))
    # howManyHerbs = pd.DataFrame(data=howManyHerbs, columns=species)
    # IDs_6 = np.arange(1,1 + len(accepted_simulations_2020))
    # howManyHerbs['ID'] = np.repeat(IDs_6,105)
    # howManyHerbs['time'] = all_times_howManyHerbs
    # howManyHerbs = pd.concat([filtered_FinalRuns, howManyHerbs])
    # howManyHerbs['accepted?'] = "howManyHerbs"


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

    # return final_runs_3, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, no_Cull




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    final_runs_3, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro, no_Cull = runODE_3()
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
    species_firstRun = np.tile(species, 144*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 405*len(accepted_simulations))
    species_thirdRun = np.tile(species, 360*len(accepted_simulations_2020))
    species_noReintro = np.tile(species, (792*len(accepted_simulations_2020)) + (144*len(accepted_simulations)))
    species_noCull = np.tile(species, (792*len(accepted_simulations_2020)) + (144*len(accepted_simulations)))
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
    g.axes[2].vlines(x=15,ymin=0.67,ymax=0.79, color='r')
    g.axes[4].vlines(x=15,ymin=1.7,ymax=2.2, color='r')
    g.axes[6].vlines(x=15,ymin=1.7,ymax=6.7, color='r')
    g.axes[8].vlines(x=15,ymin=22.5,ymax=35.1, color='r')
    g.axes[9].vlines(x=15,ymin=0.98,ymax=1.7, color='r')
    # make sure they all start from 0 
    g.axes[4].set(ylim =(0,None))
    g.axes[5].set(ylim =(0,None))
    g.axes[8].set(ylim =(0,None))

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
    n.axes[2].vlines(x=15,ymin=0.67,ymax=0.79, color='r')
    n.axes[4].vlines(x=15,ymin=1.7,ymax=2.2, color='r')
    n.axes[6].vlines(x=15,ymin=1.7,ymax=6.7, color='r')
    n.axes[8].vlines(x=15,ymin=22.5,ymax=35.1, color='r')
    n.axes[9].vlines(x=15,ymin=0.98,ymax=1.7, color='r')
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
# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# Parameter set 2 (leaning on the methods)

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
                # grassland parkland
                [0, -0.00041, -0.00039, -0.78, -0.00065, -0.00056, 0, -0.00014, -0.0036, -0.01, -0.15],
                # longhorn cattle  
                [0, 0, 0, 0.34, -0.31, 0, 0, 0, 0, 0.0067, 0.37],
                # organic carbon
                [0, 0.0074, 0.0046, 0.05, 0.0069, -0.082, 0.0053, 0.004, 0.0055, 0.0012, 0.048],  
                # red deer  
                [0, 0, 0, 0.34, 0, 0, -0.3, 0, 0, 0.016, 0.35],
                # roe deer 
                [0, 0, 0, 11.9, 0, 0, 0, -14.5, 0, 0.57, 11.25],
                # tamworth pig 
                [0, 0, 0, 4, 0, 0, 0, 0, -14.4, 0.34, 3.1],  
                # thorny scrub
                [0, -0.034, -0.031, 0, -0.044, 0, -0.031, -0.011, -0.028, -0.0072, -0.13],
                # woodland
                [0, -0.0057, -0.0054, 0, -0.0084, 0, -0.0062, -0.0013, -0.0026, 0.00035, -0.009]
    ]

    # generate random uniform numbers
    variation = np.random.uniform(low = 0.95, high=1.05, size = (len(species),len((species))))
    interaction_matrix = interaction_matrix * variation
    # return array
    return interaction_matrix


def generateGrowth():
    # PARAMETER SET 2: leaning on the methods (some consumers have incorrect yearly data)
    growthRates =  [0, 0, 0, 0.98, 0, 0, 0, 0, 0, 0.78, 0.066] 
    # multiply by a range
    variation = np.random.uniform(low = 0.95, high=1.05, size = (len(species),))
    growth = growthRates * variation
    return growth
    

def generateX0():
    # scale everything to abundance of one (except species to be reintroduced)
    X0 = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    return X0

# # # # --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -------


def runODE_1():
    # Define time points: first 5 years (2005, 2006, 2007, 2008), take 3 points per month
    t = np.linspace(2005, 2008.95, 8)
    all_runs = []
    all_times = []
    all_parameters = []
    NUMBER_OF_SIMULATIONS = 0
    for _ in range(totalSimulations):
        A = generateInteractionMatrix()
        r = generateGrowth()
        X0 = generateX0()
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
    (final_runs_2['longhornCattle'] <= 2.7) & (final_runs_2['longhornCattle'] >= 2.37) & # leaning on data
    (final_runs_2['tamworthPig'] <= 1.45) & (final_runs_2['tamworthPig'] >= 0.85)]
    filtered_2015 = final_runs_2[final_runs_2['ID'].isin(accepted_simulations_2015['ID'])]
    print("number passed 2015 filters:", filtered_2015.shape[0]/24)
    
    # filter 2016 values : ponies = the same, fallow deer = 3.9 but 25 were culled; longhorn got to maximum 2.03; pig got to maximum 0.85
    accepted_simulations_2016 = filtered_2015[(filtered_2015['time'] == 2016.95) & 
    (filtered_2015['exmoorPony'] <= 0.52) & (filtered_2015['exmoorPony'] >= 0.43) & 
    (filtered_2015['fallowDeer'] <= 5.12) & (filtered_2015['fallowDeer'] >= 3.93) & # leaning on data
    (filtered_2015['longhornCattle'] <= 2.34) & (filtered_2015['longhornCattle'] >= 2.04) & 
    (filtered_2015['tamworthPig'] <= 1.25) & (filtered_2015['tamworthPig'] >= 0.65)]
    filtered_2016 = filtered_2015[filtered_2015['ID'].isin(accepted_simulations_2016['ID'])]
    print("number passed 2016 filters:", filtered_2016.shape[0]/24)
    # filter 2017 values : ponies = the same; fallow = 7.34 + 1.36 culled (this was maybe supplemented so no filter), same with red; cows got to max 2.06; red deer got to 1.85 + 2 culled; pig got to 1.1
    accepted_simulations_2017 = filtered_2016[(filtered_2016['time'] == 2017.95) & 
    (filtered_2016['exmoorPony'] <= 0.48) & (filtered_2016['exmoorPony'] >= 0.39) & 
    (filtered_2016['longhornCattle'] <= 2.3) & (filtered_2016['longhornCattle'] >= 1.96) & 
    (filtered_2016['redDeer'] <= 2.31) & (filtered_2016['redDeer'] >= 1.8) & # leaning on data
    (filtered_2016['tamworthPig'] <= 1.75) & (filtered_2016['tamworthPig'] >= 1.15)]
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

    with open("ps2_stable_numberAcceptedSims.txt", "w") as text_file:
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

    return r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun




# # # # # ------ SOLVE ODE #3: Projecting forwards 10 years (2020-2030) -------

def runODE_3():
    r_thirdRun, X0_3, A_thirdRun, accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, X0_secondRun  = generateParameters3()

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
    t_noReintro = np.linspace(2009, 3009, 100)
    for X0_noReintro, r_noReintro, A_noReintro in zip(X0_3_noReintro,r_thirdRun, np.array_split(A_thirdRun,len(accepted_simulations_2020))):
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        noReintro_ABC = solve_ivp(ecoNetwork, (2009, 3009), X0_noReintro,  t_eval = t_noReintro, args=(A_noReintro, r_noReintro), method = 'RK23') 
        all_runs_2 = np.append(all_runs_2, noReintro_ABC.y)
        all_times_2 = np.append(all_times_2, noReintro_ABC.t)
    no_reintro = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations_2020), 100).transpose(),len(accepted_simulations_2020))))
    no_reintro = pd.DataFrame(data=no_reintro, columns=species)
    IDs_2 = np.arange(1,1 + len(accepted_simulations_2020))
    no_reintro['ID'] = np.repeat(IDs_2,100)
    no_reintro['time'] = all_times_2
    filtered_FinalRuns = final_runs.loc[(final_runs['accepted?'] == "Accepted") ]
    # concantenate this will the accepted runs from years 1-5
    no_reintro = pd.concat([filtered_FinalRuns, no_reintro])
    no_reintro['accepted?'] = "noReintro"

    return accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro




# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

def plotting():
    accepted_simulations_2020, final_runs_2, accepted_simulations, final_runs, NUMBER_OF_SIMULATIONS, no_reintro = runODE_3()
    # extract accepted nodes from all dataframes
    accepted_shape1 = np.repeat(final_runs['accepted?'], len(species))
    accepted_shape4 = np.repeat(no_reintro['accepted?'], len(species))
    # concatenate them
    accepted_shape = pd.concat([accepted_shape1, accepted_shape4], axis=0)
    # add a grouping variable to graph each run separately
    grouping1 = np.repeat(final_runs['ID'], len(species))
    grouping4 = np.repeat(no_reintro['ID'], len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping1, grouping4), axis=0)
    # extract the node values from all dataframes
    final_runs1 = final_runs.drop(['ID','accepted?', 'time'], axis=1).values.flatten()
    y_noReintro = no_reintro.drop(['ID', 'accepted?','time'], axis=1).values.flatten()
    # concatenate them
    y_values = np.concatenate((final_runs1, y_noReintro), axis=0)   
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_firstRun = np.tile(species, 8*NUMBER_OF_SIMULATIONS)
    species_noReintro = np.tile(species, (100*len(accepted_simulations_2020)) + (8*len(accepted_simulations)))
    species_list = np.concatenate((species_firstRun, species_noReintro), axis=0)
    # time 
    firstODEyears = np.repeat(final_runs['time'],len(species))
    indices_noReintro = np.repeat(no_reintro['time'],len(species))
    indices = pd.concat([firstODEyears, indices_noReintro], axis=0)
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
    filtered_df = final_df.loc[(final_df['runType'] == "noReintro")]
  
    # Accepted vs. Counterfactual graph (no reintroductions vs. reintroductions) vs. Euro Bison reintro
    colors = ["#6788ee", "#e26952"]
    g = sns.FacetGrid(filtered_df, col="Ecosystem Element", hue = "runType", palette = colors, col_wrap=4, sharey = False)
    g.map(sns.lineplot, 'Time', 'Median')
    g.map(sns.lineplot, 'Time', 'fivePerc')
    g.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in g.axes.flat:
        # fill in lines between quantiles
        # median - 0 1
        # upper - 2 3
        # lower - 4 5
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(), color = '#6788ee', alpha=0.2)
        # ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(), color = '#6788ee', alpha =0.2)
        # ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color = '#e26952', alpha=0.2)
        ax.set_ylabel('Abundance')
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
    plt.show()

plotting()
# graph the runs
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timeit
import seaborn as sns
import numpy.matlib
from scipy import stats
import csv


all_variables = ["grassland_growth","scrub_growth","woodland_growth",
    "pony_grass", "pony_scrub", "pony_wood", 
    "fallow_grass","fallow_scrub","fallow_wood",
    "grass_pony", "grass_fallow", "grass_grass", "grass_cattle", "grass_red","grass_roe", "grass_pig",
    "grass_scrub", "grass_wood", 
    "cattle_grass","cattle_scrub","cattle_wood",
    "red_grass", "red_scrub", "red_wood",
    "roe_grass", "roe_scrub", "roe_wood",
    "pig_grass", "pig_scrub", "pig_wood",
    "scrub_pony", "scrub_fallow", "scrub_grass", "scrub_cattle", "scrub_red","scrub_roe", "scrub_pig",
    "scrub_scrub", "scrub_wood", 
    "wood_pony", "wood_fallow", "wood_grass", "wood_cattle", "wood_red","wood_roe", "wood_pig",
    "wood_scrub", "wood_wood", 
 ]

def ecoNetwork(t, X, A, r):
    X[X<1e-8] = 0

    # consumers with PS2 have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0

    return X * (r + np.matmul(A, X))


def run_model():
    # open the accepted parameters
    # all_parameters = pd.read_csv('all_parameters.csv')

    # accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]

    species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']
    accepted_simulations = 2045 
    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]
    # growth rates
    # growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    # growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(accepted_simulations, len(species)), columns = species)
    # r_accepted = growthRates_2.to_numpy()
    # interaction matrices 
    # interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID','accepted?',"Unnamed: 0.1"], axis=1)
    # interaction_strength_2 = interaction_strength_2.dropna()
    # A_accepted = interaction_strength_2.to_numpy()

    temporary_r = [0, 0, 0.91, 0, 0, 0, 0, 0.35, 0.11] 

    temporary_A = [
            # exmoor pony - special case, no growth
            [0, 0, 2.7, 0, 0, 0, 0, 0.17, 0.4],
            # fallow deer 
            [0, 0, 3.81, 0, 0, 0, 0, 0.37, 0.49],
            # grassland parkland
            [-0.0024, -0.0032, -0.83, -0.015, -0.003, -0.00092, -0.0082, -0.046, -0.049],
            # longhorn cattle  
            [0, 0, 4.99, 0, 0, 0, 0, 0.21, 0.4],
            # red deer  
            [0, 0, 2.8, 0, 0, 0, 0, 0.29, 0.42],
            # roe deer 
            [0, 0, 4.9, 0, 0, 0, 0, 0.25, 0.38],
            # tamworth pig 
            [0, 0, 3.7, 0, 0,0, 0, 0.23, 0.41],  
            # thorny scrub
            [-0.0014, -0.005, 0, -0.0051, -0.0039, -0.0023, -0.0018, -0.015, -0.022],
            # woodland
            [-0.0041, -0.0058, 0, -0.0083, -0.004, -0.0032, -0.0037, 0.0079, -0.0062]
            ]

    sensitivity_results_list = []
    perc_numbers=[]
    parameter_names=[]
    perc_aboveBelow = [-0.5, -0.1, -0.05,-0.01, 0, 0.01, 0.05, 0.1, 0.5]

    # loop through each accepted parameter set 
    for parameter in all_variables:
    #     for temporary_r, temporary_A in zip(r_accepted, np.array_split(A_accepted,accepted_simulations)):
        for perc in perc_aboveBelow:
            r = temporary_r.copy()
            A = np.copy(temporary_A)
            # and each percentage above/below
            r[2] = temporary_r[2] + (temporary_r[2] * perc) if parameter == "grassland_growth" else temporary_r[2]
            r[7] = temporary_r[7] + (temporary_r[7] * perc) if parameter == "scrub_growth" else temporary_r[7]
            r[8] = temporary_r[8] + (temporary_r[8] * perc) if parameter == "woodland_growth" else temporary_r[8]

            # # and interactions
            if parameter == "pony_grass":  A[0][2] = temporary_A[0][2] + (temporary_A[0][2] * perc)

            if parameter == "pony_scrub":  A[0][7] = temporary_A[0][7] + (temporary_A[0][7] * perc)
            if parameter == "pony_wood":  A[0][8] = temporary_A[0][8] + (temporary_A[0][8] * perc)

            if parameter == "fallow_grass":  A[1][2] = temporary_A[1][2] + (temporary_A[1][2] * perc)
            if parameter == "fallow_scrub":  A[1][7] = temporary_A[1][7] + (temporary_A[1][7] * perc)
            if parameter == "fallow_wood":  A[1][8] = temporary_A[1][8] + (temporary_A[1][8] * perc)
            if parameter == "grass_pony":  A[2][0] = temporary_A[2][0] + (temporary_A[2][0] * perc)
            if parameter == "grass_fallow":  A[2][1] = temporary_A[2][1] + (temporary_A[2][1] * perc)
            if parameter == "grass_grass":  A[2][2] = temporary_A[2][2] + (temporary_A[2][2] * perc)
            if parameter == "grass_cattle":  A[2][3] = temporary_A[2][3] + (temporary_A[2][3] * perc)
            if parameter == "grass_red":  A[2][4] = temporary_A[2][4] + (temporary_A[2][4] * perc)
            if parameter == "grass_roe":  A[2][5] = temporary_A[2][5] + (temporary_A[2][5] * perc)
            if parameter == "grass_pig":  A[2][6] = temporary_A[2][6] + (temporary_A[2][6] * perc)
            if parameter == "grass_scrub":  A[2][7] = temporary_A[2][7] + (temporary_A[2][7] * perc)
            if parameter == "grass_wood":  A[2][8] = temporary_A[2][8] + (temporary_A[2][8] * perc)
            if parameter == "cattle_grass":  A[3][2] = temporary_A[3][2] + (temporary_A[3][2] * perc)
            if parameter == "cattle_scrub":  A[3][7] = temporary_A[3][7] + (temporary_A[3][7] * perc)
            if parameter == "cattle_wood":  A[3][8] = temporary_A[3][8] + (temporary_A[3][8] * perc)
            if parameter == "red_grass":  A[4][2] = temporary_A[4][2] + (temporary_A[4][2] * perc)

            if parameter == "red_scrub":  A[4][7] = temporary_A[4][7] + (temporary_A[4][7] * perc)
            if parameter == "red_wood":  A[4][8] = temporary_A[4][8] + (temporary_A[4][8] * perc)
            if parameter == "roe_grass":  A[5][2] = temporary_A[5][2] + (temporary_A[5][2] * perc)
            if parameter == "roe_scrub":  A[5][7] = temporary_A[5][7] + (temporary_A[5][7] * perc)
            if parameter == "roe_wood":  A[5][8] = temporary_A[5][8] + (temporary_A[5][8] * perc)
            if parameter == "pig_grass":  A[6][2] = temporary_A[6][2] + (temporary_A[6][2] * perc)
            if parameter == "pig_scrub":  A[6][7] = temporary_A[6][7] + (temporary_A[6][7] * perc)
            if parameter == "pig_wood":  A[6][8] = temporary_A[6][8] + (temporary_A[6][8] * perc)
            if parameter == "scrub_pony":  A[7][0] = temporary_A[7][0] + (temporary_A[7][0] * perc)
            if parameter == "scrub_fallow":  A[7][1] = temporary_A[7][1] + (temporary_A[7][1] * perc)
            if parameter == "scrub_grass":  A[7][2] = temporary_A[7][2] + (temporary_A[7][2] * perc)
            if parameter == "scrub_cattle":  A[7][3] = temporary_A[7][3] + (temporary_A[7][3] * perc)
            if parameter == "scrub_red":  A[7][4] = temporary_A[7][4] + (temporary_A[7][4] * perc)
            if parameter == "scrub_roe":  A[7][5] = temporary_A[7][5] + (temporary_A[7][5] * perc)
            if parameter == "scrub_pig":  A[7][6] = temporary_A[7][6] + (temporary_A[7][6] * perc)
            if parameter == "scrub_scrub":  A[7][7] = temporary_A[7][7] + (temporary_A[7][7] * perc)
            if parameter == "scrub_wood":  A[7][8] = temporary_A[7][8] + (temporary_A[7][8] * perc)
            if parameter == "wood_pony":  A[8][0] = temporary_A[8][0] + (temporary_A[8][0] * perc)
            if parameter == "wood_fallow":  A[8][1] = temporary_A[8][1] + (temporary_A[8][1] * perc)
            if parameter == "wood_grass":  A[8][2] = temporary_A[8][2] + (temporary_A[8][2] * perc)
            if parameter == "wood_cattle":  A[8][3] = temporary_A[8][3] + (temporary_A[8][3] * perc)
            if parameter == "wood_red":  A[8][4] = temporary_A[8][4] + (temporary_A[8][4] * perc)
            if parameter == "wood_roe":  A[8][5] = temporary_A[8][5] + (temporary_A[8][5] * perc)
            if parameter == "wood_pig":  A[8][6] = temporary_A[8][6] + (temporary_A[8][6] * perc)
            if parameter == "wood_scrub":  A[8][7] = temporary_A[8][7] + (temporary_A[8][7] * perc)
            if parameter == "wood_wood":  A[8][8] = temporary_A[8][8] + (temporary_A[8][8] * perc)
            t_init = np.linspace(2005, 2008.75, 12)
            results = solve_ivp(ecoNetwork, (2005, 2008.75), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
            # reshape the outputs
            y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
            y = pd.DataFrame(data=y, columns=species)
            y['time'] = results.t
            last_results = y.loc[y['time'] == 2008.75]
            last_results = last_results.drop('time', axis=1)
            last_results = last_results.values.flatten()
            # ponies, longhorn cattle, and tamworth pigs reintroduced in 2009
            starting_2009 = last_results.copy()
            starting_2009[0] = 1
            starting_2009[3] = 1
            starting_2009[6] = 1
            t_01 = np.linspace(2009, 2009.75, 3)
            second_ABC = solve_ivp(ecoNetwork, (2009,2009.75), starting_2009,  t_eval = t_01, args=(A, r), method = 'RK23')
            # 2010
            last_values_2009 = second_ABC.y[0:11, 2:3].flatten()
            starting_values_2010 = last_values_2009.copy()
            starting_values_2010[0] = 0.57
            # fallow deer reintroduced
            starting_values_2010[1] = 1
            starting_values_2010[3] = 1.5
            starting_values_2010[6] = 0.85
            t_1 = np.linspace(2010, 2010.75, 3)
            third_ABC = solve_ivp(ecoNetwork, (2010,2010.75), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
            # 2011
            last_values_2010 = third_ABC.y[0:11, 2:3].flatten()
            starting_values_2011 = last_values_2010.copy()
            starting_values_2011[0] = 0.65
            starting_values_2011[1] = 1.9
            starting_values_2011[3] = 1.7
            starting_values_2011[6] = 1.1
            t_2 = np.linspace(2011, 2011.75, 3)
            fourth_ABC = solve_ivp(ecoNetwork, (2011,2011.75), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
            # 2012
            last_values_2011 = fourth_ABC.y[0:11, 2:3].flatten()
            starting_values_2012 = last_values_2011.copy()
            starting_values_2012[0] = 0.74
            starting_values_2012[1] = 2.4
            starting_values_2012[3] = 2.2
            starting_values_2012[6] = 1.7
            t_3 = np.linspace(2012, 2012.75, 3)
            fifth_ABC = solve_ivp(ecoNetwork, (2012,2012.75), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
            # 2013
            last_values_2012 = fifth_ABC.y[0:11, 2:3].flatten()
            starting_values_2013 = last_values_2012.copy()
            starting_values_2013[0] = 0.43
            starting_values_2013[1] = 2.4
            starting_values_2013[3] = 2.4
            # red deer reintroduced
            starting_values_2013[4] = 1
            starting_values_2013[6] = 0.3
            t_4 = np.linspace(2013, 2013.75, 3)
            sixth_ABC = solve_ivp(ecoNetwork, (2013,2013.75), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
            # 2014
            last_values_2013 = sixth_ABC.y[0:11, 2:3].flatten()
            starting_values_2014 = last_values_2013.copy()
            starting_values_2014[0] = 0.44
            starting_values_2014[1] = 2.4
            starting_values_2014[3] = 5
            starting_values_2014[4] = 1
            starting_values_2014[6] = 0.9
            t_5 = np.linspace(2014, 2014.75, 3)
            seventh_ABC = solve_ivp(ecoNetwork, (2014,2014.75), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23') 
            # 2015
            last_values_2014 = seventh_ABC.y[0:11, 2:3].flatten()
            starting_values_2015 = last_values_2014
            starting_values_2015[0] = 0.43
            starting_values_2015[1] = 2.4
            starting_values_2015[3] = 2
            starting_values_2015[4] = 1
            starting_values_2015[6] = 0.71
            t_2015 = np.linspace(2015, 2015.75, 3)
            ABC_2015 = solve_ivp(ecoNetwork, (2015,2015.75), starting_values_2015,  t_eval = t_2015, args=(A, r), method = 'RK23')
            # 2016
            last_values_2015 = ABC_2015.y[0:11, 2:3].flatten()
            starting_values_2016 = last_values_2015.copy()
            starting_values_2016[0] = 0.48
            starting_values_2016[1] = 3.3
            starting_values_2016[3] = 1.7
            starting_values_2016[4] = 2
            starting_values_2016[6] = 0.7
            t_2016 = np.linspace(2016, 2016.75, 3)
            ABC_2016 = solve_ivp(ecoNetwork, (2016,2016.75), starting_values_2016,  t_eval = t_2016, args=(A, r), method = 'RK23')       
            # 2017
            last_values_2016 = ABC_2016.y[0:11, 2:3].flatten()
            starting_values_2017 = last_values_2016.copy()
            starting_values_2017[0] = 0.43
            starting_values_2017[1] = 3.9
            starting_values_2017[3] = 1.7
            starting_values_2017[4] = 1.1
            starting_values_2017[6] = 0.95
            t_2017 = np.linspace(2017, 2017.75, 3)
            ABC_2017 = solve_ivp(ecoNetwork, (2017,2017.75), starting_values_2017,  t_eval = t_2017, args=(A, r), method = 'RK23')     
            # 2018
            last_values_2017 = ABC_2017.y[0:11, 2:3].flatten()
            starting_values_2018 = last_values_2017.copy()
            starting_values_2018[0] = 0
            starting_values_2018[1] = 6.0
            starting_values_2018[3] = 1.9
            starting_values_2018[4] = 1.9
            starting_values_2018[6] = 0.84
            t_2018 = np.linspace(2018, 2018.75, 3)
            ABC_2018 = solve_ivp(ecoNetwork, (2018,2018.75), starting_values_2018,  t_eval = t_2018, args=(A, r), method = 'RK23')     
            # 2019
            last_values_2018 = ABC_2018.y[0:11, 2:3].flatten()
            starting_values_2019 = last_values_2018.copy()
            starting_values_2019[0] = 0
            starting_values_2019[1] = 6.6
            starting_values_2019[3] = 1.7
            starting_values_2019[4] = 2.9
            starting_values_2019[6] = 0.44
            t_2019 = np.linspace(2019, 2019.75, 3)
            ABC_2019 = solve_ivp(ecoNetwork, (2019,2019.75), starting_values_2019,  t_eval = t_2019, args=(A, r), method = 'RK23')
            # 2020
            last_values_2019 = ABC_2019.y[0:11, 2:3].flatten()
            starting_values_2020 = last_values_2019.copy()
            starting_values_2020[0] = 0.65
            starting_values_2020[1] = 5.9
            starting_values_2020[3] = 1.5
            starting_values_2020[4] = 2.7
            starting_values_2020[6] = 0.55
            t_2020 = np.linspace(2020, 2020.75, 3)
            ABC_2020 = solve_ivp(ecoNetwork, (2020,2020.75), starting_values_2020,  t_eval = t_2020, args=(A, r), method = 'RK23')     
            # concatenate & append all the runs
            combined_runs = np.hstack((results.y, second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y, ABC_2020.y))
            combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t, ABC_2020.t))
            # reshape the outputs
            y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 48).transpose(),1)))
            y_2 = pd.DataFrame(data=y_2, columns=species)
            y_2['time'] = combined_times

            # check how many filters were passed
            y_2["passed_filters"] = 0
            year_2009 = y_2.loc[y_2['time'] == 2008.75]
            if (year_2009.iloc[0]['roeDeer'] <= 3.3) & (year_2009.iloc[0]['roeDeer'] >= 1):
                y_2["passed_filters"] += 1
            if (year_2009.iloc[0]['grasslandParkland'] <= 1) & (year_2009.iloc[0]['grasslandParkland'] >= 0.18):
                y_2["passed_filters"] += 1
            if (year_2009.iloc[0]['woodland'] <= 2.9) & (year_2009.iloc[0]['woodland'] >= 1):
                y_2["passed_filters"] += 1
            if (year_2009.iloc[0]['thornyScrub'] <= 4.9) & (year_2009.iloc[0]['thornyScrub'] >= 1):
                y_2["passed_filters"] += 1
            # check 2020
            my_time = y_2.loc[y_2['time'] == 2020.75]
            if (my_time.iloc[0]['exmoorPony'] <= 0.7) & (my_time.iloc[0]['exmoorPony'] >= 0.61):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['roeDeer'] <= 6.7) & (my_time.iloc[0]['roeDeer'] >= 1.7):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['grasslandParkland'] <= 0.41) & (my_time.iloc[0]['grasslandParkland'] >=0.18):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['woodland'] <= 4.6) & (my_time.iloc[0]['woodland'] >= 2.8):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['thornyScrub'] <= 14.4) & (my_time.iloc[0]['thornyScrub'] >=9.7):
                y_2["passed_filters"] += 1
            # check 2015 
            my_time = y_2.loc[y_2['time'] == 2015.75]
            if (my_time.iloc[0]['exmoorPony'] <= 0.49) & (my_time.iloc[0]['exmoorPony'] >= 0.39):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['fallowDeer'] <= 3.9) & (my_time.iloc[0]['fallowDeer'] >= 2.7):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['longhornCattle'] <= 2.9) & (my_time.iloc[0]['longhornCattle'] >=2):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['redDeer'] <= 1.4) & (my_time.iloc[0]['redDeer'] >=0.6):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['tamworthPig'] <= 1.6) & (my_time.iloc[0]['tamworthPig'] >=0.6):
                y_2["passed_filters"] += 1
            # check 2016 
            my_time = y_2.loc[y_2['time'] == 2016.75]
            if (my_time.iloc[0]['exmoorPony'] <= 0.5) & (my_time.iloc[0]['exmoorPony'] >= 0.43):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['fallowDeer'] <= 4.5) & (my_time.iloc[0]['fallowDeer'] >= 3.3):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['longhornCattle'] <= 2.5) & (my_time.iloc[0]['longhornCattle'] >=1.6):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['redDeer'] <= 2.4) & (my_time.iloc[0]['redDeer'] >=1.6):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['tamworthPig'] <= 1.4) & (my_time.iloc[0]['tamworthPig'] >=0.35):
                y_2["passed_filters"] += 1
            # check 2017
            my_time = y_2.loc[y_2['time'] == 2017.75]
            if (my_time.iloc[0]['exmoorPony'] <= 0.48) & (my_time.iloc[0]['exmoorPony'] >= 0.39):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['fallowDeer'] <= 6.6) & (my_time.iloc[0]['fallowDeer'] >= 5.4):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['longhornCattle'] <= 2.5) & (my_time.iloc[0]['longhornCattle'] >=1.6):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['redDeer'] <= 1.5) & (my_time.iloc[0]['redDeer'] >=0.7):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['tamworthPig'] <= 1.6) & (my_time.iloc[0]['tamworthPig'] >=0.6):
                y_2["passed_filters"] += 1
            # check 2018
            my_time = y_2.loc[y_2['time'] == 2018.75]
            if (my_time.iloc[0]['fallowDeer'] <= 8.1) & (my_time.iloc[0]['fallowDeer'] >= 6.9):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['longhornCattle'] <= 2.7) & (my_time.iloc[0]['longhornCattle'] >=1.7):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['redDeer'] <= 2.2) & (my_time.iloc[0]['redDeer'] >=1.5):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['tamworthPig'] <= 1.7) & (my_time.iloc[0]['tamworthPig'] >=0.7):
                y_2["passed_filters"] += 1
            # check 2019
            my_time = y_2.loc[y_2['time'] == 2019.75]
            if (my_time.iloc[0]['fallowDeer'] <= 8.9) & (my_time.iloc[0]['fallowDeer'] >= 7.7):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['longhornCattle'] <= 2.5) & (my_time.iloc[0]['longhornCattle'] >=1.6):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['redDeer'] <= 3.2) & (my_time.iloc[0]['redDeer'] >=2.5):
                y_2["passed_filters"] += 1
            if (my_time.iloc[0]['tamworthPig'] <= 1) & (my_time.iloc[0]['tamworthPig'] >=0):
                y_2["passed_filters"] += 1

            # what percentage of filters did they pass?
            sensitivity_results_list.append(y_2.loc[0,"passed_filters"]/32)
            perc_numbers.append(str(perc))
            parameter_names.append(str(parameter))
    # append to dataframe
    merged_dfs = pd.concat([pd.DataFrame({'Filters': sensitivity_results_list}), pd.DataFrame({'Percentage': perc_numbers}),(pd.DataFrame({'Parameter Name': parameter_names}).reset_index(drop=True))], axis=1)
    merged_dfs.to_csv("sensitivity_table.csv")


def plot_sensitivity(): 
    # open df 
    merged_dfs = pd.read_csv("sensitivity_table.csv")
    merged_dfs = merged_dfs.rename(columns={'Parameter Name': 'Parameter_Name'})

    # change the columns for those we don't want to focus on
    merged_dfs["Top_Parameters"] = ["All other parameters" if cell != "fallow_grass" and cell !="grass_grass" and cell != "grass_scrub" and cell != "grassland_growth" and cell != "pony_wood" and cell != "scrub_growth" and cell != "scrub_scrub" and cell != "woodland_growth" else cell for cell in merged_dfs.Parameter_Name] 


    # drop the non-special ones
    df = merged_dfs[merged_dfs.Top_Parameters != "All other parameters"]
    df["Filters"] = df["Filters"] * 100

    df["Top_Parameters"] = ["Scrub diagonal" if cell == "scrub_scrub" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Grass diagonal" if cell == "grass_grass" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Grass impact on fallow deer" if cell == "fallow_grass" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Scrub impact on grassland" if cell == "grass_scrub" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Woodland impact on ponies" if cell == "pony_wood" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Grassland growth" if cell == "grassland_growth" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Scrubland growth" if cell == "scrub_growth" else cell for cell in df["Top_Parameters"]]
    df["Top_Parameters"] = ["Woodland growth" if cell == "woodland_growth" else cell for cell in df["Top_Parameters"]]

    # now plot it

    # palette = {'darkgrey' for c in merged_dfs.Parameter_Name.unique()}
    g = sns.lineplot(data=df, x="Percentage", y="Filters", hue="Top_Parameters", palette="Paired", marker="o")
    g.set(ylim=(0, 100))

    # plt.legend(title='Parameters', ncol=2)
    plt.title('Sensitivity test results')
    plt.xlabel('Delta')
    plt.ylabel('Percentage of filters passed')
    plt.legend(title='Parameters', loc='lower right')
    plt.tight_layout()
    plt.savefig('sensitivity_ouput.png')

    plt.show()

# run_model()
plot_sensitivity()

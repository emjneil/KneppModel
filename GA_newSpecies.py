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

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland','new_species']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0

    # consumers with PS2 have negative growth rate 
    r[0] = np.log(1/(100*X[0])) if X[0] != 0 else 0
    r[1] = np.log(1/(100*X[1])) if X[1] != 0 else 0
    r[3] = np.log(1/(100*X[3])) if X[3] != 0 else 0
    r[4] = np.log(1/(100*X[4])) if X[4] != 0 else 0
    r[5] = np.log(1/(100*X[5])) if X[5] != 0 else 0
    r[6] = np.log(1/(100*X[6])) if X[6] != 0 else 0
    r[9] = np.log(1/(100*X[9])) if X[9] != 0 else 0

    return X * (r + np.matmul(A, X))

def run_model(X0, A, r):

    all_times = []
    t_init = np.linspace(2005, 2008.95, 12)
    results = solve_ivp(ecoNetwork, (2005, 2008.95), X0,  t_eval = t_init, args=(A, r), method = 'RK23')

    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
    y = pd.DataFrame(data=y, columns=species)
    all_times = np.append(all_times, results.t)
    y['time'] = all_times
    last_results = y.loc[y['time'] == 2008.95]
    last_results = last_results.drop('time', axis=1)
    last_results = last_results.values.flatten()
    # ponies, longhorn cattle, and tamworth pigs reintroduced in 2009
    starting_2009 = last_results.copy()
    starting_2009[0] = 1
    starting_2009[3] = 1
    starting_2009[6] = 1
    t_01 = np.linspace(2009, 2009.95, 3)
    second_ABC = solve_ivp(ecoNetwork, (2009,2009.95), starting_2009,  t_eval = t_01, args=(A, r), method = 'RK23')
    # 2010
    last_values_2009 = second_ABC.y[0:11, 2:3].flatten()
    starting_values_2010 = last_values_2009.copy()
    starting_values_2010[0] = 0.57
    # fallow deer reintroduced
    starting_values_2010[1] = 1
    starting_values_2010[3] = 1.5
    starting_values_2010[6] = 0.85
    t_1 = np.linspace(2010, 2010.95, 3)
    third_ABC = solve_ivp(ecoNetwork, (2010,2010.95), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
    # 2011
    last_values_2010 = third_ABC.y[0:11, 2:3].flatten()
    starting_values_2011 = last_values_2010.copy()
    starting_values_2011[0] = 0.65
    starting_values_2011[1] = 1.9
    starting_values_2011[3] = 1.7
    starting_values_2011[6] = 1.1
    t_2 = np.linspace(2011, 2011.95, 3)
    fourth_ABC = solve_ivp(ecoNetwork, (2011,2011.95), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
    # 2012
    last_values_2011 = fourth_ABC.y[0:11, 2:3].flatten()
    starting_values_2012 = last_values_2011.copy()
    starting_values_2012[0] = 0.74
    starting_values_2012[1] = 2.4
    starting_values_2012[3] = 2.2
    starting_values_2012[6] = 1.7
    t_3 = np.linspace(2012, 2012.95, 3)
    fifth_ABC = solve_ivp(ecoNetwork, (2012,2012.95), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
    # 2013
    last_values_2012 = fifth_ABC.y[0:11, 2:3].flatten()
    starting_values_2013 = last_values_2012.copy()
    starting_values_2013[0] = 0.43
    starting_values_2013[1] = 2.4
    starting_values_2013[3] = 2.4
    # red deer reintroduced
    starting_values_2013[4] = 1
    starting_values_2013[6] = 0.3
    t_4 = np.linspace(2013, 2013.95, 3)
    sixth_ABC = solve_ivp(ecoNetwork, (2013,2013.95), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
    # 2014
    last_values_2013 = sixth_ABC.y[0:11, 2:3].flatten()
    starting_values_2014 = last_values_2013.copy()
    starting_values_2014[0] = 0.44
    starting_values_2014[1] = 2.4
    starting_values_2014[3] = 5
    starting_values_2014[4] = 1
    starting_values_2014[6] = 0.9
    t_5 = np.linspace(2014, 2014.95, 3)
    seventh_ABC = solve_ivp(ecoNetwork, (2014,2014.95), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23') 
    # 2015
    last_values_2014 = seventh_ABC.y[0:11, 2:3].flatten()
    starting_values_2015 = last_values_2014
    starting_values_2015[0] = 0.43
    starting_values_2015[1] = 2.4
    starting_values_2015[3] = 2
    starting_values_2015[4] = 1
    starting_values_2015[6] = 0.71
    t_2015 = np.linspace(2015, 2015.95, 3)
    ABC_2015 = solve_ivp(ecoNetwork, (2015,2015.95), starting_values_2015,  t_eval = t_2015, args=(A, r), method = 'RK23')
    # 2016
    last_values_2015 = ABC_2015.y[0:11, 2:3].flatten()
    starting_values_2016 = last_values_2015.copy()
    starting_values_2016[0] = 0.48
    starting_values_2016[1] = 3.3
    starting_values_2016[3] = 1.7
    starting_values_2016[4] = 2
    starting_values_2016[6] = 0.7
    t_2016 = np.linspace(2016, 2016.95, 3)
    ABC_2016 = solve_ivp(ecoNetwork, (2016,2016.95), starting_values_2016,  t_eval = t_2016, args=(A, r), method = 'RK23')       
    # 2017
    last_values_2016 = ABC_2016.y[0:11, 2:3].flatten()
    starting_values_2017 = last_values_2016.copy()
    starting_values_2017[0] = 0.43
    starting_values_2017[1] = 3.9
    starting_values_2017[3] = 1.7
    starting_values_2017[4] = 1.1
    starting_values_2017[6] = 0.95
    t_2017 = np.linspace(2017, 2017.95, 3)
    ABC_2017 = solve_ivp(ecoNetwork, (2017,2017.95), starting_values_2017,  t_eval = t_2017, args=(A, r), method = 'RK23')     
    # 2018
    last_values_2017 = ABC_2017.y[0:11, 2:3].flatten()
    starting_values_2018 = last_values_2017.copy()
    # pretend bison were reintroduced (to estimate growth rate / interaction values)
    starting_values_2018[0] = 0
    starting_values_2018[1] = 6.0
    starting_values_2018[3] = 1.9
    starting_values_2018[4] = 1.9
    starting_values_2018[6] = 0.84
    t_2018 = np.linspace(2018, 2018.95, 3)
    ABC_2018 = solve_ivp(ecoNetwork, (2018,2018.95), starting_values_2018,  t_eval = t_2018, args=(A, r), method = 'RK23')     
    # 2019
    last_values_2018 = ABC_2018.y[0:11, 2:3].flatten()
    starting_values_2019 = last_values_2018.copy()
    starting_values_2019[0] = 0
    starting_values_2019[1] = 6.6
    starting_values_2019[3] = 1.7
    starting_values_2019[4] = 2.9
    starting_values_2019[6] = 0.44
    t_2019 = np.linspace(2019, 2019.95, 3)
    ABC_2019 = solve_ivp(ecoNetwork, (2019,2019.95), starting_values_2019,  t_eval = t_2019, args=(A, r), method = 'RK23')
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
    
    # forecast fifty years 
    years = 100
    previous_model = ABC_2020
    forecasting_models = {}
    forecasting_models_time = {}
    for year in range(years): 
        my_year = 2021 + (year-0.25)
        # get the initial conditions of the previous year
        previous_year = (2021 + year) - 1
        last_values = previous_model.y[0:10, 2:3].flatten()
        # my starting values
        starting_values = last_values.copy()
        starting_values[0] = 0.65
        starting_values[1] = 5.9
        starting_values[3] = 1.5
        starting_values[4] = 2.7
        starting_values[6] = 0.55
        # introduce new species
        starting_values[9] = 1

        t = np.linspace(previous_year, my_year, 3)
        previous_model = solve_ivp(ecoNetwork, (previous_year,my_year), starting_values,  t_eval = t, args=(A, r), method = 'RK23')   
        my_year_rounded = 2021 + (year)
        forecasting_models[("ABC_" + str(my_year_rounded))] = previous_model.y
        forecasting_models_time[("ABC_" + str(my_year_rounded))] = previous_model.t

    # concatenate & append all the runs
    combined_runs = np.hstack((results.y,second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y,ABC_2020.y, 
                        forecasting_models["ABC_2021"],forecasting_models["ABC_2022"],forecasting_models["ABC_2023"],forecasting_models["ABC_2024"],forecasting_models["ABC_2025"],forecasting_models["ABC_2026"],forecasting_models["ABC_2027"],forecasting_models["ABC_2028"],forecasting_models["ABC_2029"],
                        forecasting_models["ABC_2030"],forecasting_models["ABC_2031"],forecasting_models["ABC_2032"],forecasting_models["ABC_2033"],forecasting_models["ABC_2034"],forecasting_models["ABC_2035"],forecasting_models["ABC_2036"],forecasting_models["ABC_2037"],forecasting_models["ABC_2038"],forecasting_models["ABC_2039"],
                        forecasting_models["ABC_2040"],forecasting_models["ABC_2041"],forecasting_models["ABC_2042"],forecasting_models["ABC_2043"],forecasting_models["ABC_2044"],forecasting_models["ABC_2045"],forecasting_models["ABC_2046"],forecasting_models["ABC_2047"],forecasting_models["ABC_2048"],forecasting_models["ABC_2049"],
                        forecasting_models["ABC_2050"],forecasting_models["ABC_2051"],forecasting_models["ABC_2052"],forecasting_models["ABC_2053"],forecasting_models["ABC_2054"],forecasting_models["ABC_2055"],forecasting_models["ABC_2056"],forecasting_models["ABC_2057"],forecasting_models["ABC_2058"],forecasting_models["ABC_2059"],
                        forecasting_models["ABC_2060"],forecasting_models["ABC_2061"],forecasting_models["ABC_2062"],forecasting_models["ABC_2063"],forecasting_models["ABC_2064"],forecasting_models["ABC_2065"],forecasting_models["ABC_2066"],forecasting_models["ABC_2067"],forecasting_models["ABC_2068"],forecasting_models["ABC_2069"],
                        forecasting_models["ABC_2070"],forecasting_models["ABC_2071"],forecasting_models["ABC_2072"],forecasting_models["ABC_2073"],forecasting_models["ABC_2074"],forecasting_models["ABC_2075"],forecasting_models["ABC_2076"],forecasting_models["ABC_2077"],forecasting_models["ABC_2078"],forecasting_models["ABC_2079"],
                        forecasting_models["ABC_2080"],forecasting_models["ABC_2081"],forecasting_models["ABC_2082"],forecasting_models["ABC_2083"],forecasting_models["ABC_2084"],forecasting_models["ABC_2085"],forecasting_models["ABC_2086"],forecasting_models["ABC_2087"],forecasting_models["ABC_2088"],forecasting_models["ABC_2089"],
                        forecasting_models["ABC_2090"],forecasting_models["ABC_2091"],forecasting_models["ABC_2092"],forecasting_models["ABC_2093"],forecasting_models["ABC_2094"],forecasting_models["ABC_2095"],forecasting_models["ABC_2096"],forecasting_models["ABC_2097"],forecasting_models["ABC_2098"],forecasting_models["ABC_2099"],
                        forecasting_models["ABC_2100"],forecasting_models["ABC_2101"],forecasting_models["ABC_2102"],forecasting_models["ABC_2103"],forecasting_models["ABC_2104"],forecasting_models["ABC_2105"],forecasting_models["ABC_2106"],forecasting_models["ABC_2107"],forecasting_models["ABC_2108"],forecasting_models["ABC_2109"],
                        forecasting_models["ABC_2110"],forecasting_models["ABC_2111"],forecasting_models["ABC_2112"],forecasting_models["ABC_2113"],forecasting_models["ABC_2114"],forecasting_models["ABC_2115"],forecasting_models["ABC_2116"],forecasting_models["ABC_2117"],forecasting_models["ABC_2118"],forecasting_models["ABC_2119"],
                        forecasting_models["ABC_2120"]                        ))
    # now times
    combined_times = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t,ABC_2020.t,
                    forecasting_models_time["ABC_2021"],forecasting_models_time["ABC_2022"],forecasting_models_time["ABC_2023"],forecasting_models_time["ABC_2024"],forecasting_models_time["ABC_2025"],forecasting_models_time["ABC_2026"],forecasting_models_time["ABC_2027"],forecasting_models_time["ABC_2028"],forecasting_models_time["ABC_2029"],
                    forecasting_models_time["ABC_2030"],forecasting_models_time["ABC_2031"],forecasting_models_time["ABC_2032"],forecasting_models_time["ABC_2033"],forecasting_models_time["ABC_2034"],forecasting_models_time["ABC_2035"],forecasting_models_time["ABC_2036"],forecasting_models_time["ABC_2037"],forecasting_models_time["ABC_2038"],forecasting_models_time["ABC_2039"],
                    forecasting_models_time["ABC_2040"],forecasting_models_time["ABC_2041"],forecasting_models_time["ABC_2042"],forecasting_models_time["ABC_2043"],forecasting_models_time["ABC_2044"],forecasting_models_time["ABC_2045"],forecasting_models_time["ABC_2046"],forecasting_models_time["ABC_2047"],forecasting_models_time["ABC_2048"],forecasting_models_time["ABC_2049"],
                    forecasting_models_time["ABC_2050"],forecasting_models_time["ABC_2051"],forecasting_models_time["ABC_2052"],forecasting_models_time["ABC_2053"],forecasting_models_time["ABC_2054"],forecasting_models_time["ABC_2055"],forecasting_models_time["ABC_2056"],forecasting_models_time["ABC_2057"],forecasting_models_time["ABC_2058"],forecasting_models_time["ABC_2059"],
                    forecasting_models_time["ABC_2060"],forecasting_models_time["ABC_2061"],forecasting_models_time["ABC_2062"],forecasting_models_time["ABC_2063"],forecasting_models_time["ABC_2064"],forecasting_models_time["ABC_2065"],forecasting_models_time["ABC_2066"],forecasting_models_time["ABC_2067"],forecasting_models_time["ABC_2068"],forecasting_models_time["ABC_2069"],
                    forecasting_models_time["ABC_2070"],forecasting_models_time["ABC_2071"],forecasting_models_time["ABC_2072"],forecasting_models_time["ABC_2073"],forecasting_models_time["ABC_2074"],forecasting_models_time["ABC_2075"],forecasting_models_time["ABC_2076"],forecasting_models_time["ABC_2077"],forecasting_models_time["ABC_2078"],forecasting_models_time["ABC_2079"],
                    forecasting_models_time["ABC_2080"],forecasting_models_time["ABC_2081"],forecasting_models_time["ABC_2082"],forecasting_models_time["ABC_2083"],forecasting_models_time["ABC_2084"],forecasting_models_time["ABC_2085"],forecasting_models_time["ABC_2086"],forecasting_models_time["ABC_2087"],forecasting_models_time["ABC_2088"],forecasting_models_time["ABC_2089"],
                    forecasting_models_time["ABC_2090"],forecasting_models_time["ABC_2091"],forecasting_models_time["ABC_2092"],forecasting_models_time["ABC_2093"],forecasting_models_time["ABC_2094"],forecasting_models_time["ABC_2095"],forecasting_models_time["ABC_2096"],forecasting_models_time["ABC_2097"],forecasting_models_time["ABC_2098"],forecasting_models_time["ABC_2099"],
                    forecasting_models_time["ABC_2100"],forecasting_models_time["ABC_2101"],forecasting_models_time["ABC_2102"],forecasting_models_time["ABC_2103"],forecasting_models_time["ABC_2104"],forecasting_models_time["ABC_2105"],forecasting_models_time["ABC_2106"],forecasting_models_time["ABC_2107"],forecasting_models_time["ABC_2108"],forecasting_models_time["ABC_2109"],
                    forecasting_models_time["ABC_2110"],forecasting_models_time["ABC_2111"],forecasting_models_time["ABC_2112"],forecasting_models_time["ABC_2113"],forecasting_models_time["ABC_2114"],forecasting_models_time["ABC_2115"],forecasting_models_time["ABC_2116"],forecasting_models_time["ABC_2117"],forecasting_models_time["ABC_2118"],forecasting_models_time["ABC_2119"],
                    forecasting_models_time["ABC_2120"]                        ))

    
    # reshape the outputs
    y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 348).transpose(),1)))
    y_2 = pd.DataFrame(data=y_2, columns=species)
    y_2['time'] = combined_times
    # choose the final year (we want to compare the final year to the middle of the filters)
    last_year_1 = y_2.loc[y_2['time'] == 2119.75]
    last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
    return last_year_1, y_2



def objectiveFunction(x):

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0]

    r = [0, 0, 0.91, 0, 0, 0, 0, 0.35, 0.11, 0]  # ps2

    A = [
        # exmoor pony - special case, no growth
        [0, 0, 2.7, 0, 0, 0, 0, 0.17, 0.4, 0],
        # fallow deer 
        [0, 0, 3.81, 0, 0, 0, 0, 0.37, 0.49, 0],
        # grassland parkland
        [-0.0024, -0.0032, -0.83, -0.015, -0.003, -0.00092, -0.0082, -0.046, -0.049, x[0]],
        # longhorn cattle  
        [0, 0, 4.99, 0, 0, 0, 0, 0.21, 0.4, 0],
        # red deer  
        [0, 0, 2.8, 0, 0, 0, 0, 0.29, 0.42, 0],
        # roe deer 
        [0, 0, 4.9, 0, 0, 0, 0, 0.25, 0.38, 0],
        # tamworth pig 
        [0, 0, 3.7, 0, 0,0, 0, 0.23, 0.41, 0],  
        # thorny scrub
        [-0.0014, -0.005, 0, -0.0051, -0.0039, -0.0023, -0.0018, -0.015, -0.022, x[1]],
        # woodland
        [-0.0041, -0.0058, 0, -0.0083, -0.004, -0.0032, -0.0037, 0.0079, -0.0062, x[2]],
        # new species
        [0,0,x[3],0,0,0,0,x[4],x[5],0]

        ]


    # run the model
    last_year_1,y_2 = run_model(X0, A, r)
    # find runs with outputs closest to the middle of the filtering conditions
    result = ( 
        # want 25% grassland - 0.3; 10% = 0.1
        # (((last_year_1[8]-8.6))**2) # 50% wood
        # (((last_year_1[8]-7.8))**2) # 45% wood
        (((last_year_1[8]-4.3))**2) # 25% wood

        )
    # print the output
    if result < 1:
        print(result)
    return (result)


def run_optimizer():
    bds = np.array([
        # new species impacts
        [-0.0035,-0.001],[-0.1,0],[-0.1,0],
        # gains from habitats
        [0,5],[0,1],[0,1],
    ])

    algorithm_param = {'max_num_iteration': 25,\
                    'population_size':150,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.3,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

    optimization =  ga(function = objectiveFunction, dimension = 6, variable_type = 'real',variable_boundaries= bds, algorithm_parameters = algorithm_param, function_timeout=30)
    optimization.run()
    print(optimization)
    with open('optimize_newSpecies_ps2.txt', 'w') as f:
        print(optimization.output_dict, file=f)
    return optimization.output_dict



def graph_results():
    output_parameters = run_optimizer()
    # define X0
    X0 = [0,0,1,0,0,1,0,1,1,0]

    # open the other parameters
    all_parameters = pd.read_csv('all_parameters_ps2_unstable.csv')
    # how many were accepted? 
    number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/((len(species)-1)*2)
    # get the accepted parameters
    accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
    accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)-1), columns = species[0:9]).to_numpy()
    # select accepted interaction strengths
    interaction_strength_2 = accepted_parameters.drop(["Unnamed: 0.1",'X0', 'growth', 'ID', 'accepted?'], axis=1).dropna()
    A_accepted = interaction_strength_2.to_numpy()
    
    all_runs_backcast = []
    all_times_backcast = []
    runs_left = number_accepted

    for r, A in zip(accepted_r, np.array_split(A_accepted,number_accepted)):
        r = np.append(r, 0)

        # add new species
        A = [
        # exmoor pony - special case, no growth
        [A[0,0], A[0,1], A[0,2], A[0,3], A[0,4], A[0,5], A[0,6], A[0,7], A[0,8], 0],
        # fallow deer 
        [A[1,0], A[1,1], A[1,2], A[1,3], A[1,4], A[1,5], A[1,6], A[1,7], A[1,8], 0],
        # grassland parkland
        [A[2,0], A[2,1], A[2,2], A[2,3],A[2,4], A[2,5], A[2,6], A[2,7], A[2,8],output_parameters["variable"][0]],
        # longhorn cattle  
        [0, 0, A[3,2], A[3,3], 0, 0, 0, A[3,7], A[3,8],0],
        # red deer  
        [0, 0, A[4,2], 0, A[4,4], 0, 0, A[4,7], A[4,8],0],
        # roe deer 
        [0, 0, A[5,2], 0, 0, A[5,5], 0, A[5,7], A[5,8],0],
        # tamworth pig 
        [0, 0, A[6,2], 0, 0,0, A[6,6], A[6,7], A[6,8],0],  
        # thorny scrub
        [A[7,0], A[7,1], 0, A[7,3], A[7,4], A[7,5], A[7,6], A[7,7], A[7,8],output_parameters["variable"][1]],
        # woodland
        [A[8,0], A[8,1], 0, A[8,3], A[8,4], A[8,5], A[8,6], A[8,7], A[8,8],output_parameters["variable"][2]],
        # new species
        [0,0,output_parameters["variable"][3],0,0,0,0,output_parameters["variable"][4],output_parameters["variable"][5],0]
]       
        # start the first run
        t_init = np.linspace(2005, 2008.75, 12)
        results = solve_ivp(ecoNetwork, (2005, 2008.75), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
        # reshape the outputs
        y = (np.vstack(np.hsplit(results.y.reshape(len(species), 12).transpose(),1)))
        y = pd.DataFrame(data=y, columns=species)
        y['time'] = results.t
        last_results = y.loc[y['time'] == 2008.75]
        last_results = last_results.drop('time', axis=1).values.flatten()
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
        # forecast fifty years 
        years = 100
        previous_model = ABC_2020
        forecasting_models = {}
        forecasting_models_time = {}
        for year in range(years): 
            my_year = 2021 + (year-0.25)
            # get the initial conditions of the previous year
            previous_year = (2021 + year) - 1
            last_values = previous_model.y[0:11, 2:3].flatten()
            # my starting values
            starting_values = last_values.copy()
            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5 
            starting_values[4] = 2.7
            starting_values[6] = 0.55
            starting_values[9] = 1
            t = np.linspace(previous_year, my_year, 3)
            previous_model = solve_ivp(ecoNetwork, (previous_year,my_year), starting_values,  t_eval = t, args=(A, r), method = 'RK23')   
            my_year_rounded = 2021 + (year)
            forecasting_models[("ABC_" + str(my_year_rounded))] = previous_model.y
            forecasting_models_time[("ABC_" + str(my_year_rounded))] = previous_model.t

        # concatenate & append all the runs
        all_runs_single = np.hstack((results.y,second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, ABC_2015.y, ABC_2016.y, ABC_2017.y, ABC_2018.y, ABC_2019.y,ABC_2020.y, 
                            forecasting_models["ABC_2021"],forecasting_models["ABC_2022"],forecasting_models["ABC_2023"],forecasting_models["ABC_2024"],forecasting_models["ABC_2025"],forecasting_models["ABC_2026"],forecasting_models["ABC_2027"],forecasting_models["ABC_2028"],forecasting_models["ABC_2029"],
                            forecasting_models["ABC_2030"],forecasting_models["ABC_2031"],forecasting_models["ABC_2032"],forecasting_models["ABC_2033"],forecasting_models["ABC_2034"],forecasting_models["ABC_2035"],forecasting_models["ABC_2036"],forecasting_models["ABC_2037"],forecasting_models["ABC_2038"],forecasting_models["ABC_2039"],
                            forecasting_models["ABC_2040"],forecasting_models["ABC_2041"],forecasting_models["ABC_2042"],forecasting_models["ABC_2043"],forecasting_models["ABC_2044"],forecasting_models["ABC_2045"],forecasting_models["ABC_2046"],forecasting_models["ABC_2047"],forecasting_models["ABC_2048"],forecasting_models["ABC_2049"],
                            forecasting_models["ABC_2050"],forecasting_models["ABC_2051"],forecasting_models["ABC_2052"],forecasting_models["ABC_2053"],forecasting_models["ABC_2054"],forecasting_models["ABC_2055"],forecasting_models["ABC_2056"],forecasting_models["ABC_2057"],forecasting_models["ABC_2058"],forecasting_models["ABC_2059"],
                            forecasting_models["ABC_2060"],forecasting_models["ABC_2061"],forecasting_models["ABC_2062"],forecasting_models["ABC_2063"],forecasting_models["ABC_2064"],forecasting_models["ABC_2065"],forecasting_models["ABC_2066"],forecasting_models["ABC_2067"],forecasting_models["ABC_2068"],forecasting_models["ABC_2069"],
                            forecasting_models["ABC_2070"],forecasting_models["ABC_2071"],forecasting_models["ABC_2072"],forecasting_models["ABC_2073"],forecasting_models["ABC_2074"],forecasting_models["ABC_2075"],forecasting_models["ABC_2076"],forecasting_models["ABC_2077"],forecasting_models["ABC_2078"],forecasting_models["ABC_2079"],
                            forecasting_models["ABC_2080"],forecasting_models["ABC_2081"],forecasting_models["ABC_2082"],forecasting_models["ABC_2083"],forecasting_models["ABC_2084"],forecasting_models["ABC_2085"],forecasting_models["ABC_2086"],forecasting_models["ABC_2087"],forecasting_models["ABC_2088"],forecasting_models["ABC_2089"],
                            forecasting_models["ABC_2090"],forecasting_models["ABC_2091"],forecasting_models["ABC_2092"],forecasting_models["ABC_2093"],forecasting_models["ABC_2094"],forecasting_models["ABC_2095"],forecasting_models["ABC_2096"],forecasting_models["ABC_2097"],forecasting_models["ABC_2098"],forecasting_models["ABC_2099"],
                            forecasting_models["ABC_2100"],forecasting_models["ABC_2101"],forecasting_models["ABC_2102"],forecasting_models["ABC_2103"],forecasting_models["ABC_2104"],forecasting_models["ABC_2105"],forecasting_models["ABC_2106"],forecasting_models["ABC_2107"],forecasting_models["ABC_2108"],forecasting_models["ABC_2109"],
                            forecasting_models["ABC_2110"],forecasting_models["ABC_2111"],forecasting_models["ABC_2112"],forecasting_models["ABC_2113"],forecasting_models["ABC_2114"],forecasting_models["ABC_2115"],forecasting_models["ABC_2116"],forecasting_models["ABC_2117"],forecasting_models["ABC_2118"],forecasting_models["ABC_2119"],
                            forecasting_models["ABC_2120"]
                            ))
        all_runs_backcast = np.append(all_runs_backcast,all_runs_single)
        # now times
        all_times_single = np.hstack((results.t, second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, ABC_2015.t, ABC_2016.t, ABC_2017.t, ABC_2018.t, ABC_2019.t,ABC_2020.t,
                        forecasting_models_time["ABC_2021"],forecasting_models_time["ABC_2022"],forecasting_models_time["ABC_2023"],forecasting_models_time["ABC_2024"],forecasting_models_time["ABC_2025"],forecasting_models_time["ABC_2026"],forecasting_models_time["ABC_2027"],forecasting_models_time["ABC_2028"],forecasting_models_time["ABC_2029"],
                        forecasting_models_time["ABC_2030"],forecasting_models_time["ABC_2031"],forecasting_models_time["ABC_2032"],forecasting_models_time["ABC_2033"],forecasting_models_time["ABC_2034"],forecasting_models_time["ABC_2035"],forecasting_models_time["ABC_2036"],forecasting_models_time["ABC_2037"],forecasting_models_time["ABC_2038"],forecasting_models_time["ABC_2039"],
                        forecasting_models_time["ABC_2040"],forecasting_models_time["ABC_2041"],forecasting_models_time["ABC_2042"],forecasting_models_time["ABC_2043"],forecasting_models_time["ABC_2044"],forecasting_models_time["ABC_2045"],forecasting_models_time["ABC_2046"],forecasting_models_time["ABC_2047"],forecasting_models_time["ABC_2048"],forecasting_models_time["ABC_2049"],
                        forecasting_models_time["ABC_2050"],forecasting_models_time["ABC_2051"],forecasting_models_time["ABC_2052"],forecasting_models_time["ABC_2053"],forecasting_models_time["ABC_2054"],forecasting_models_time["ABC_2055"],forecasting_models_time["ABC_2056"],forecasting_models_time["ABC_2057"],forecasting_models_time["ABC_2058"],forecasting_models_time["ABC_2059"],
                        forecasting_models_time["ABC_2060"],forecasting_models_time["ABC_2061"],forecasting_models_time["ABC_2062"],forecasting_models_time["ABC_2063"],forecasting_models_time["ABC_2064"],forecasting_models_time["ABC_2065"],forecasting_models_time["ABC_2066"],forecasting_models_time["ABC_2067"],forecasting_models_time["ABC_2068"],forecasting_models_time["ABC_2069"],
                        forecasting_models_time["ABC_2070"],forecasting_models_time["ABC_2071"],forecasting_models_time["ABC_2072"],forecasting_models_time["ABC_2073"],forecasting_models_time["ABC_2074"],forecasting_models_time["ABC_2075"],forecasting_models_time["ABC_2076"],forecasting_models_time["ABC_2077"],forecasting_models_time["ABC_2078"],forecasting_models_time["ABC_2079"],
                        forecasting_models_time["ABC_2080"],forecasting_models_time["ABC_2081"],forecasting_models_time["ABC_2082"],forecasting_models_time["ABC_2083"],forecasting_models_time["ABC_2084"],forecasting_models_time["ABC_2085"],forecasting_models_time["ABC_2086"],forecasting_models_time["ABC_2087"],forecasting_models_time["ABC_2088"],forecasting_models_time["ABC_2089"],
                        forecasting_models_time["ABC_2090"],forecasting_models_time["ABC_2091"],forecasting_models_time["ABC_2092"],forecasting_models_time["ABC_2093"],forecasting_models_time["ABC_2094"],forecasting_models_time["ABC_2095"],forecasting_models_time["ABC_2096"],forecasting_models_time["ABC_2097"],forecasting_models_time["ABC_2098"],forecasting_models_time["ABC_2099"],
                        forecasting_models_time["ABC_2100"],forecasting_models_time["ABC_2101"],forecasting_models_time["ABC_2102"],forecasting_models_time["ABC_2103"],forecasting_models_time["ABC_2104"],forecasting_models_time["ABC_2105"],forecasting_models_time["ABC_2106"],forecasting_models_time["ABC_2107"],forecasting_models_time["ABC_2108"],forecasting_models_time["ABC_2109"],
                        forecasting_models_time["ABC_2110"],forecasting_models_time["ABC_2111"],forecasting_models_time["ABC_2112"],forecasting_models_time["ABC_2113"],forecasting_models_time["ABC_2114"],forecasting_models_time["ABC_2115"],forecasting_models_time["ABC_2116"],forecasting_models_time["ABC_2117"],forecasting_models_time["ABC_2118"],forecasting_models_time["ABC_2119"],
                        forecasting_models_time["ABC_2120"]
        ))

        # all_times_forecast = np.append(all_times_forecast,forecasting_models_time)
        all_times_backcast = np.append(all_times_backcast,all_times_single)
        # note the runs left
        runs_left = runs_left +- 1
    # reshape the outputs
    final_results = (np.vstack(np.hsplit(all_runs_backcast.reshape(len(species)*int(number_accepted), 348).transpose(),number_accepted)))
    final_results = pd.DataFrame(data=final_results, columns=species)
    final_results['time'] = all_times_backcast


    # put it in a dataframe
    y_values = final_results[["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland", "new_species"]].values.flatten()
    species_list = np.tile(["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland","new_species"],len(final_results)) 
    indices = np.repeat(final_results['time'], 10)


    final_results = pd.DataFrame(
        {'Abundance': y_values, 'Ecosystem Element': species_list, 'Time': indices})

    # calculate median 
    m = final_results.groupby(['Time', 'Ecosystem Element'])[['Abundance']].apply(np.median)
    m.name = 'Median'
    final_results = final_results.join(m, on=['Time', 'Ecosystem Element'])
    # calculate quantiles
    perc1 = final_results.groupby(['Time','Ecosystem Element'])['Abundance'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_results = final_results.join(perc1, on=['Time', 'Ecosystem Element'])
    perc2 = final_results.groupby(['Time', 'Ecosystem Element'])['Abundance'].quantile(.05)
    perc2.name = "fivePerc"
    final_results = final_results.join(perc2, on=['Time', 'Ecosystem Element'])
    final_df = final_results.reset_index(drop=True)
    
    # now graph it
    palette=['#db5f57', '#57d3db', '#57db5f','#5f57db', '#db57d3']
    f = sns.FacetGrid(final_df, col="Ecosystem Element", palette = palette, col_wrap=4, sharey = False)
    f.map(sns.lineplot, 'Time', 'Median')
    f.map(sns.lineplot, 'Time', 'fivePerc')
    f.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in f.axes.flat:
        ax.fill_between(ax.lines[1].get_xdata(),ax.lines[1].get_ydata(), ax.lines[2].get_ydata(),color="#db5f57", alpha =0.2)
        ax.set_ylabel('Abundance')
        ax.set_xlabel('Time (Years)')

    # show plot
    f.fig.suptitle('Engineering towards an ecosystem with 25% woodland')

    plt.tight_layout()
    plt.savefig('engineering_newSpecies_ps1_unstable.png')
    plt.show()



graph_results()
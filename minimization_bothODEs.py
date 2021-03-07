4.# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
from skopt import gp_minimize
import numpy.matlib


# # # # # --------- MINIMIZATION ----------- # # # # # # #

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','organicCarbon','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']


def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    X[X>1e5] = 1e5
    return X * (r + np.matmul(A, X))


def objectiveFunction(x):
    # only primary producers should have intrinsic growth rate
    x = np.insert(x,0,0)
    x = np.insert(x,1,0)
    x = np.insert(x,3,0)
    x = np.insert(x,4,0)
    x = np.insert(x,5,0)
    x = np.insert(x,6,0)
    x = np.insert(x,7,0)
    r =  x[0:10]

    # insert interaction matrices of 0
    x = np.insert(x,11,0)
    x = np.insert(x,12,0)
    x = np.insert(x,13,0)
    x = np.insert(x,14,0)
    x = np.insert(x,15,0)
    x = np.insert(x,16,0)
    x = np.insert(x,17,0)
    x = np.insert(x,18,0)
    x = np.insert(x,19,0)
    x = np.insert(x,20,0)
    x = np.insert(x,23,0)
    x = np.insert(x,24,0)
    x = np.insert(x,25,0)
    x = np.insert(x,26,0)
    x = np.insert(x,27,0)
    x = np.insert(x,34,0)
    x = np.insert(x,40,0)
    x = np.insert(x,41,0)
    x = np.insert(x,44,0)
    x = np.insert(x,45,0)
    x = np.insert(x,46,0)
    x = np.insert(x,47,0)
    x = np.insert(x,60,0)
    x = np.insert(x,61,0)
    x = np.insert(x,63,0)
    x = np.insert(x,64,0)
    x = np.insert(x,66,0)
    x = np.insert(x,67,0)
    x = np.insert(x,70,0)
    x = np.insert(x,71,0)
    x = np.insert(x,73,0)
    x = np.insert(x,74,0)
    x = np.insert(x,75,0)
    x = np.insert(x,77,0)
    x = np.insert(x,80,0)
    x = np.insert(x,81,0)
    x = np.insert(x,83,0)
    x = np.insert(x,84,0)
    x = np.insert(x,85,0)
    x = np.insert(x,86,0)
    x = np.insert(x,92,0)
    x = np.insert(x,94,0)
    x = np.insert(x,102,0)
    x = np.insert(x,104,0)

    # define X0
    X0 = [0,0,1,0,1,0,1,0,1,1]
    # define interaction strength
    interaction_strength = x[10:110]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(10,10),index = species, columns=species)
    A = interaction_strength.to_numpy()


    # run 2005, 2006, 2007, 2008
    all_times = []
    t_init = np.linspace(0, 47.9, 144)
    results = solve_ivp(ecoNetwork, (0, 47.9), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 144).transpose(),1)))
    y = pd.DataFrame(data=y, columns=species)
    all_times = np.append(all_times, results.t)
    y['time'] = all_times
    last_results = y.loc[y['time'] == 47.9]
    last_results = last_results.drop('time', axis=1)
    last_results = last_results.values.flatten()
    # ponies, longhorn cattle, and tamworth pigs reintroduced in 2009
    last_results[0] = 1
    last_results[3] = 1
    last_results[7] = 1

    # run for 12 months: 2009
    t = np.linspace(48, 59.9, 36)
    second_ABC = solve_ivp(ecoNetwork, (48,59.9), last_results,  t_eval = t, args=(A, r), method = 'RK23')
    # identify last month (year running March-March like at Knepp)
    starting_values_2010 = second_ABC.y[0:10, 35:36].flatten()
    # force 2010's values; fallow deer reintroduced this year
    starting_values_2010[0] = 0.57
    starting_values_2010[1] = 1
    starting_values_2010[3] = 1.45
    starting_values_2010[7] = 0.85
    t_1 = np.linspace(60, 71.9, 36)
    third_ABC = solve_ivp(ecoNetwork, (60,71.9), starting_values_2010,  t_eval = t_1, args=(A, r), method = 'RK23')
    # take those values and re-run for another year (2011), adding forcings
    starting_values_2011 = third_ABC.y[0:10, 35:36].flatten()
    starting_values_2011[0] = 0.65
    starting_values_2011[1] = 1.93
    starting_values_2011[3] = 1.74
    starting_values_2011[7] = 1.1
    t_2 = np.linspace(72, 83.9, 36)
    fourth_ABC = solve_ivp(ecoNetwork, (72,83.9), starting_values_2011,  t_eval = t_2, args=(A, r), method = 'RK23')
    # take those values and re-run for another year (2012), adding forcings
    starting_2012 = fourth_ABC.y[0:10, 35:36].flatten()
    starting_values_2012 = starting_2012
    starting_values_2012[0] = 0.74
    starting_values_2012[1] = 2.38
    starting_values_2012[3] = 2.19
    starting_values_2012[7] = 1.65
    t_3 = np.linspace(84, 95.9, 36)
    fifth_ABC = solve_ivp(ecoNetwork, (84,95.9), starting_values_2012,  t_eval = t_3, args=(A, r), method = 'RK23')
    # take those values and re-run for another year (2013), adding forcings
    starting_2013 = fifth_ABC.y[0:10, 35:36].flatten()
    starting_values_2013 = starting_2013
    starting_values_2013[0] = 0.43
    starting_values_2013[1] = 2.38
    starting_values_2013[3] = 2.43
    # red deer reintroduced
    starting_values_2013[5] = 1
    starting_values_2013[7] = 0.3
    t_4 = np.linspace(96, 107.9, 36)
    sixth_ABC = solve_ivp(ecoNetwork, (96,107.9), starting_values_2013,  t_eval = t_4, args=(A, r), method = 'RK23')
    # take those values and re-run for another year (2014), adding forcings
    starting_2014 = sixth_ABC.y[0:10, 35:36].flatten()
    starting_values_2014 = starting_2014
    starting_values_2014[0] = 0.43
    starting_values_2014[1] = 2.38
    starting_values_2014[3] = 4.98
    starting_values_2014[5] = 1
    starting_values_2014[7] = 0.9
    t_5 = np.linspace(108, 119.9, 36)
    seventh_ABC = solve_ivp(ecoNetwork, (108,119.9), starting_values_2014,  t_eval = t_5, args=(A, r), method = 'RK23')
    # set numbers for 2015
    starting_values_2015 = seventh_ABC.y[0:10, 35:36].flatten()
    starting_values_2015[0] = 0.43
    starting_values_2015[1] = 2.38
    starting_values_2015[3] = 2.01
    starting_values_2015[5] = 1
    starting_values_2015[7] = 0.9


    ## 2015 ## 

    # now go month by month: March & April 2015
    t_March2015 = np.linspace(120, 121.9, 6)
    March2015_ABC = solve_ivp(ecoNetwork, (120,121.9), starting_values_2015,  t_eval = t_March2015, args=(A, r), method = 'RK23')
    last_values_April2015 = March2015_ABC.y[0:10, 5:6].flatten()
    # pigs had one death in April
    starting_values_May2015 = last_values_April2015
    starting_values_May2015[7] = 1.1
    # May 2015
    t_May2015 = np.linspace(122, 122.9, 3)
    May2015_ABC = solve_ivp(ecoNetwork, (122,122.9), starting_values_May2015,  t_eval = t_May2015, args=(A, r), method = 'RK23')
    last_values_May2015 = May2015_ABC.y[0:10, 2:3].flatten()
    # pigs had 8 deaths in May
    starting_values_June2015 = last_values_May2015
    starting_values_June2015[7] = 0.7
    # June 2015
    t_June2015 = np.linspace(123, 123.9, 3)
    June2015_ABC = solve_ivp(ecoNetwork, (123,123.9), starting_values_June2015,  t_eval = t_June2015, args=(A, r), method = 'RK23')
    last_values_June2015 = June2015_ABC.y[0:10, 2:3].flatten()
    # 5 cow deaths in June 2015 (net 5 cows)
    starting_values_July2015 = last_values_June2015
    starting_values_July2015[3] = 2.43
    # July & Aug 2015
    t_July2015 = np.linspace(124, 125.9, 6)
    July2015_ABC = solve_ivp(ecoNetwork, (124,125.9), starting_values_July2015,  t_eval = t_July2015, args=(A, r), method = 'RK23')
    starting_values_Sept2015 = July2015_ABC.y[0:10, 5:6].flatten()
    # 2 fallow deer deaths in August
    starting_values_Sept2015[1] = starting_values_Sept2015[1] - 0.048
    # Sept 2015
    t_Sept2015 = np.linspace(126, 126.9, 3)
    Sept2015_ABC = solve_ivp(ecoNetwork, (126,126.9), starting_values_Sept2015,  t_eval = t_Sept2015, args=(A, r), method = 'RK23')
    starting_values_Oct2015 = Sept2015_ABC.y[0:10, 2:3].flatten()
    # 2 fallow deer killed in September
    starting_values_Oct2015[1] = starting_values_Oct2015[1]-0.048
    # Oct 2015
    t_Oct2015 = np.linspace(127, 127.9, 3)
    Oct2015_ABC = solve_ivp(ecoNetwork, (127,127.9), starting_values_Oct2015,  t_eval = t_Oct2015, args=(A, r), method = 'RK23')
    starting_values_Nov2015 = Oct2015_ABC.y[0:10, 2:3].flatten()
    # 3 fallow deer culled, 39 cows culled
    starting_values_Nov2015[1] = starting_values_Nov2015[1] - 0.072
    starting_values_Nov2015[3] = 1.72
    # Nov 2015
    t_Nov2015 = np.linspace(128, 128.9, 3)
    Nov2015_ABC = solve_ivp(ecoNetwork, (128,128.9), starting_values_Nov2015,  t_eval = t_Nov2015, args=(A, r), method = 'RK23')
    starting_values_Dec2015 = Nov2015_ABC.y[0:10, 2:3].flatten()
    # 7 fallow deer culled, 1 pig culled in November
    starting_values_Dec2015[1] = starting_values_Dec2015[1] - 0.17
    starting_values_Dec2015[7] = 0.65
    # Dec 2015
    t_Dec2015 = np.linspace(129, 129.9, 3)
    Dec2015_ABC = solve_ivp(ecoNetwork, (129,129.9), starting_values_Dec2015,  t_eval = t_Dec2015, args=(A, r), method = 'RK23')
    starting_values_Jan2016 = Dec2015_ABC.y[0:10, 2:3].flatten()
    # 6 fallow deer culled, 5 pigs moved offsite in Dec 2015
    starting_values_Jan2016[1] = starting_values_Jan2016[1] - 0.14
    starting_values_Jan2016[3] = 1.6
    # Jan 2016
    t_Jan2016 = np.linspace(130, 130.9, 3)
    Jan2016_ABC = solve_ivp(ecoNetwork, (130,130.9), starting_values_Jan2016,  t_eval = t_Jan2016, args=(A, r), method = 'RK23')
    starting_values_Feb2016 = Jan2016_ABC.y[0:10, 2:3].flatten()
    # 7 fallow deer culled, 1 pig added, 4 pigs culled in Jan 2016
    starting_values_Feb2016[1] = starting_values_Feb2016[1] - 0.17
    starting_values_Feb2016[7] = 0.50
    # Feb 2016
    t_Feb2016 = np.linspace(131, 131.9, 3)
    Feb2016_ABC = solve_ivp(ecoNetwork, (131,131.9), starting_values_Feb2016,  t_eval = t_Feb2016, args=(A, r), method = 'RK23')
    last_values_Feb2016 = Feb2016_ABC.y[0:10, 2:3].flatten()
    # 10 fallow deer culled, 2 pigs killed in Feb 2016
    starting_values_March2016 = last_values_Feb2016
    starting_values_March2016[1] = starting_values_March2016[1] - 0.24
    starting_values_March2016[7] = 0.35



    ## 2016 ##

    # March 2016
    t_March2016 = np.linspace(132, 132.9, 3)
    March2016_ABC = solve_ivp(ecoNetwork, (132,132.9), starting_values_March2016,  t_eval = t_March2016, args=(A, r), method = 'RK23')
    last_values_March2016 = March2016_ABC.y[0:10, 2:3].flatten()
    # change values: 1 pony added, 3 pigs moved in and 4 moved out in March 2016
    starting_values_April2016 = last_values_March2016
    starting_values_April2016[0] = 0.48
    starting_values_April2016[7] = 0.45
    # April 2016
    t_Apr2016 = np.linspace(133, 133.9, 3)
    April2016_ABC = solve_ivp(ecoNetwork, (133,133.9), starting_values_April2016,  t_eval = t_Apr2016, args=(A, r), method = 'RK23')
    last_values_Apr2016 = April2016_ABC.y[0:10, 2:3].flatten()
    # 1 cattle moved on-site
    starting_values_May2016 = last_values_Apr2016
    starting_values_May2016[3] = 1.94
    # May 2016
    t_May2016 = np.linspace(134, 134.9, 3)
    May2016_ABC = solve_ivp(ecoNetwork, (134,134.9), starting_values_May2016,  t_eval = t_May2016, args=(A, r), method = 'RK23')
    last_values_May2016 = May2016_ABC.y[0:10, 2:3].flatten()
    # 2 cow deaths
    starting_values_June2016 = last_values_May2016
    starting_values_June2016[3] = 2.04
    # June 2016
    t_June2016 = np.linspace(135, 135.9, 3)
    June2016_ABC = solve_ivp(ecoNetwork, (135,135.9), starting_values_June2016,  t_eval = t_June2016, args=(A, r), method = 'RK23')
    last_values_June2016 = June2016_ABC.y[0:10, 2:3].flatten()
    starting_values_July2016 = last_values_June2016
    # 30 cows sold/moved off site, and 4 moved on-site
    starting_values_July2016[3] = 1.68
    # July 2016
    t_July2016 = np.linspace(136, 136.9, 3)
    July2016_ABC = solve_ivp(ecoNetwork, (136,136.9), starting_values_July2016,  t_eval = t_July2016, args=(A, r), method = 'RK23')
    starting_values_Aug2016 = July2016_ABC.y[0:10, 2:3].flatten()
    # 2 cows sold
    starting_values_Aug2016[3] = 1.64
    # Aug 2016
    t_Aug2016 = np.linspace(137, 137.9, 3)
    Aug2016_ABC = solve_ivp(ecoNetwork, (137,137.9), starting_values_Aug2016,  t_eval = t_Aug2016, args=(A, r), method = 'RK23')
    starting_values_Sep2016 = Aug2016_ABC.y[0:10, 2:3].flatten()
    # 5 fallow deer deaths
    starting_values_Sep2016[1] = starting_values_Sep2016[1] - 0.12
    # Sept 2016
    t_Sept2016 = np.linspace(138, 138.9, 3)
    Sept2016_ABC = solve_ivp(ecoNetwork, (138,138.9), starting_values_Sep2016,  t_eval = t_Sept2016, args=(A, r), method = 'RK23')
    last_values_Sept2016 = Sept2016_ABC.y[0:10, 2:3].flatten()
    # 9 cows sold, 19 born or moved onto site (unclear)
    starting_values_Oct2016 = last_values_Sept2016
    starting_values_Oct2016[3] = 1.83
    # Oct & Nov 2016
    t_Oct2016 = np.linspace(139, 140.9, 6)
    Oct2016_ABC = solve_ivp(ecoNetwork, (139,140.9), starting_values_Oct2016,  t_eval = t_Oct2016, args=(A, r), method = 'RK23')
    starting_values_Dec2016 = Oct2016_ABC.y[0:10, 5:6].flatten()
    # 3 fallow deaths, 5 cow sales
    starting_values_Dec2016[1] = starting_values_Dec2016[1] - 0.072
    starting_values_Dec2016[3] = 1.74
    # Dec 2016
    t_Dec2016 = np.linspace(141, 141.9, 3)
    Dec2016_ABC = solve_ivp(ecoNetwork, (141,141.9), starting_values_Dec2016,  t_eval = t_Dec2016, args=(A, r), method = 'RK23')
    starting_values_Jan2017 = Dec2016_ABC.y[0:10, 2:3].flatten()
    # 9 fallow deaths, 4 pig sales, 13 cow sales
    starting_values_Jan2017[1] = starting_values_Jan2017[1] - 0.22
    starting_values_Jan2017[3] = 1.49
    starting_values_Jan2017[7] = 0.65
    # Jan 2017
    t_Jan2017 = np.linspace(142, 142.9, 3)
    Jan2017_ABC = solve_ivp(ecoNetwork, (142,142.9), starting_values_Jan2017,  t_eval = t_Jan2017, args=(A, r), method = 'RK23')
    starting_values_Feb2017 = Jan2017_ABC.y[0:10, 2:3].flatten()
    # 4 pigs sold
    starting_values_Feb2017[7] = 0.45
    # Feb 2017
    t_Feb2017 = np.linspace(143, 143.9, 3)
    Feb2017_ABC = solve_ivp(ecoNetwork, (143,143.9), starting_values_Feb2017,  t_eval = t_Feb2017, args=(A, r), method = 'RK23')
    last_values_Feb2017 = Feb2017_ABC.y[0:10, 2:3].flatten()
    # 10 fallow deer, 2 pigs killed, and in 2017 they start with 14 red deer (not clear if culled) and one less pony
    starting_values_March2017 = last_values_Feb2017
    starting_values_March2017[0] = 0.43
    starting_values_March2017[1] = starting_values_March2017[1] - 0.24
    starting_values_March2017[5] = 1.08
    starting_values_March2017[7] = 0.35



    ## 2017 ##
    # March & April 2017
    t_March2017 = np.linspace(144, 145.9, 6)
    March2017_ABC = solve_ivp(ecoNetwork, (144,145.9), starting_values_March2017,  t_eval = t_March2017, args=(A, r), method = 'RK23')
    last_values_Apr2017 = March2017_ABC.y[0:10, 5:6].flatten()
    # 3 cows added moved on-site
    starting_values_May2017 = last_values_Apr2017
    starting_values_May2017[3] = 1.89
    # May & June 2017
    t_May2017 = np.linspace(146, 147.9, 6)
    May2017_ABC = solve_ivp(ecoNetwork, (146,147.9), starting_values_May2017,  t_eval = t_May2017, args=(A, r), method = 'RK23')
    last_values_May2017 = May2017_ABC.y[0:10, 2:3].flatten()
    last_values_June2017 = May2017_ABC.y[0:10, 5:6].flatten()
    starting_values_July2017 = last_values_June2017
    # 24 cows moved off-site and 3 moved on-site
    starting_values_July2017[3] = 1.77
    # July & Aug 2017
    t_July2017 = np.linspace(148, 149.9, 6)
    July2017_ABC = solve_ivp(ecoNetwork, (148,149.9), starting_values_July2017,  t_eval = t_July2017, args=(A, r), method = 'RK23')
    starting_values_Sept2017 = July2017_ABC.y[0:10, 5:6].flatten()
    # 16 fallow deer deaths
    starting_values_Sept2017[1] = starting_values_Sept2017[1] - 0.38
    # Sept 2017
    t_Sept2017 = np.linspace(150, 150.9, 3)
    Sept2017_ABC = solve_ivp(ecoNetwork, (150, 150.9), starting_values_Sept2017,  t_eval = t_Sept2017, args=(A, r), method = 'RK23')
    starting_values_Oct2017 = Sept2017_ABC.y[0:10, 2:3].flatten()
    # 5 fallow deaths, 24 cows sold and 3 moved off-site, and 23 moved on-site
    starting_values_Oct2017[1] = starting_values_Oct2017[1] - 0.12
    starting_values_Oct2017[3] = 1.70
    # Oct 2017
    t_Oct2017 = np.linspace(151, 151.9, 3)
    Oct2017_ABC = solve_ivp(ecoNetwork, (151,151.9), starting_values_Oct2017,  t_eval = t_Oct2017, args=(A, r), method = 'RK23')
    starting_values_Nov2017 = Oct2017_ABC.y[0:10, 2:3].flatten()
    # 4 fallow deaths, 2 cows moved off-site
    starting_values_Nov2017[1] = starting_values_Nov2017[1] - 0.096
    starting_values_Nov2017[3] = 1.66
    # Nov 2017
    t_Nov2017 = np.linspace(152, 152.9, 3)
    Nov2017_ABC = solve_ivp(ecoNetwork, (152,152.9), starting_values_Nov2017,  t_eval = t_Nov2017, args=(A, r), method = 'RK23')
    starting_values_Dec2017 = Nov2017_ABC.y[0:10, 2:3].flatten()
    # 2 fallow deer deaths
    starting_values_Dec2017[1] = starting_values_Dec2017[1] - 0.024
    # Dec 2018
    t_Dec2017 = np.linspace(153, 153.9, 3)
    Dec2017_ABC = solve_ivp(ecoNetwork, (153,153.9), starting_values_Dec2017,  t_eval = t_Dec2017, args=(A, r), method = 'RK23')
    starting_values_Jan2018 = Dec2017_ABC.y[0:10, 2:3].flatten()
    # 46 fallow deaths, 1 red deer death, 4 pig sales
    starting_values_Jan2018[1] = starting_values_Jan2018[1] - 1.1
    starting_values_Jan2018[5] = starting_values_Jan2018[5] - 0.08
    starting_values_Jan2018[7] = 0.9
    # Jan 2018
    t_Jan2018 = np.linspace(154, 154.9, 3)
    Jan2018_ABC = solve_ivp(ecoNetwork, (154,154.9), starting_values_Jan2018,  t_eval = t_Jan2018, args=(A, r), method = 'RK23')
    last_values_Jan2018 = Jan2018_ABC.y[0:10, 2:3].flatten()
    # 9 pigs sold
    starting_values_Feb2018 = last_values_Jan2018
    starting_values_Feb2018[7] = 0.55
    # Feb 2018
    t_Feb2018 = np.linspace(155, 155.9, 3)
    Feb2018_ABC = solve_ivp(ecoNetwork, (155,155.9), starting_values_Feb2018,  t_eval = t_Feb2018, args=(A, r), method = 'RK23')
    last_values_Feb2018 = Feb2018_ABC.y[0:10, 2:3].flatten()
    # 14 fallow deaths, 1 red deer death, ponies back to 9
    starting_values_March2018 = last_values_Feb2018
    starting_values_March2018[0] = 0.39
    starting_values_March2018[1] = starting_values_March2018[1] - 0.33
    starting_values_March2018[5] = starting_values_March2018[5] - 0.08


    ## 2018 ##
    
    # March & April 2018
    t_March2018 = np.linspace(156, 157.9, 6)
    March2018_ABC = solve_ivp(ecoNetwork, (156,157.9), starting_values_March2018,  t_eval = t_March2018, args=(A, r), method = 'RK23')
    last_values_Apr2018 = March2018_ABC.y[0:10, 5:6].flatten()
    # 1 cow moved on-site
    starting_values_May2018 = last_values_Apr2018
    starting_values_May2018[3] = 1.91
    # May & June 2018
    t_May2018 = np.linspace(158, 159.9, 6)
    May2018_ABC = solve_ivp(ecoNetwork, (158,159.9), starting_values_May2018,  t_eval = t_May2018, args=(A, r), method = 'RK23')
    last_values_May2018 = May2018_ABC.y[0:10, 2:3].flatten()
    last_values_June2018 = May2018_ABC.y[0:10, 5:6].flatten()
    starting_values_July2018 = last_values_June2018
    # 2 cows moved on-site, 22 cow deaths/moved off-site
    starting_values_July2018[3] = 1.94
    # July 2018
    t_July2018 = np.linspace(160, 160.9, 3)
    July2018_ABC = solve_ivp(ecoNetwork, (160,160.9), starting_values_July2018,  t_eval = t_July2018, args=(A, r), method = 'RK23')
    starting_values_Aug2018 = July2018_ABC.y[0:10, 2:3].flatten()
    # 1 red deer death, 1 pig death
    starting_values_Aug2018[5] = starting_values_Aug2018[5] - 0.077
    starting_values_Aug2018[7] = 1.1
    # Aug 2018
    t_Aug2018 = np.linspace(161, 161.9, 3)
    Aug2018_ABC = solve_ivp(ecoNetwork, (161,161.9), starting_values_Aug2018,  t_eval = t_Aug2018, args=(A, r), method = 'RK23')
    starting_values_Sept2018 = Aug2018_ABC.y[0:10, 2:3].flatten()
    # 1 red deer death, 15 fallow deer deaths, 9 pony transfers, 1 longhorn transfer
    starting_values_Sept2018[0] = 0
    starting_values_Sept2018[1] = starting_values_Sept2018[1] - 0.36
    starting_values_Sept2018[3] = 1.92
    starting_values_Sept2018[5] = starting_values_Sept2018[5] - 0.077
    # Sept 2018
    t_Sept2018 = np.linspace(162, 162.9, 3)
    Sept2018_ABC = solve_ivp(ecoNetwork, (162,162.9), starting_values_Sept2018,  t_eval = t_Sept2018, args=(A, r), method = 'RK23')
    starting_values_Oct2018 = Sept2018_ABC.y[0:10, 2:3].flatten()
    # 19 fallow deer deaths, 14 longhorns sold and 2 moved off-site, 20 longhorns moved on-site
    starting_values_Oct2018[1] = starting_values_Oct2018[1] - 0.45
    starting_values_Oct2018[3] = 2.00
    # Oct 2018
    t_Oct2018 = np.linspace(163, 163.9, 3)
    Oct2018_ABC = solve_ivp(ecoNetwork, (163,163.9), starting_values_Oct2018,  t_eval = t_Oct2018, args=(A, r), method = 'RK23')
    starting_values_Nov2018 = Oct2018_ABC.y[0:10, 2:3].flatten()
    # 4 fallow deaths, 1 tamworth death, 5 longhorn sales
    starting_values_Nov2018[1] = starting_values_Nov2018[1] - 0.096
    starting_values_Nov2018[3] = 1.91
    starting_values_Nov2018[7] = 1.05
    # Nov 2018
    t_Nov2018 = np.linspace(164, 164.9, 3)
    Nov2018_ABC = solve_ivp(ecoNetwork, (164,164.9), starting_values_Nov2018,  t_eval = t_Nov2018, args=(A, r), method = 'RK23')
    starting_values_Dec2018 = Nov2018_ABC.y[0:10, 2:3].flatten()
    # 8 longhorn sales, 12 pig sales
    starting_values_Dec2018[3] = 1.75
    starting_values_Dec2018[7] = 0.45
    # Dec 2018
    t_Dec2018 = np.linspace(165, 165.9, 3)
    Dec2018_ABC = solve_ivp(ecoNetwork, (165,165.9), starting_values_Dec2018,  t_eval = t_Dec2018, args=(A, r), method = 'RK23')
    starting_values_Jan2019 = Dec2018_ABC.y[0:10, 2:3].flatten()
    # 1 red deer death, 19 fallow deaths, 5 cows sold and 1 cow moved on-site
    starting_values_Jan2019[1] = starting_values_Jan2019[1] - 0.45
    starting_values_Jan2019[5] = starting_values_Jan2019[5] - 0.077
    starting_values_Jan2019[3] = 1.68
    # Jan & Feb 2019
    t_Jan2019 = np.linspace(166, 167.9, 6)
    Jan2019_ABC = solve_ivp(ecoNetwork, (166,167.9), starting_values_Jan2019,  t_eval = t_Jan2019, args=(A, r), method = 'RK23')
    starting_values_March2019 = Jan2019_ABC.y[0:10, 5:6].flatten()
    # 1 cow sold
    starting_values_March2019[3] = 1.64



    ## 2019 ##
    
    # March 2019
    t_March2019 = np.linspace(168, 168.9, 3)
    March2019_ABC = solve_ivp(ecoNetwork, (168,168.9), starting_values_March2019,  t_eval = t_March2019, args=(A, r), method = 'RK23')
    starting_values_April2019 = March2019_ABC.y[0:10, 2:3].flatten()
    # 7 red deer and 7 fallow deer culled
    starting_values_April2019[3] = starting_values_April2019[3] - 0.17
    starting_values_April2019[5] = starting_values_April2019[5] - 0.54
    # April 2019
    t_April2019 = np.linspace(169, 169.9, 3)
    April2019_ABC = solve_ivp(ecoNetwork, (169,169.9), starting_values_April2019,  t_eval = t_April2019, args=(A, r), method = 'RK23')
    last_values_April2019 = April2019_ABC.y[0:10, 2:3].flatten()
    # 1 pig sold
    starting_values_May2019 = last_values_April2019
    starting_values_May2019[7] = 0.4
    # May & June 2019
    t_May2019 = np.linspace(170, 171.9, 6)
    May2019_ABC = solve_ivp(ecoNetwork, (170,171.9), starting_values_May2019,  t_eval = t_May2019, args=(A, r), method = 'RK23')
    last_values_May2019 = May2019_ABC.y[0:10, 2:3].flatten()
    last_values_June2019 = May2019_ABC.y[0:10, 5:6].flatten()
    starting_values_July2019 = last_values_June2019
    # 28 longhorns moved off-sites
    starting_values_July2019[3] = 1.68
    # July 2019
    t_July2019 = np.linspace(172, 172.9, 3)
    July2019_ABC = solve_ivp(ecoNetwork, (172,172.9), starting_values_July2019,  t_eval = t_July2019, args=(A, r), method = 'RK23')
    last_values_July2019 = July2019_ABC.y[0:10, 2:3].flatten()
    # 26 pigs sold, 3 longhorns sold, 5 longhorns moved off-site
    starting_values_Aug2019 = last_values_July2019
    starting_values_Aug2019[3] = 1.72
    starting_values_Aug2019[7] = 0.45
    # Aug & Sept 2019
    t_Aug2019 = np.linspace(173, 174.9, 6)
    Aug2019_ABC = solve_ivp(ecoNetwork, (173,174.9), starting_values_Aug2019,  t_eval = t_Aug2019, args=(A, r), method = 'RK23')
    starting_values_Oct2019 = Aug2019_ABC.y[0:10, 5:6].flatten()
    # 15 fallow deaths, 19 cows sold and 4 moved off-site, 25 moved on-site
    starting_values_Oct2019[1] = starting_values_Oct2019[1] - 0.36
    starting_values_Oct2019[3] = 1.75
    # Oct 2019
    t_Oct2019 = np.linspace(175, 175.9, 3)
    Oct2019_ABC = solve_ivp(ecoNetwork, (175,175.9), starting_values_Oct2019,  t_eval = t_Oct2019, args=(A, r), method = 'RK23')
    starting_values_Nov2019 = Oct2019_ABC.y[0:10, 2:3].flatten()
    # 5 cows moved off-site
    starting_values_Nov2019[3] = 1.66
    # Nov 2019
    t_Nov2019 = np.linspace(176, 176.9, 3)
    Nov2019_ABC = solve_ivp(ecoNetwork, (176,176.9), starting_values_Nov2019,  t_eval = t_Nov2019, args=(A, r), method = 'RK23')
    starting_values_Dec2019 = Nov2019_ABC.y[0:10, 2:3].flatten()
    # 1 cow death, 7 fallow deaths, 3 red deer deaths
    starting_values_Dec2019[1] = starting_values_Dec2019[1] - 0.17
    starting_values_Dec2019[3] = 1.64
    starting_values_Dec2019[5] = starting_values_Dec2019[5] - 0.23
    # Dec 2019
    t_Dec2019 = np.linspace(177, 177.9, 3)
    Dec2019_ABC = solve_ivp(ecoNetwork, (177,177.9), starting_values_Dec2019,  t_eval = t_Dec2019, args=(A, r), method = 'RK23')
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
    Feb2020_ABC = solve_ivp(ecoNetwork, (179,179.9), starting_values_Feb2020,  t_eval = t_Feb2020, args=(A, r), method = 'RK23')
    starting_values_March2020 = Feb2020_ABC.y[0:10, 2:3].flatten()
    # 2 pigs sold, 12 fallow deers killed, 2 reds killed, 1 cow moved off-site
    starting_values_March2020[1] = starting_values_March2020[1] - 0.29
    starting_values_March2020[3] = 1.49
    starting_values_March2020[5] = starting_values_March2020[5] - 0.15
    starting_values_March2020[7] = 0.4


    ## 2020 ##
    # March 2020
    t_March2020 = np.linspace(180, 180.9, 3)
    March2020_ABC = solve_ivp(ecoNetwork, (180,180.9), starting_values_March2020,  t_eval = t_March2020, args=(A, r), method = 'RK23')
    last_values_March2020 = March2020_ABC.y[0:10, 2:3].flatten()
    # 1 pig death, 15 ponies added, 1 cow sold, 3 cows moved on-site
    starting_values_April2020 = last_values_March2020
    starting_values_April2020[0] = 0.65
    starting_values_April2020[3] = 1.53
    starting_values_April2020[7] = 0.35
    # April & May 2020
    t_April2020 = np.linspace(181, 183, 6)
    April2020_ABC = solve_ivp(ecoNetwork, (181, 183), starting_values_April2020,  t_eval = t_April2020, args=(A, r), method = 'RK23')
    last_values_May2020 = April2020_ABC.y[0:10, 5:6].flatten()

    # concatenate & append all the runs
    combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, March2015_ABC.y, May2015_ABC.y, June2015_ABC.y,  July2015_ABC.y, Sept2015_ABC.y, Oct2015_ABC.y, Nov2015_ABC.y, Dec2015_ABC.y, Jan2016_ABC.y, Feb2016_ABC.y, March2016_ABC.y, April2016_ABC.y, May2016_ABC.y, June2016_ABC.y, July2016_ABC.y, Aug2016_ABC.y, Sept2016_ABC.y, Oct2016_ABC.y, Dec2016_ABC.y, Jan2017_ABC.y, Feb2017_ABC.y, March2017_ABC.y, May2017_ABC.y, July2017_ABC.y, Sept2017_ABC.y, Oct2017_ABC.y, Nov2017_ABC.y, Dec2017_ABC.y, Jan2018_ABC.y, Feb2018_ABC.y, March2018_ABC.y, May2018_ABC.y, July2018_ABC.y, Aug2018_ABC.y, Sept2018_ABC.y, Oct2018_ABC.y, Nov2018_ABC.y, Dec2018_ABC.y, Jan2019_ABC.y, March2019_ABC.y, April2019_ABC.y, May2019_ABC.y, July2019_ABC.y, Aug2019_ABC.y, Oct2019_ABC.y, Nov2019_ABC.y, Dec2019_ABC.y, Jan2020_ABC.y, Feb2020_ABC.y, March2020_ABC.y, April2020_ABC.y))
    combined_times = np.hstack((second_ABC.t, third_ABC.t, fourth_ABC.t, fifth_ABC.t, sixth_ABC.t, seventh_ABC.t, March2015_ABC.t, May2015_ABC.t, June2015_ABC.t,  July2015_ABC.t, Sept2015_ABC.t, Oct2015_ABC.t, Nov2015_ABC.t, Dec2015_ABC.t, Jan2016_ABC.t, Feb2016_ABC.t, March2016_ABC.t, April2016_ABC.t, May2016_ABC.t, June2016_ABC.t, July2016_ABC.t, Aug2016_ABC.t, Sept2016_ABC.t, Oct2016_ABC.t, Dec2016_ABC.t, Jan2017_ABC.t, Feb2017_ABC.t, March2017_ABC.t, May2017_ABC.t, July2017_ABC.t, Sept2017_ABC.t, Oct2017_ABC.t, Nov2017_ABC.t, Dec2017_ABC.t, Jan2018_ABC.t, Feb2018_ABC.t, March2018_ABC.t, May2018_ABC.t, July2018_ABC.t, Aug2018_ABC.t, Sept2018_ABC.t, Oct2018_ABC.t, Nov2018_ABC.t, Dec2018_ABC.t, Jan2019_ABC.t, March2019_ABC.t, April2019_ABC.t, May2019_ABC.t, July2019_ABC.t, Aug2019_ABC.t, Oct2019_ABC.t, Nov2019_ABC.t, Dec2019_ABC.t, Jan2020_ABC.t, Feb2020_ABC.t, March2020_ABC.t, April2020_ABC.t))
    
    # reshape the outputs
    y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 405).transpose(),1)))
    y_2 = pd.DataFrame(data=y_2, columns=species)
    y_2['time'] = combined_times
    
    # choose the final year (we want to compare the final year to the middle of the filters)
    last_year_1 = y.loc[y['time'] == 47.9]
    last_year_1 = last_year_1.drop('time', axis=1).values.flatten()
    last_year_2 = y_2.loc[y_2['time'] == 183]
    last_year_2 = last_year_2.drop('time', axis=1).values.flatten()

    result = ( 
        # in April 2015, cows had 8 births and pigs had 5 births (net 1 pig), ponies the same
        (((starting_values_May2015[3]-2.17)**2)/2.17) + (((starting_values_May2015[7]-1.15)**2)/1.15) + (((starting_values_May2015[0]-0.43)**2)/0.43) +
        # in May 2015, there were 14 births for cows, ponies the same
        (((starting_values_June2015[3]-2.43)**2)/2.43) + (((starting_values_June2015[0]-0.43)**2)/0.43) +
        # in June 2015, cows had 5 births (net 5 cows), ponies the same
        (((last_values_June2015[3]-2.53)**2)/2.53) + (((last_values_June2015[0]-0.43)**2)/0.43) +
        # in Feb 2016, there were still 10 ponies (want to keep them from falling too much)
        (((last_values_Feb2016[0]-0.43)**2)/0.43) +
        # in March 2016, there were 140 fallow deer, 26 red deer, and 2 pigs were born (add 2, pre-supplement/cull)
        (((last_values_March2016[1]-3.3)**2)/3.3) + (((last_values_March2016[5]-2)**2)/2) + (((last_values_March2016[7]-0.5)**2)/0.5) +
        # in April 2016, there were 16 cattle births (+16 pre-moving 1 on-site), ponies the same
        (((last_values_Apr2016[3]-1.92)**2)/1.92) + (((last_values_Apr2016[0]-0.48)**2)/0.48) +
        # in May 2016, there were 8 pig births and 7 cow births (add 7 before culling 2)
        (((last_values_May2016[3]-2.08)**2)/2.08) + (((last_values_May2016[7]-0.85)**2)/0.85) +
        # In June 2016, there were 7 cow births (pre-culls), ponies the same
        (((last_values_June2016[3]-2.17)**2)/2.17) + (((last_values_June2016[0]-0.48)**2)/0.48) +
        # in Sept 2016, 19 cows were born (unclear - maybe supplemented)
        (((last_values_Sept2016[3]-2.19)**2)/2.19) + 
        # in Feb 2017, there were still 11 ponies 
        (((last_values_Feb2017[0]-0.48)**2)/0.48) +
        # in March 2017, there were 165 fallow deer
        (((starting_values_March2017[1]-3.93)**2)/3.93) +
        # in April 2017,  there were 15 pig births and 18 cow births (pre cow supplement), still 10 ponies
        (((last_values_Apr2017[3]-1.83)**2)/1.83) +  (((last_values_Apr2017[7]-1.1)**2)/1.1) + (((last_values_Apr2017[0]-0.43)**2)/0.43) +
        # In May 2017, there were 9 cow births
        (((last_values_May2017[3]-2.06)**2)/2.06) +  
        # in June 2017, 6 cows were born (pre 24 moving off site and 3 moving on-site), still 10 ponies
        (((last_values_June2017[3]-2.17)**2)/2.17) + (((last_values_June2017[0]-0.43)**2)/0.43) +
        # in Jan 2018, 2 pigs were born (pre 9 pigs being sold), still 10 ponies
        (((last_values_Jan2018[7]-1)**2)/1) + (((last_values_Jan2018[0]-0.43)**2)/0.43) +
        # in March 2018, there are 24 red deer and 251 fallow and 5 pig births
        (((last_values_Feb2018[5]-1.85)**2)/1.85) + (((last_values_Feb2018[1]-5.98)**2)/5.98) + (((last_values_Feb2018[7]-0.8)**2)/0.8) +
        # in April 2018, there were 12 cow births (pre 1 being moved on-site), still 9 ponies
        (((last_values_Apr2018[3]-1.89)**2)/1.89) + (((last_values_Apr2018[0]-0.39)**2)/0.39) +
        # in May 2018, there were 16 cow births and 7 pig births
        (((last_values_May2018[3]-2.21)**2)/2.21) + (((last_values_May2018[7]-1.15)**2)/1.15) +
        # in June 2018, there were six cow births (pre- 2 moving on-site and 22 sales), still 9 ponies
        (((last_values_June2018[3]-2.32)**2)/2.32) + (((last_values_June2018[0]-0.39)**2)/0.39) +
        # in March 2019, there were 278 fallow deer and 37 red deer
        (((starting_values_March2019[1]-6.79)**2)/6.79) + (((starting_values_March2019[5]-2.85)**2)/2.85) +
        # in April 2019, there were 14 longhorn births
        (((last_values_April2019[3]-1.91)**2)/1.91) + 
        # in May 2019, there were 9 longhorn births
        (((last_values_May2019[3]-2.1)**2)/2.1) + 
        # in June 2019, there were 7 longhorn births (pre 28 being moved off-site)
        (((last_values_June2019[3]-2.21)**2)/2.21) + 
        # in July 2019, there were 28 pig births (pre 26 being sold)
        (((last_values_July2019[7]-1.8)**2)/1.8) +
        # in March 2020, there were 35 red deer and 247 fallow deer
        (((last_values_March2020[5]-2.7)**2)/2.7) + (((last_values_March2020[1]-5.88)**2)/5.88) +
        # in May 2020, there were 12 pig births
        (((last_values_May2020[7]-0.95)**2)/0.95) +             
        # 2005 filtering conditions for all nodes
        (((last_year_1[2]-0.97)**2)/0.97) +  (((last_year_1[4]-1.4)**2)/1.4) + (((last_year_1[6]-2.2)**2)/2.2) + 
        (((last_year_1[8]-10)**2)/10) + (((last_year_1[9]-1.2)**2)/1.2) + 
        # End of May 2020 filtering conditions for all nodes - we don't know how many fallow & red grew to (next survey March 2021). & there were still 15 ponies
        (((last_year_2[0]-0.65)**2)/0.65) + (((last_year_2[2]-0.73)**2)/0.73) + 
        (((last_year_2[4]-2)**2)/2) + (((last_year_2[6]-4.2)**2)/4.2) + 
        (((last_year_2[8]-26.6)**2)/26.6) + (((last_year_2[9]-1.36)**2)/1.36)
        )
    if result < 100:
        print(last_year_2, '\n' "r",result)
    return (result)


# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.97            1.4        2.2              10              1.2

# [exmoor pony    fallow    'arableGrass',  longhorn, orgCarb    red deer   'roeDeer',tamworthPig,  'thornyScrub','woodland'])
#    0.65         > 5.88         0.73       1.53        2        > 2.7         4.2       0.95           26.6          1.36


growth_bds = ((0.3,1),(0.2,0.8),(0.025,0.05))

interactionbds = (
    # exmoor pony; these have no growth (no stallions)
    (-0.1,0),
    # fallow deer
    (-0.1,-0),(0.1,1),(0,0.1),(0,1),
    # grassland parkland
    (-0.01,0),(-0.01,0),(-0.01,0),(-0.01,0),(-0.01,0),(-0.01,0),(-0.01,0),(-0.1,0),(-0.1,0),
    # longhorn cattle
    (0.1,1),(-0.1,-0.0),(0,0.1),(0,1),
    # organic carbon
    (0.001,0.1),(0.001,0.1),(0.01,0.1),(0.001,0.1),(-0.1,-0.08),(0.001,0.1),(0.001,0.01),(0.001,0.1),(0.001,0.1),(0.01,0.1),
    # red deer
    (0.1,1),(-0.1,0),(0,0.1),(0,1),
    # roe deer
    (0,1),(-0.1,0),(0,0.1),(0,1),
    # tamworth pig
    (0,1),(-0.1,0),(0,0.1),(0,1),
    # thorny scrubland
    (-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.01,-0.001),(-0.1,-0.01),
    # woodland
    (-0.001,-0.0001),(-0.001,-0.0001),(-0.001,-0.0001),(-0.001,-0.0001),(-0.001,-0.0001),(-0.001,-0.0001),(0.0001,0.0005),(-0.01,-0.007))


#                     (-0.78,-0.74),(-0.004,-0.002),(-0.002,-0.0005),(-0.008,-0.005),(-0.013,-0.01),(-0.016,-0.012),
#                     (0.05,0.095),(0.0035,0.01),(-0.1,-0.09),(0.0015,0.004),(0.0025,0.007),(0.00015,0.00025),(0.05,0.095),
#                     (-0.07,-0.03),(-0.04,-0.01),(-0.05,-0.02),(-0.0019,-0.0017),(-0.018,-0.01),
#                     (-0.0005,-0.0003),(-0.0003,-0.0001),(-0.0004,-0.0002),(0.0001,0.00035),(-0.009,-0.007)
# )


# combine them into one dataframe
bds =  growth_bds + interactionbds

optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 1000)
# print(optimization, file=open("final_optimizationOutput.txt", "w"))
print(optimization)

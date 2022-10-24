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


# PS1  - unstable
significant_variables = ["grasslandParkland","thornyScrub","woodland",
"fallow_fallow","fallow_grass","fallow_scrub","fallow_wood", "grass_fallow",
"grass_grass","grass_scrub","grass_wood","cattle_grass","cattle_cattle","cattle_scrub","cattle_wood",
"red_grass","red_red","red_scrub","red_wood","roe_grass","roe_roe","scrub_cattle","scrub_roe",
"scrub_scrub","scrub_wood","wood_scrub","wood_wood"]

# PS2
# significant_variables = ["grasslandParkland","thornyScrub","woodland",
#     "fallow_fallow","fallow_scrub","fallow_wood",
#     "grass_grass","grass_scrub","cattle_cattle","cattle_scrub",
#     "red_grass","red_red","red_scrub","red_wood", "pig_grass","pig_pig","pig_scrub","pig_wood",
#     "scrub_scrub","scrub_wood","wood_scrub","wood_wood"]

#PS1 with stability
# significant_variables = ["grasslandParkland","thornyScrub","woodland",
# "fallow_fallow","fallow_grass","fallow_scrub","grass_grass","grass_scrub","grass_wood",
# "cattle_cattle","red_grass","red_red","red_scrub","red_wood", "roe_grass","roe_roe","roe_scrub",
# "pig_scrub","scrub_fallow","scrub_pig",
# "scrub_scrub","scrub_wood","wood_fallow","wood_roe","wood_scrub","wood_wood"]


# PS2 with stability
# significant_variables = ["thornyScrub","woodland",
# "fallow_fallow","grass_grass","grass_scrub","grass_wood",
# "cattle_grass","cattle_cattle","red_red","red_wood", "roe_roe","roe_scrub",
# "pig_pig","pig_wood","scrub_scrub","scrub_wood","wood_scrub","wood_wood"]


# significant_variables = ["grassland_growth","scrub_growth","woodland_growth","red_scrub"]


def ecoNetwork(t, X, A, r):
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))


def run_model():
    # open the accepted parameters
    # all_parameters = pd.read_csv('all_parameters_ps1_unstable.csv')
    all_parameters = pd.read_csv('all_parameters_ps1_unstable.csv')

    accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]

    species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']
    accepted_simulations = 2970 #ps1 - unstable
    # accepted_simulations = 3651 #ps2
    # accepted_simulations = 285 #ps1 - stable
    # accepted_simulations = 118 #ps2 - stable

    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]
    # growth rates
    growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(accepted_simulations, len(species)), columns = species)
    r_accepted = growthRates_2.to_numpy()
    # interaction matrices 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID','accepted?',"Unnamed: 0.1"], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    A_accepted = interaction_strength_2.to_numpy()

    sensitivity_results_list = []
    perc_numbers=[]
    parameter_names=[]
    perc_aboveBelow = [-0.5, -0.25, -0.1, -0.05,-0.01, 0, 0.01, 0.05, 0.1, 0.25, 0.5]

    # loop through each accepted parameter set 
    for parameter in significant_variables:
        for temporary_r, temporary_A in zip(r_accepted, np.array_split(A_accepted,accepted_simulations)):
            for perc in perc_aboveBelow:
                r = temporary_r.copy()
                A = temporary_A.copy()
            # and each percentage above/below
                if parameter == "grassland_growth": r[2] = temporary_r[2] + (temporary_r[2] * perc)
                if parameter == "scrub_growth": r[7] = temporary_r[7] + (temporary_r[7] * perc)
                if parameter == "woodland_growth": r[8] = temporary_r[8] + (temporary_r[8] * perc)
                # and interactions
                if parameter == "fallow_fallow":  A[1,1] = temporary_A[1,1] + (temporary_A[1,1] * perc)
                if parameter == "fallow_grass":  A_accepted[1,2] = temporary_A[1,2] + (temporary_A[1,2] * perc)
                if parameter == "fallow_scrub":  A[1,7] = temporary_A[1,7] + (temporary_A[1,7] * perc)
                if parameter == "fallow_wood":  A[1,8] = temporary_A[1,8] + (temporary_A[1,8] * perc)
                if parameter == "grass_grass":  A[2,2] = temporary_A[2,2] + (temporary_A[2,2] * perc)
                if parameter == "grass_scrub":  A[2,7] = temporary_A[2,7] + (temporary_A[2,7] * perc)
                if parameter == "grass_wood":  A[2,8] = temporary_A[2,8] + (temporary_A[2,8] * perc)
                if parameter == "cattle_grass":  A[3,2] = temporary_A[3,2] + (temporary_A[3,2] * perc)
                if parameter == "cattle_cattle":  A[3,3] = temporary_A[3,3] + (temporary_A[3,3] * perc)
                if parameter == "cattle_scrub":  A[3,7] = temporary_A[3,7] + (temporary_A[3,7] * perc)
                if parameter == "cattle_wood":  A[3,8] = temporary_A[3,8] + (temporary_A[3,8] * perc)
                if parameter == "red_grass":  A[4,2] = temporary_A[4,2] + (temporary_A[4,2] * perc)
                if parameter == "red_red":  A[4,4] = temporary_A[4,4] + (temporary_A[4,4] * perc)
                if parameter == "red_scrub":  A[4,7] = temporary_A[4,7] + (temporary_A[4,7] * perc)
                if parameter == "red_wood":  A[4,8] = temporary_A[4,8] + (temporary_A[4,8] * perc)
                if parameter == "roe_grass":  A[5,2] = temporary_A[5,2] + (temporary_A[5,2] * perc)
                if parameter == "scrub_wood":  A[7,8] = temporary_A[7,8] + (temporary_A[7,8] * perc)
                if parameter == "wood_wood":  A[8,8] = temporary_A[8,8] + (temporary_A[8,8] * perc)

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
                # pretend bison were reintroduced (to estimate growth rate / interaction values)
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
                if (my_time.iloc[0]['woodland'] <= 3.7) & (my_time.iloc[0]['woodland'] >=3.3):
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

    merged_dfs.to_csv("sensitivity_table_ps1_unstable.csv")

    # plot it
    sns.lineplot(data=merged_dfs, x="Percentage", y="Filters", hue="Parameter Name",palette="Spectral")
    plt.title('Sensitivity test results for non-uniform parameters')
    plt.xlabel('Delta')
    plt.ylabel('Percentage of filters passed')
    plt.legend(title='Parameter names', ncol=2)
    plt.savefig('ks-test-all-parameters_ps1_unstable.png')

    plt.show()

run_model()
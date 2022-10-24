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
import sys


species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']
def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))

# first - forecast 50 years 
def run_model():
    all_parameters = pd.read_csv('all_parameters_ps1_stable.csv')
    # how many were accepted? 
    number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)
    # get the accepted parameters
    accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
    accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
    # select accepted interaction strengths
    interaction_strength_2 = accepted_parameters.drop(["Unnamed: 0.1",'X0', 'growth', 'ID', 'accepted?'], axis=1).dropna()
    A_reality = interaction_strength_2.to_numpy()
    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]

    all_runs_backcast = []
    all_times_backcast = []
    runs_left = number_accepted

    for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
        # print("remaining runs", runs_left)

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
        # forecast fifty years 
        years = 100
        previous_model = ABC_2020
        forecasting_models = {}
        forecasting_models_time = {}
        for year in range(years): 
            my_year = 2021 + (year-0.25)
            # get the initial conditions of the previous year
            previous_year = (2021 + year) - 1
            last_values = previous_model.y[0:9, 2:3].flatten()
            # my starting values
            starting_values = last_values.copy()
            starting_values[0] = 0.65
            starting_values[1] = 5.9
            starting_values[3] = 1.5
            starting_values[4] = 2.7
            starting_values[6] = 0.55
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
                        forecasting_models_time["ABC_2120"],
                                                    ))
        # all_times_forecast = np.append(all_times_forecast,forecasting_models_time)
        all_times_backcast = np.append(all_times_backcast,all_times_single)
        # note the runs left
        runs_left = runs_left +- 1

    # reshape the outputs
    final_outputs = (np.vstack(np.hsplit(all_runs_backcast.reshape(len(species)*int(number_accepted), 348).transpose(),number_accepted)))
    final_outputs = pd.DataFrame(data=final_outputs, columns=species)
    final_outputs['time'] = all_times_backcast
    final_outputs.to_csv("forecasting_ps1_stable.csv")

run_model()


# next, assess the counterfactual
def counterfactual():
    all_parameters = pd.read_csv('all_parameters_ps1_stable.csv')
    # how many were accepted? 
    number_accepted = (len(all_parameters.loc[(all_parameters['accepted?'] == "Accepted")]))/(len(species)*2)
    # get the accepted parameters
    accepted_parameters = all_parameters.loc[(all_parameters['accepted?'] == "Accepted")].iloc[:,1:]
    accepted_growth = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    accepted_r = pd.DataFrame(accepted_growth.values.reshape(int(number_accepted), len(species)), columns = species).to_numpy()
    # select accepted interaction strengths
    interaction_strength_2 = accepted_parameters.drop(["Unnamed: 0.1",'X0', 'growth', 'ID', 'accepted?'], axis=1).dropna()
    A_reality = interaction_strength_2.to_numpy()
    X0 = [0, 0, 1, 0, 0, 1, 0, 1, 1]
    all_runs = []
    all_times = []
    for r, A in zip(accepted_r, np.array_split(A_reality,number_accepted)):
        t_init = np.linspace(2005, 2120, 200)
        results = solve_ivp(ecoNetwork, (2005, 2120), X0,  t_eval = t_init, args=(A, r), method = 'RK23')
        # append results
        all_runs = np.append(all_runs, results.y)
        all_times = np.append(all_times, results.t) 
    # combine them
    combined_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*int(number_accepted), 200).transpose(),number_accepted)))
    combined_runs = pd.DataFrame(data=combined_runs, columns=species)
    combined_runs['time'] = all_times
    combined_runs.to_csv("counterfactual_ps1_stable.csv")

counterfactual()

# graph those two 
def graph_forecasting():
    forecasting = pd.read_csv("forecasting_ps1_stable.csv").iloc[:,1:]
    forecasting['runType'] = "forecasting"
    counterfactual = pd.read_csv("counterfactual_ps1_stable.csv").iloc[:,1:]
    counterfactual['runType'] = "counterfactual"
    # concat them
    final_results = pd.concat([forecasting, counterfactual])

    # put it in a dataframe
    y_values = final_results[["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland"]].values.flatten()
    species_list = np.tile(["exmoorPony", "fallowDeer", "grasslandParkland", "longhornCattle", "redDeer", "roeDeer", "tamworthPig","thornyScrub", "woodland"],len(final_results)) 
    indices = np.repeat(final_results['time'], 9)
    runType = np.repeat(final_results['runType'], 9)
    # here's the final dataframe
    final_results = pd.DataFrame(
        {'Abundance': y_values, 'Ecosystem Element': species_list, 'Time': indices, 'runType': runType})

    # calculate median 
    m = final_results.groupby(['Time', 'runType', 'Ecosystem Element'])[['Abundance']].apply(np.median)
    m.name = 'Median'
    final_results = final_results.join(m, on=['Time', 'runType', 'Ecosystem Element'])
    # calculate quantiles
    perc1 = final_results.groupby(['Time', 'runType', 'Ecosystem Element'])['Abundance'].quantile(.95)
    perc1.name = 'ninetyfivePerc'
    final_results = final_results.join(perc1, on=['Time', 'runType', 'Ecosystem Element'])
    perc2 = final_results.groupby(['Time', 'runType', 'Ecosystem Element'])['Abundance'].quantile(.05)
    perc2.name = "fivePerc"
    final_results = final_results.join(perc2, on=['Time','runType', 'Ecosystem Element'])
    final_df = final_results.reset_index(drop=True)
    
    # now graph it
    palette=['#db5f57', '#57d3db', '#57db5f','#5f57db', '#db57d3']
    f = sns.FacetGrid(final_df, col="Ecosystem Element", hue = "runType", palette = palette, col_wrap=4, sharey = False)
    f.map(sns.lineplot, 'Time', 'Median')
    f.map(sns.lineplot, 'Time', 'fivePerc')
    f.map(sns.lineplot, 'Time', 'ninetyfivePerc')
    for ax in f.axes.flat:
        ax.fill_between(ax.lines[2].get_xdata(),ax.lines[2].get_ydata(), ax.lines[4].get_ydata(),color="#db5f57", alpha =0.2)
        ax.fill_between(ax.lines[3].get_xdata(),ax.lines[3].get_ydata(), ax.lines[5].get_ydata(), color="#57d3db", alpha=0.2)
        ax.set_ylabel('Abundance')
        ax.set_xlabel('Time (Years)')

    # add subplot titles
    axes = f.axes.flatten()
    # fill between the quantiles
    axes[0].set_title("Exmoor pony")
    axes[1].set_title("Fallow deer")
    axes[2].set_title("Grassland")
    axes[3].set_title("Longhorn cattle")
    axes[4].set_title("Red deer")
    axes[5].set_title("Roe deer")
    axes[6].set_title("Tamworth pig")
    axes[7].set_title("Thorny scrub")
    axes[8].set_title("Woodland")

    # stop the plots from overlapping
    f.fig.suptitle('Forecasting vs. Counterfactual')
    plt.tight_layout()
    plt.legend(labels=['Forecasting current dynamics', 'Counterfactual'],bbox_to_anchor=(2.2, 0), loc='lower right', fontsize=12)
    plt.savefig('counterfactual_ps1_stable.png')
    plt.show()

graph_forecasting()
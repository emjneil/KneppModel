# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

# download packages
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string
import networkx as nx
import timeit
import seaborn as sns


##### ------------------- DEFINE THE FUNCTION ------------------------

# time the program
start = timeit.default_timer()

# define the number of simulations to try. Bode et al. ran a million
NUMBER_OF_SIMULATIONS = 500

def ecoNetwork(t, X, interaction_strength_chunk, rowContents_growth):
    # define new array to return
    output_array = []
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * X[outer_index]
        # second loop
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount2 = amount + (interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index]) 
        # append values to output_array
        output_array.append(amount2)
    # stop things from going to zero
    for i, (a,b) in enumerate(zip(X, output_array)):
        if a + b < 1e-8:
            X[i] = 0
            output_array[i] = 0
    # rescale habitat data
    totalHabitat = ((output_array[4] + X[4]) + (output_array[5] + X[5]) + (output_array[6] + X[6]))
    final_array = []
    # if habitats aren't all at zero, scale them to 4.5
    if totalHabitat != 0 or None:
        final_array = [output_array[0], output_array[1], output_array[2], output_array[3], (((output_array[4] + X[4])/totalHabitat) * 0.045) -X[4], (((output_array[5] + X[5])/totalHabitat) * 0.045) -X[5], (((output_array[6] + X[6])/totalHabitat) * 0.045) -X[6], output_array[7]]
    # otherwise return zero
    else:
        final_array = [output_array[0],output_array[1], output_array[2], output_array[3], 0, 0, 0, output_array[7]]
    # return array
    return final_array


# # # ---------------------- ODE #1: GENERATE INTERACTION MATRIX -------------------------

interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
# store species in a list
species = list(interactionMatrix_csv.columns.values)


# # # -------- GENERATE PARAMETERS ------ 

def generateInteractionMatrix():
    interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
    # store species in a list
    species = list(interactionMatrix_csv.columns.values)
    # remember the shape of the csv array
    interaction_length = len(interactionMatrix_csv)
    # pull the original sign of the interaction
    original_sign=[
                    -1,0,0,0,-1,-1,-1,0,
                    0,-1,0,0,-1,-1,-1,0,
                    0,0,-1,0,-1,-1,-1,0,
                    0,0,0,-1,-1,-1,-1,1,
                    1,1,1,1,-1,0,0,0, 
                    1,1,1,1,-1,-1,-1,0,
                    1,1,1,1,-1,1,-1,0,
                    0,0,0,-1,0,0,0,-1]
    # find the shape of original_sign
    shape = len(original_sign)
    # make lots of these arrays, half-to-double minimization outputs & consistent with sign
    iterStrength_list = np.array(
        [[np.random.uniform(interactionMatrix_csv.values.flatten()-(interactionMatrix_csv.values.flatten()*0.5), interactionMatrix_csv.values.flatten()+(interactionMatrix_csv.values.flatten()*0.5), (shape,)) * original_sign] for i in range(NUMBER_OF_SIMULATIONS)])
    # make sure they're still min/max -1 to 1
    iterStrength_list[iterStrength_list > 1] = 1
    iterStrength_list[iterStrength_list < -1] = -1
    iterStrength_list.shape = (NUMBER_OF_SIMULATIONS, interaction_length, interaction_length)
    # convert to multi-index so that species' headers/cols can be added
    names = ['runs', 'species', 'z']
    index = pd.MultiIndex.from_product([range(s) for s in iterStrength_list.shape], names=names)
    interaction_strength = pd.DataFrame({'iterStrength_list': iterStrength_list.flatten()}, index=index)['iterStrength_list']
    # re-shape so each run in NUMBER_OF_SIMULATIONS is its own column
    interaction_strength = interaction_strength.unstack(level='z').swaplevel(0, 1).sort_index()
    interaction_strength.index = interaction_strength.index.swaplevel(0, 1)
    interaction_strength = interaction_strength.sort_values(by=['runs', 'species'], ascending=[True, True])
    # add headers/columns
    interaction_strength.columns = species
    interaction_strength.index = species * NUMBER_OF_SIMULATIONS
    return interaction_strength


def generateGrowth():
    growthRates_csv = pd.read_csv('./growthRates.csv')
    # generate new dataframe with random uniform distribution between the values 
    growthRates = pd.DataFrame(np.random.uniform(low=growthRates_csv.values-(growthRates_csv.values*0.5),high=growthRates_csv.values+(growthRates_csv.values*0.5), size=(NUMBER_OF_SIMULATIONS, len(species))),columns=growthRates_csv.columns)
    return growthRates
    

def generateX0():
    initial_numbers_csv = pd.read_csv('./initial_numbers.csv')
    # generate new dataframe with random uniform distribution
    X0 = pd.DataFrame(np.random.uniform(low=initial_numbers_csv.values-(initial_numbers_csv*0.5),high=initial_numbers_csv.values+(initial_numbers_csv*0.5), size=(NUMBER_OF_SIMULATIONS, len(species))),columns=initial_numbers_csv.columns)
    # keep habitat types scaled to the total size (4.45km2 - scaled)
    sumHabitat = X0['arableGrass'] +  X0['thornyScrub'] + X0['woodland']
    X0['arableGrass'] = (X0['arableGrass']/sumHabitat) * 0.045
    X0['thornyScrub'] = (X0['thornyScrub']/sumHabitat) * 0.045
    X0['woodland'] = (X0['woodland']/sumHabitat) * 0.045
    return X0


# # # # --- MAKE NETWORK X VISUAL OF INTERACTION MATRIX --

# def networkVisual():
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

def runODE_1():
    # Define time points: first 10 years (2000-2009)
    t = np.linspace(0, 10, 50)
    all_runs = []
    all_parameters = []
    X0 = generateX0()
    growthRates = generateGrowth()
    interaction_strength = generateInteractionMatrix()
    # loop through each row of data
    for (rowNumber, rowContents_X0), (rowNumber, rowContents_growth), (interaction_strength_chunk) in zip(X0.iterrows(),growthRates.iterrows(),np.array_split(interaction_strength,NUMBER_OF_SIMULATIONS)):
        # concantenate the parameters
        X0_growth = pd.concat([rowContents_X0.rename('X0'), rowContents_growth.rename('growth')], axis = 1)
        parameters_used = pd.concat([X0_growth, interaction_strength_chunk])
        # run the model
        first_ABC = solve_ivp(ecoNetwork, (0, 10), rowContents_X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # first_ABC = integrate.odeint(ecoNetwork, rowContents_X0, t, args=(interaction_strength_chunk, rowContents_growth))        
        # append all the runs
        all_runs = np.append(all_runs, first_ABC.y)
        # append all the parameters
        all_parameters.append(parameters_used)
    # check the final runs
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 50).transpose(),NUMBER_OF_SIMULATIONS)))
    final_runs = pd.DataFrame(data=final_runs, columns=species)
    # append all the parameters to a dataframe
    all_parameters = pd.concat(all_parameters)
    # add ID to all_parameters
    all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])
    return final_runs,all_parameters, X0


# --------- FILTER OUT UNREALISTIC RUNS -----------

def filterRuns_1():
    final_runs,all_parameters,X0 = runODE_1()
    # add ID to dataframe
    IDs = np.arange(1,1 + NUMBER_OF_SIMULATIONS)
    final_runs['ID'] = np.repeat(IDs,50)
    # select only the last run (filter to the year 2009)
    accepted_year = final_runs.iloc[49::50, :]
    for_printing = accepted_year.dropna()
    # print that
    with pd.option_context('display.max_columns', None):
        print(for_printing*100)
    # add filtering criteria 
    accepted_simulations = accepted_year[
    (accepted_year['roeDeer'] <= 9/100) & (accepted_year['roeDeer'] >= 4.9/100) &
    (accepted_year['fox'] <= 2.2/100) & (accepted_year['fox'] >= 0.22/100) &
    (accepted_year['rabbits'] <= 2260/100) & (accepted_year['rabbits'] >= 1/100) &
    (accepted_year['arableGrass'] <= 4/100) & (accepted_year['arableGrass'] >= 3.1/100) &
    (accepted_year['woodland'] <=0.8/100) & (accepted_year['woodland'] >= 0.2/100) &
    (accepted_year['thornyScrub'] <= 0.9/100) & (accepted_year['thornyScrub'] >= 0.2/100)
    ]
    print(accepted_simulations.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # add accepted ID to original dataframe
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Yes', 'No')
    return accepted_parameters, accepted_simulations, final_runs




# # # # ---------------------- ODE #2: Years 2009-2018 -------------------------

def generateParameters2():
    accepted_parameters, accepted_simulations, final_runs = filterRuns_1()
    # # select growth rates 
    growthRates_2 = accepted_parameters.loc[accepted_parameters['growth'].notnull(), ['growth']]
    growthRates_2 = pd.DataFrame(growthRates_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    # # make the final runs of ODE #1 the initial conditions
    accepted_simulations = accepted_simulations.drop('ID', axis=1)
    accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']] = accepted_simulations.values.flatten()
    # # select X0 
    X0_2 = accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']]
    X0_2 = pd.DataFrame(X0_2.values.reshape(len(accepted_simulations), len(species)), columns = species)
    # # add reintroduced species
    X0_2.loc[:, 'largeHerb'] = [26.5/100 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [7.9/100 for i in X0_2.index]
    # # select interaction matrices part of the dataframes 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    # make the interactions with the reintroduced species random between -1 and 1 (depending on sign)
    # rows
    herbRows = interaction_strength_2.loc[interaction_strength_2['largeHerb'] > 0, 'largeHerb'] 
    interaction_strength_2.loc[interaction_strength_2['largeHerb'] > 0, 'largeHerb'] = [np.random.uniform(low=0, high=1) for i in herbRows.index]
    tamRows = interaction_strength_2.loc[interaction_strength_2['tamworthPig'] > 0, 'tamworthPig'] 
    interaction_strength_2.loc[interaction_strength_2['tamworthPig'] > 0, 'tamworthPig'] = [np.random.uniform(low=0, high=1) for i in tamRows.index]
    # columns
    interaction_strength_2.loc['largeHerb','thornyScrub']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['largeHerb','arableGrass']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['largeHerb','woodland']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['tamworthPig','thornyScrub']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['tamworthPig','arableGrass']  = [np.random.uniform(low=-1, high=0)]
    interaction_strength_2.loc['tamworthPig','woodland']  = [np.random.uniform(low=-1, high=0)]
    return growthRates_2, X0_2, interaction_strength_2, accepted_simulations, accepted_parameters, final_runs




# # # # # ------ SOLVE ODE #2: Pre-reintroductions (2009-2018) -------

def runODE_2():
    growthRates_2, X0_2, interaction_strength_2, accepted_simulations, accepted_parameters, final_runs  = generateParameters2()
    all_runs_2 = []
    all_parameters_2 = []
    t_2 = np.linspace(0, 1, 5)
    # loop through each row of data
    for (rowNumber, rowContents_X0), (rowNumber, rowContents_growth), (interaction_strength_chunk) in zip(X0_2.iterrows(),growthRates_2.iterrows(),np.array_split(interaction_strength_2,len(accepted_simulations))):
        # concantenate the parameters
        X0_growth_2 = pd.concat([rowContents_X0.rename('X0'), rowContents_growth.rename('growth')], axis = 1)
        parameters_used_2 = pd.concat([X0_growth_2, interaction_strength_chunk])
        # run the model for one year 2009-2010 (to account for herbivore numbers being manually controlled every year)
        second_ABC = solve_ivp(ecoNetwork, (0, 1), rowContents_X0,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2010 = second_ABC.y[0:13, 4:5]
        starting_values_2010 = starting_2010.flatten()
        starting_values_2010[0] = 41.8/100
        starting_values_2010[2] = 3.8/100
        # run the model for another year 2010-2011
        third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2011 = third_ABC.y[0:13, 4:5]
        starting_values_2011 = starting_2011.flatten()
        starting_values_2011[0] = 47.6/100
        starting_values_2011[2] = 4.9/100
        # run the model for 2011-2012
        fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2012 = fourth_ABC.y[0:13, 4:5]
        starting_values_2012 = starting_2012.flatten()
        starting_values_2012[0] = 55.3/100
        starting_values_2012[2] = 7.4/100
        # run the model for 2012-2013
        fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2013 = fifth_ABC.y[0:13, 4:5]
        starting_values_2013 = starting_2013.flatten()
        starting_values_2013[0] = 87/100
        starting_values_2013[2] = 1.3/100
        # run the model for 2011-2012
        sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2014 = sixth_ABC.y[0:13, 4:5]
        starting_values_2014 = starting_2014.flatten()
        starting_values_2014[0] = 52.4/100
        starting_values_2014[2] = 4/100
        # run the model for 2011-2012
        seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2015 = seventh_ABC.y[0:13, 4:5]
        starting_values_2015 = starting_2015.flatten()
        starting_values_2015[0] = 60.7/100
        starting_values_2015[2] = 2/100
        # run the model for 2011-2012
        eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2016 = eighth_ABC.y[0:13, 4:5]
        starting_values_2016 = starting_2016.flatten()
        starting_values_2016[0] = 63.6/100
        starting_values_2016[2] = 1.6/100
        # run the model for 2011-2012
        ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2016,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2017 = ninth_ABC.y[0:13, 4:5]
        starting_values_2017 = starting_2017.flatten()
        starting_values_2017[0] = np.random.uniform(low=26.5/100,high=87/100)
        starting_values_2017[2] = np.random.uniform(low=1.3/100,high=7.9/100)
        # run the model for 2011-2012
        tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2017,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # take those values and re-run for another year, adding forcings
        starting_2018 = tenth_ABC.y[0:13, 4:5]
        starting_values_2018 = starting_2018.flatten()
        starting_values_2018[0] = np.random.uniform(low=26.5/100,high=87/100)
        starting_values_2018[2] = np.random.uniform(low=1.3/100,high=7.9/100)
        # run the model for 2011-2012
        eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2018,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
        # concatenate & append all the runs
        combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y))
        # print(combined_runs)
        all_runs_2 = np.append(all_runs_2, combined_runs)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)
        
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 50).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(final_runs_2)
    # add ID to the dataframe
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,50)
    return final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs




# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = runODE_2()
    # select only the last run (filter to the year 2010); make sure these are in line with the new min/max X0 values (from the minimization), not the original X0 bounds
    accepted_year_2018 = final_runs_2.iloc[49::50, :]
    accepted_simulations_2018 = accepted_year_2018[(accepted_year_2018['roeDeer'] <= 18/100) & (accepted_year_2018['roeDeer'] >= 4.5/100) &
    (accepted_year_2018['fox'] <= 2.4/100) & (accepted_year_2018['fox'] >= 0.22/100) &
    (accepted_year_2018['rabbits'] <= 2260/100) & (accepted_year_2018['rabbits'] >= 1/100) &
    (accepted_year_2018['arableGrass'] <= 2.7/100) & (accepted_year_2018['arableGrass'] >= 2.5/100) &
    (accepted_year_2018['woodland'] <= 0.8/100) & (accepted_year_2018['woodland'] >= 0.3/100) &
    (accepted_year_2018['thornyScrub'] <= 1.6/100) & (accepted_year_2018['thornyScrub'] >= 0.9/100)
    ]
    print(accepted_simulations_2018.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2018 = accepted_simulations_2018[accepted_simulations_2018['ID'].isin(accepted_simulations_2018['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2018['ID']), 'Yes', 'No')
    return accepted_simulations_2018, accepted_parameters_2018, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs



# # # # # ----------------------------- PLOTTING POPULATIONS (2000-2010) ----------------------------- 

# make sure it's SD not SE 

def plotting():
    accepted_simulations_2018, accepted_parameters_2018, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = filterRuns_2()
    # extract accepted nodes from all dataframes
    accepted_shape1 = np.repeat(final_runs['accepted?'], len(species))
    accepted_shape2 = np.repeat(final_runs_2['accepted?'], len(species))
    # concatenate them
    accepted_shape = pd.concat([accepted_shape1, accepted_shape2])
    # extract the node values from all dataframes
    final_runs1 = final_runs.drop(['ID', 'accepted?'], axis=1)
    final_runs1 = final_runs1.values.flatten()
    final_runs2 = final_runs_2.drop(['ID', 'accepted?'], axis=1)
    final_runs2 = final_runs2.values.flatten()
    # concatenate them
    y_values = np.concatenate((final_runs1, final_runs2), axis=None)
    # we want species column to be spec1,spec2,spec3,spec4, etc.
    species_firstRun = np.tile(species, 50*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 50*len(accepted_simulations))
    species_list = np.concatenate((species_firstRun, species_secondRun), axis=None)
    # add a grouping variable to graph each run separately
    grouping1 = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    grouping_variable1 = np.repeat(grouping1,50*len(species))
    grouping2 = np.arange(NUMBER_OF_SIMULATIONS+2, NUMBER_OF_SIMULATIONS + 2 + len(accepted_simulations))
    grouping_variable2 = np.repeat(grouping2,50*len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping_variable1, grouping_variable2), axis=None)
    # years - we've looked at 19 so far (2000-2018)
    year = np.arange(1,11)
    year2 = np.arange(11,21)
    indices1 = np.repeat(year,len(species)*5)
    indices1 = np.tile(indices1, NUMBER_OF_SIMULATIONS)
    indices2 = np.repeat(year2,len(species)*5)
    indices2 = np.tile(indices2, len(accepted_simulations))
    indices = np.concatenate((indices1, indices2), axis=None)
    # put it in a dataframe
    example_df = pd.DataFrame(
        {'nodeValues': y_values*100, 'runNumber': grouping_variable, 'species': species_list, 'time': indices, 'runAccepted': accepted_shape})
    # plot
    sns.relplot(x="time", y="nodeValues",
                hue="runAccepted", col="species",
                height=2.5, aspect=.75, facet_kws=dict(sharey=False),
                kind="line", legend="full", col_wrap=4, data=example_df)
    # g = sns.FacetGrid(example_df, col='species', hue='runAccepted', sharey = False, height = 2.5, col_wrap = 3, margin_titles=True)
    # g.map(sns.lineplot, 'time', 'nodeValues')
    # g.add_legend()
    plt.show()


plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
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
NUMBER_OF_SIMULATIONS = 200

def ecoNetwork(t, X, interaction_strength_chunk, rowContents_growth):
    # define new array to return
    output_array = []
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * X[outer_index]
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount += interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index]
            if amount.item() >= 5:
                amount = None
                break
        # append values to output_array
        output_array.append(amount)
    # return array
    return output_array



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
    original_sign=[]
    for outer_index, outer_species in enumerate(species):
        for inner_index, inner_species in enumerate(species):
            # pull the sign
            new_sign = interactionMatrix_csv[inner_species][outer_species]
            original_sign.append(new_sign)
    # find the shape of original_sign
    shape = len(original_sign)
    # make lots of these arrays, keeping sign consistent with original_sign
    iterStrength_list = np.array(
        [[np.random.uniform(0, 1, (shape,)) * original_sign] for i in range(NUMBER_OF_SIMULATIONS)])
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
    growthRates = pd.DataFrame(np.random.uniform(low=growthRates_csv.iloc[0],high=growthRates_csv.iloc[1], size=(NUMBER_OF_SIMULATIONS, len(species))),columns=growthRates_csv.columns)
    return growthRates

def generateX0():
    initial_numbers_csv = pd.read_csv('./initial_numbers.csv')
    # generate new dataframe with random uniform distribution
    X0_raw = pd.DataFrame(np.random.uniform(low=initial_numbers_csv.iloc[0],high=initial_numbers_csv.iloc[1], size=(NUMBER_OF_SIMULATIONS, len(species))),columns=initial_numbers_csv.columns)
    # normalize each row of data to 0-1; divide by 200 to keep it consisntent between runs
    X0 = X0_raw.div(200, axis=0)
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
    t = np.linspace(0, 10, 100)
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
        first_ABC = solve_ivp(ecoNetwork, (0, 10), rowContents_X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
        # append all the runs
        all_runs = np.append(all_runs, first_ABC.y)
        # append all the parameters
        all_parameters.append(parameters_used)
    # check the final runs
    final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 100).transpose(),NUMBER_OF_SIMULATIONS)))
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
    final_runs['ID'] = np.repeat(IDs,100)
    # select only the last run (filter to the year 2009)
    accepted_year = final_runs.iloc[99::100, :]
    for_printing = accepted_year.dropna()
    # print that
    with pd.option_context('display.max_columns', None):
        print(for_printing)
        print(for_printing.shape)
    # add filtering criteria
    accepted_simulations = accepted_year[(accepted_year['roeDeer'] <= (X0['roeDeer'].max()*4)) & (accepted_year['roeDeer'] >= (X0['roeDeer'].min()))
    # (accepted_year['fox'] <= (X0['fox'].max())) & (accepted_year['fox'] >= (X0['fox'].min())) &
    # (accepted_year['rabbits'] <= (X0['rabbits'].max())) & (accepted_year['rabbits'] >= (X0['rabbits'].min())) &
    # (accepted_year['arableGrass'] <= (X0['arableGrass'].max())) & (accepted_year['arableGrass'] >= (X0['arableGrass'].min()*0.79))
    # (accepted_year['woodland'] <= (X0['woodland'].max()*1.57)) & (accepted_year['woodland'] >= (X0['woodland'].min()*2.4))
    # (accepted_year['thornyScrub'] <= (X0['thornyScrub'].max()*2)) & (accepted_year['thornyScrub'] >= (X0['thornyScrub'].min()))
    ]
    print(accepted_simulations.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]
    # add accepted ID to original dataframe
    final_runs['accepted?'] = np.where(final_runs['ID'].isin(accepted_parameters['ID']), 'Yes', 'No')
    return accepted_parameters, accepted_simulations, final_runs



# # # # ---------------------- ODE #2: Years 2009-2010 -------------------------

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
    # # add reintroduced species & divide by 200 to keep the scaling/normalization consistent
    X0_2.loc[:, 'largeHerb'] = [np.random.uniform(low=38.25, high=77.68)/200 for i in X0_2.index]
    X0_2.loc[:,'tamworthPig'] = [np.random.uniform(low=6.36, high=11.77)/200 for i in X0_2.index]
    # # select interaction matrices part of the dataframes 
    interaction_strength_2 = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
    interaction_strength_2 = interaction_strength_2.dropna()
    return growthRates_2, X0_2, interaction_strength_2, accepted_simulations, accepted_parameters, final_runs


# # # # # ------ SOLVE ODE #2: Pre-reintroductions (2009-2010) -------

def runODE_2():
    growthRates_2, X0_2, interaction_strength_2, accepted_simulations, accepted_parameters, final_runs  = generateParameters2()
    all_runs_2 = []
    all_parameters_2 = []
    t_2 = np.linspace(0, 1, 10)
    # loop through each row of data
    for (rowNumber, rowContents_X0), (rowNumber, rowContents_growth), (interaction_strength_chunk) in zip(X0_2.iterrows(),growthRates_2.iterrows(),np.array_split(interaction_strength_2,len(accepted_simulations))):
        # concantenate the parameters
        X0_growth_2 = pd.concat([rowContents_X0.rename('X0'), rowContents_growth.rename('growth')], axis = 1)
        parameters_used_2 = pd.concat([X0_growth_2, interaction_strength_chunk])
        # run the model
        second_ABC = solve_ivp(ecoNetwork, (0, 1), rowContents_X0,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
        # append all the runs
        all_runs_2 = np.append(all_runs_2, second_ABC.y)
        # append all the parameters
        all_parameters_2.append(parameters_used_2)
    # check the final runs
    final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*len(accepted_simulations), 10).transpose(),len(accepted_simulations))))
    final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)
    # add ID to the dataframe
    IDs = np.arange(1,1 + len(accepted_simulations))
    final_runs_2['ID'] = np.repeat(IDs,10)
    return final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs


# --------- FILTER OUT UNREALISTIC RUNS: Post-reintroductions -----------
def filterRuns_2():
    final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = runODE_2()
    # select only the last run (filter to the year 2010)
    accepted_year_2010 = final_runs_2.iloc[9::10, :]
    accepted_simulations_2010 = accepted_year_2010[(accepted_year_2010['roeDeer'] <= (X0_2['roeDeer'].max())) & (accepted_year_2010['roeDeer'] >= (X0_2['roeDeer'].min()))
    # # (accepted_year_2010['fox'] <= (X0_2['fox'].max()*100)) & (accepted_year_2010['fox'] >= (X0_2['fox'].min()*.0001))
    # # (accepted_year_2010['rabbits'] <= (X0_2['rabbits'].max())) & (accepted_year_2010['rabbits'] >= (X0_2['rabbits'].min())) &
    # # (accepted_year_2010['arableGrass'] <= (X0_2['arableGrass'].max())) & (accepted_year_2010['arableGrass'] >= (X0_2['arableGrass'].min())) &
    # # (accepted_year_2010['woodland'] <= (X0_2['woodland'].max())) & (accepted_year_2010['woodland'] >= (X0_2['woodland'].min()))
    ]
    print(accepted_simulations_2010.shape)
    # match ID number in accepted_simulations to its parameters in all_parameters
    accepted_parameters_2010 = accepted_simulations_2010[accepted_simulations_2010['ID'].isin(accepted_simulations_2010['ID'])]
    # add accepted ID to original dataframe
    final_runs_2['accepted?'] = np.where(final_runs_2['ID'].isin(accepted_simulations_2010['ID']), 'Yes', 'No')
    return accepted_simulations_2010, accepted_parameters_2010, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs




# # # # # ----------------------------- PLOTTING POPULATIONS  (2000-2010) ----------------------------- 

def plotting():
    accepted_simulations_2010, accepted_parameters_2010, final_runs_2, all_parameters_2, X0_2, accepted_simulations, accepted_parameters, final_runs = filterRuns_2()
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
    species_firstRun = np.tile(species, 100*NUMBER_OF_SIMULATIONS)
    species_secondRun = np.tile(species, 10*len(accepted_simulations))
    species_list = np.concatenate((species_firstRun, species_secondRun), axis=None)
    # add a grouping variable to graph each run separately
    grouping1 = np.arange(1,NUMBER_OF_SIMULATIONS+1)
    grouping_variable1 = np.repeat(grouping1,100*len(species))
    grouping2 = np.arange(NUMBER_OF_SIMULATIONS+2, NUMBER_OF_SIMULATIONS + 2 + len(accepted_simulations))
    grouping_variable2 = np.repeat(grouping2,10*len(species))
    # concantenate them 
    grouping_variable = np.concatenate((grouping_variable1, grouping_variable2), axis=None)
    # years - we've looked at 11 so far (2000-2010)
    year = np.arange(1,11)
    indices1 = np.repeat(year,len(species)*10)
    indices1 = np.tile(indices1, NUMBER_OF_SIMULATIONS)
    indices2 = np.repeat(11,len(species)*10)
    indices2 = np.tile(indices2, len(accepted_simulations))
    indices = np.concatenate((indices1, indices2), axis=None)
    # put it in a dataframe
    example_df = pd.DataFrame(
        {'nodeValues': y_values, 'runNumber': grouping_variable, 'species': species_list, 'time': indices, 'runAccepted': accepted_shape})
    # color palette
    palette = dict(zip(example_df.runAccepted.unique(),
                    sns.color_palette("RdBu_r", 2)))
    # plot
    sns.relplot(x="time", y="nodeValues",
                hue="runAccepted", col="species",
                palette=palette, height=3, aspect=.75, facet_kws=dict(sharex=False, sharey=False),
                kind="line", legend="full", col_wrap=4, data=example_df)
    plt.show()


plotting()

# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/len(species), 'Total nodes: ' , len(species))
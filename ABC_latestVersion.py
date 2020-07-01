# ---- Approximate Bayesian Computation Model of the Knepp Estate ------
from scipy import integrate
from scipy.integrate import solve_ivp
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string
import networkx as nx
import timeit
import seaborn as sns

# time the program
start = timeit.default_timer()

##### ------- DEFINE THE FUNCTION -------------

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
        # append values to output_array
        output_array.append(amount)

    # return array
    return output_array


# Define the number of simulations to try. Bode et al. ran a million
NUMBER_OF_SIMULATIONS = 10


# ------ GENERATE INTERACTION MATRIX ------

# PARAMETER MATRIX
interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
# store species in a list
species = list(interactionMatrix_csv.columns.values)
# remember the shape of the csv array
interaction_length = len(interactionMatrix_csv)

# pull the original sign / number from csv
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
#  re-shape so each run in NUMBER_OF_SIMULATIONS is its own column
interaction_strength = interaction_strength.unstack(level='z').swaplevel(0, 1).sort_index()
interaction_strength.index = interaction_strength.index.swaplevel(0, 1)
interaction_strength = interaction_strength.sort_values(by=['runs', 'species'], ascending=[True, True])
# add headers/columns
interaction_strength.columns = species
interaction_strength.index = species * NUMBER_OF_SIMULATIONS

# find the shape of original_sign
shape = len(original_sign)
# make lots of these arrays, keeping sign consistent with original_sign
iterStrength_list = np.array(
    [[np.random.uniform(0, 1, (shape,)) * original_sign] for i in range(NUMBER_OF_SIMULATIONS)])
iterStrength_list.shape = (NUMBER_OF_SIMULATIONS, interaction_length, interaction_length)

# convert to multi-index so that species' headers/cols can be added
names = ['runs', 'y', 'z']
index = pd.MultiIndex.from_product([range(s) for s in iterStrength_list.shape], names=names)
interaction_strength = pd.DataFrame({'iterStrength_list': iterStrength_list.flatten()}, index=index)[
    'iterStrength_list']
#  re-shape so each run in NUMBER_OF_SIMULATIONS is its own column
interaction_strength = interaction_strength.unstack(level='z').swaplevel(0, 1).sort_index()
interaction_strength.index = interaction_strength.index.swaplevel(0, 1)
interaction_strength = interaction_strength.sort_values(by=['runs', 'y'], ascending=[True, True])
# add headers/columns
interaction_strength.columns = species
interaction_strength.index = species * NUMBER_OF_SIMULATIONS


# --- MAKE NETWORK X VISUAL OF INTERACTION MATRIX --
# networkVisual = nx.DiGraph(interactionMatrix_csv)
# # increase distance between nodes
# pos = nx.spring_layout(networkVisual, scale=4)
# # draw graph
# nx.draw(networkVisual, pos, with_labels=True, font_size = 8, node_size = 4000, node_color = 'lightgray')
# plt.show()
# # figure out if some nodes or links should be different colors? e.g. neg vs positive effects...



# --- GENERATE GROWTH RATES
growthRates_csv = pd.read_csv('./growthRates.csv')
# generate new dataframe with random uniform distribution
growthRates = pd.DataFrame(np.random.uniform(low=growthRates_csv.iloc[0],high=growthRates_csv.iloc[1], size=(NUMBER_OF_SIMULATIONS, interaction_length)),columns=growthRates_csv.columns)


# --- GENERATE INITIAL NUMBERS
initial_numbers_csv = pd.read_csv('./initial_numbers.csv')
# generate new dataframe with random uniform distribution
X0_raw = pd.DataFrame(np.random.uniform(low=initial_numbers_csv.iloc[0],high=initial_numbers_csv.iloc[1], size=(NUMBER_OF_SIMULATIONS, interaction_length)),columns=initial_numbers_csv.columns)
# normalize each row of data to 0-1; divide by 200 to keep it consisntent between runs
X0 = X0_raw.div(200, axis=0)


###### --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -----------

# Define time points: first 10 years (2000-2009)
t = np.linspace(0, 10, 100)

all_runs = []
all_parameters = []

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
# check the final runsÂ
final_runs = (np.vstack(np.hsplit(all_runs.reshape(len(species)*NUMBER_OF_SIMULATIONS, 100).transpose(),NUMBER_OF_SIMULATIONS)))
final_runs = pd.DataFrame(data=final_runs, columns=species)

# append all the parameters to a dataframe
all_parameters = pd.concat(all_parameters)
# add ID to all_parameters
all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])



###### --------- FILTER OUT UNREALISTIC RUNS -----------

# select only the last run (filter to the year 2009)
accepted_year = final_runs.iloc[99::100, :]
# add ID to the dataframe
accepted_year.insert(0, 'ID', range(1, 1 + len(accepted_year)))

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(accepted_year)

# filter the conditions
accepted_simulations = accepted_year[(accepted_year['fallowDeer'] <= (X0['fallowDeer'].max())) & (accepted_year['fallowDeer'] >= (X0['fallowDeer'].min())) 
# (accepted_year['smallMammal'] <= (X0['smallMammal'].max())) & (accepted_year['smallMammal'] >= (X0['smallMammal'].min())) &
# (accepted_year['secondaryConsumer'] <= (X0['secondaryConsumer'].max())) & (accepted_year['secondaryConsumer'] >= (X0['secondaryConsumer'].min())) &
# (accepted_year['fox'] <= (X0['fox'].max())) & (accepted_year['fox'] >= (X0['fox'].min())) &
# (accepted_year['songbirdsCorvids'] <= (X0['songbirdsCorvids'].max())) & (accepted_year['songbirdsCorvids'] >= (X0['songbirdsCorvids'].min()*1.62)) &
# (accepted_year['decomposers'] <= (X0['decomposers'].max())) & (accepted_year['decomposers'] >= (X0['decomposers'].min())) &
# (accepted_year['arableGrass'] <= (X0['arableGrass'].max())) & (accepted_year['arableGrass'] >= (X0['arableGrass'].min()*1.52)) &
# (accepted_year['woodland'] <= (X0['woodland'].max()*1.30)) & (accepted_year['woodland'] >= (X0['woodland'].min()*1.4)) &
# (accepted_year['thornyScrub'] <= (X0['thornyScrub'].max()*2)) & (accepted_year['thornyScrub'] >= (X0['thornyScrub'].min())) &
# (accepted_year['Water'] <= (X0['Water'].max())) & (accepted_year['Water'] >= (X0['Water'].min())) &
# (accepted_year['organicMatter'] <= (X0['organicMatter'].max()*2)) & (accepted_year['organicMatter'] >= (X0['organicMatter'].min())) &
# (accepted_year['soilNutrients'] <= (X0['soilNutrients'].max()*2.16)) & (accepted_year['soilNutrients'] >= (X0['soilNutrients'].min()))
]

print(accepted_simulations)

# match ID number in accepted_simulations to its parameters in all_parameters
accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]




###### -------------------- ODE #2: 2009-2010  -----------

accepted_simulations = accepted_simulations.drop('ID', axis=1)
# make the final runs of ODE #1 the initial conditions
accepted_parameters.loc[accepted_parameters['X0'].notnull(), ['X0']] = accepted_simulations.values.flatten()
accepted_parameters.index.name='species'
# take min/max for each growth rate accepted_dataframes; normal distribution between those values
minValue_grow = accepted_parameters.groupby(['species'], sort=False)['growth'].min()
maxValue_grow = accepted_parameters.groupby(['species'], sort=False)['growth'].max()
growthRates_2 = pd.DataFrame(np.random.uniform(low=minValue_grow,high=maxValue_grow, size=(NUMBER_OF_SIMULATIONS, interaction_length)),columns=species)
# take min/max for each X0 in accepted_dataframes; normal distribution between those values
minValue_X0 = accepted_parameters.groupby(['species'], sort=False)['X0'].min()
maxValue_X0 = accepted_parameters.groupby(['species'], sort=False)['X0'].max()
X0_2 = pd.DataFrame(np.random.uniform(low=minValue_X0,high=maxValue_X0, size=(NUMBER_OF_SIMULATIONS, interaction_length)),columns=species)
# add forcings; these species are reintroduced. Divide by 200 to keep the scaling/normalization consistent
X0_2.loc[:, 'fallowDeer'] = [np.random.uniform(low=1.25, high=2.18)/200 for i in X0_2.index]
X0_2.loc[:,'longhornCattle'] = [np.random.uniform(low=35, high=70)/200 for i in X0_2.index]
# X0_2.loc[:,'exmoorPony'] = [np.random.uniform(low=2, high=5.5)/200 for i in X0_2.index]
# X0_2.loc[:,'tamworthPig'] = [np.random.uniform(low=6.36, high=11.77)/200 for i in X0_2.index]

# select interaction matrices part of the dataframes 
interaction_values = accepted_parameters.drop(['X0', 'growth', 'ID'], axis=1)
interaction_values = interaction_values.dropna()
# select the min/max of each interaction and make it a normal distribution between these
minimum_results = []
maximum_results = []
for outer_index, outer_species in enumerate(species):
    for inner_index, inner_species in enumerate(species):
        mini = (interaction_values[inner_species][outer_species]).min()
        maxi = (interaction_values[inner_species][outer_species]).max()
        minimum_results.append(mini)
        maximum_results.append(maxi)
# make lots of these arrays, keeping values consistent with the accepted ranges
iterStrength_list = np.array([[np.random.uniform(minimum_results, maximum_results, (shape,))] for i in range(NUMBER_OF_SIMULATIONS)])
iterStrength_list.shape = (NUMBER_OF_SIMULATIONS, interaction_length, interaction_length)
# convert to multi-index so that species' headers/cols can be added
names = ['runs', 'y', 'z']
index = pd.MultiIndex.from_product([range(s) for s in iterStrength_list.shape], names=names)
interaction_strength_2 = pd.DataFrame({'iterStrength_list': iterStrength_list.flatten()}, index=index)[
    'iterStrength_list']
#  re-shape so each run in NUMBER_OF_SIMULATIONS is its own column
interaction_strength_2 = interaction_strength_2.unstack(level='z').swaplevel(0, 1).sort_index()
interaction_strength_2.index = interaction_strength_2.index.swaplevel(0, 1)
interaction_strength_2 = interaction_strength_2.sort_values(by=['runs', 'y'], ascending=[True, True])
# add headers/columns
interaction_strength_2.columns = species
interaction_strength_2.index = species * NUMBER_OF_SIMULATIONS


# re-run the ODE: 2009-2010
all_runs_2 = []
all_parameters_2 = []

t_2 = np.linspace(0, 1, 100)

# loop through each row of data
for (rowNumber, rowContents_X0), (rowNumber, rowContents_growth), (interaction_strength_chunk) in zip(X0_2.iterrows(),growthRates_2.iterrows(),np.array_split(interaction_strength_2,NUMBER_OF_SIMULATIONS)):
    # concantenate the parameters
    X0_growth_2 = pd.concat([rowContents_X0.rename('X0'), rowContents_growth.rename('growth')], axis = 1)
    parameters_used_2 = pd.concat([X0_growth, interaction_strength_chunk])
    # run the model
    second_ABC = solve_ivp(ecoNetwork, (0, 1), rowContents_X0,  t_eval = t_2, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # append all the runs
    all_runs_2 = np.append(all_runs_2, second_ABC.y)
    # append all the parameters
    all_parameters_2.append(parameters_used_2)
# check the final runsÂ
final_runs_2 = (np.vstack(np.hsplit(all_runs_2.reshape(len(species)*NUMBER_OF_SIMULATIONS, 100).transpose(),NUMBER_OF_SIMULATIONS)))
final_runs_2 = pd.DataFrame(data=final_runs_2, columns=species)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(final_runs_2)

# ------ Plotting populations ------- 
# organize the data
y_values = final_runs_2.values.flatten()
# we want species column to be spec1,spec2,spec3,spec4, etc., repeat
species_list = np.tile(species, 100*NUMBER_OF_SIMULATIONS)
# add a grouping variable to graph each run separately
grouping = np.arange(1,NUMBER_OF_SIMULATIONS+1)
grouping_variable = np.repeat(grouping,100*len(species))
# we want the timing to increment by 1 every len(species)
timing = np.arange(0,100)
timing_shape = np.repeat(timing,len(species))
indices = np.tile(timing_shape, NUMBER_OF_SIMULATIONS)
# put it in a dataframe
example_df = pd.DataFrame(
    {'nodeValues': y_values, 'runNumber': grouping_variable, 'species': species_list, 'time': indices})

# color palette
palette = dict(zip(example_df.runNumber.unique(),
                   sns.color_palette("rocket_r", NUMBER_OF_SIMULATIONS)))
# plot
sns.relplot(x="time", y="nodeValues",
            hue="runNumber", col="species",
            size_order=["T1", "T2"], palette=palette,
            height=5, aspect=.75, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=example_df)
plt.show()


# calculate the time it takes to run per node
stop = timeit.default_timer()
time = []

print('Total time: ', (stop - start))
print('Time per node: ', (stop - start)/interaction_length, 'Total nodes: ' , interaction_length)


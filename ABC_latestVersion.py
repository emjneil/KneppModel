# ---- Approximate Bayesian Computation Model of the Knepp Estate ------
from scipy import integrate
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string
import networkx as nx

##### ------- DEFINE THE FUNCTION -------------

def ecoNetwork(X, t, interaction_strength_chunk, rowContents_growth):
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

    # Define the maximum number of herbivores (culling)
    # herb_max = 250

    # # for the herbivores in amount, don't allow them to go over the max herbivore value defined above
    # # make sure herbivores are in position 1
    # output_array[0] = np.where((X[0] >= herb_max), herb_max - X[0], output_array[0])

    # return array
    return output_array


# Define the number of simulations to try. Bode et al. ran a million
NUMBER_OF_SIMULATIONS = 100



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
# make list = to number of nodes and their scale
scaled_list = pd.DataFrame([])
# normalize each row of data to 0-1
X0 = X0_raw.div(X0_raw.sum(axis=1), axis=0)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
#     print(X0)


###### --------- SOLVE ODE #1: Pre-reintroductions (2000-2009) -----------

# Define time points: first 10 years (2000-2009)
t = np.linspace(0, 10, 100)

all_runs = []
all_parameters = []

# loop through each row of data
for (rowNumber, rowContents_X0), (rowNumber, rowContents_growth), (interaction_strength_chunk) in zip(X0.iterrows(),growthRates.iterrows(),np.array_split(interaction_strength,NUMBER_OF_SIMULATIONS)):
    parameters_used = pd.concat([rowContents_X0, rowContents_growth, interaction_strength_chunk])
    first_ABC = integrate.odeint(ecoNetwork, rowContents_X0, t, args=(interaction_strength_chunk, rowContents_growth))
    # append all the runs
    all_runs = np.append(all_runs, first_ABC)
    # append all the parameters
    all_parameters.append(parameters_used)
# check the final runs
final_runs = pd.DataFrame(all_runs.reshape(NUMBER_OF_SIMULATIONS * 100, len(species)), columns=species)
# append all the parameters to a dataframe
all_parameters = pd.concat(all_parameters)
# add ID to all_parameters
all_parameters['ID'] = ([(x+1) for x in range(NUMBER_OF_SIMULATIONS) for _ in range(len(parameters_used))])



##### ------ Plotting Populations ---------
fallowDeer, longhornCattle, redDeer, roeDeer, exmoorPony, tamworthPig, beaver, hedgehog, mustelids, fox, smallRodent, rabbits, bats, amphibianLizard, snakes, songbirds, raptors = first_ABC.T
plt.plot(t, fallowDeer, label = 'Fallow deer')
plt.plot(t, longhornCattle, label = 'Longhorn cattle')
plt.plot(t, redDeer, label = 'Red deer')
plt.plot(t, roeDeer, label = 'Roe deer')
plt.plot(t, exmoorPony, label = 'Exmoor pony')
plt.plot(t, tamworthPig, label = 'Tamworth pig')
plt.plot(t, beaver, label = 'Beavers')
plt.plot(t, hedgehog, label = 'Hedgehog')
plt.plot(t, mustelids, label = 'Mustelids')
plt.plot(t, fox, label = 'Fox')
plt.plot(t, smallRodent, label = ' Small Rodent')
plt.plot(t, rabbits, label = 'Rabbits')
plt.plot(t, bats, label = 'Bats')
plt.plot(t, amphibianLizard, label = 'Amphibians and Lizards')
plt.plot(t, snakes, label = 'Snakes')
plt.plot(t, songbirds, label = 'Songbirds')
plt.plot(t, raptors, label = 'Raptors')

plt.legend(loc='upper right')
plt.show()


###### --------- FILTER OUT UNREALISTIC RUNS -----------

# select only the last run (filter to the year 2009)
accepted_year = final_runs.iloc[99::100, :]
# add ID to the dataframe
accepted_year.insert(0, 'ID', range(1, 1 + len(accepted_year)))
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(accepted_year)


# filter the conditions 
# fallowDeer, longhornCattle, redDeer, exmoorPony, tamworthPig, beaver, hedgehog are all = 0
accepted_simulations = accepted_year[(accepted_year['roeDeer'] <= (X0['roeDeer'].max())) & (accepted_year['roeDeer'] >= (X0['roeDeer'].min())) &
(accepted_year['mustelids'] <= (X0['mustelids'].max())) & (accepted_year['mustelids'] >= (X0['mustelids'].min())) &
(accepted_year['fox'] <= (X0['fox'].max())) & (accepted_year['fox'] >= (X0['fox'].min())) &
(accepted_year['smallRodent'] <= (X0['smallRodent'].max())) & (accepted_year['smallRodent'] >= (X0['smallRodent'].min())) &
(accepted_year['rabbits'] <= (X0['rabbits'].max())) & (accepted_year['rabbits'] >= (X0['rabbits'].min())) &
(accepted_year['bats'] <= (X0['bats'].max())) & (accepted_year['bats'] >= (X0['bats'].min())) &
(accepted_year['amphibianLizard'] <= (X0['amphibianLizard'].max())) & (accepted_year['amphibianLizard'] >= (X0['amphibianLizard'].min())) &
(accepted_year['snakes'] <= (X0['snakes'].max())) & (accepted_year['snakes'] >= (X0['snakes'].min())) &
(accepted_year['songbirds'] <= (X0['songbirds'].max())) & (accepted_year['songbirds'] >= (X0['songbirds'].min()*1.78)) &
(accepted_year['raptors'] <= (X0['raptors'].max())) & (accepted_year['raptors'] >= (X0['raptors'].min()))
]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
print(accepted_simulations)

# match ID number in accepted_simulations to its parameters in all_parameters
accepted_parameters = all_parameters[all_parameters['ID'].isin(accepted_simulations['ID'])]


###### --------- ODE #2  -----------

# take the accepted_parameters and make them the initial conditions of the next ODE, + forcings
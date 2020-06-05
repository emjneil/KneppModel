# ---- Approximate Bayesian Computation Model of the Knepp Estate ------
from scipy import integrate
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string
import networkx as nx


##### ------- Define the function -------------

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
    herb_max = 250

    # for the herbivores in amount, don't allow them to go over the max herbivore value defined above
    # make sure herbivores are in position 1
    output_array[0] = np.where((X[0] >= herb_max), herb_max - X[0], output_array[0])

    # return array
    return output_array


# Define the number of simulations to try
NUMBER_OF_SIMULATIONS = 4

# ------ Generate the interaction pairs ------

# PARAMETER MATRIX
interactionMatrix_csv = pd.read_csv('./parameterMatrix.csv', index_col=[0])
# store species in a list
species = list(interactionMatrix_csv.columns.values)
# remember the shape of the csv array
interaction_length = len(interactionMatrix_csv)

# # --- make networkX visual of parameter matrix --
# networkVisual = nx.Graph()
# # add nodes
# nx.set_node_attributes(networkVisual, species)
# # add edges
# networkVisual.add_edge()
# # show plot
# nx.draw(networkVisual, with_labels=True)
# plot.draw()
# plt.show()


# pull the original sign / number from csv
original_sign = []

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
names = ['x', 'y', 'z']
index = pd.MultiIndex.from_product([range(s) for s in iterStrength_list.shape], names=names)
interaction_strength = pd.DataFrame({'iterStrength_list': iterStrength_list.flatten()}, index=index)[
    'iterStrength_list']
#  re-shape so each run in NUMBER_OF_SIMULATIONS is its own column
interaction_strength = interaction_strength.unstack(level='z').swaplevel(0, 1).sort_index()
interaction_strength.index = interaction_strength.index.swaplevel(0, 1)
interaction_strength = interaction_strength.sort_values(by=['x', 'y'], ascending=[True, True])
# add headers/columns
interaction_strength.index = species * NUMBER_OF_SIMULATIONS
interaction_strength.columns = species


# # ------- Define the parameters to try in the first ABC ------


# # --- GROWTH RATES
growthRates_csv = pd.read_csv('./growthRates.csv')
# generate new dataframe with random uniform distribution
growthRates = pd.DataFrame(np.random.uniform(low=growthRates_csv.iloc[0],high=growthRates_csv.iloc[1], size=(NUMBER_OF_SIMULATIONS, interaction_length)),columns=growthRates_csv.columns)

# # --- INITIAL NUMBERS
initial_numbers_csv = pd.read_csv('./initial_numbers.csv')
# generate new dataframe with random uniform distribution
X0 = pd.DataFrame(np.random.uniform(low=initial_numbers_csv.iloc[0],high=initial_numbers_csv.iloc[1], size=(NUMBER_OF_SIMULATIONS, interaction_length)),columns=initial_numbers_csv.columns)


###### --------- Solve the ODE-----------

# Define time points: first 5 years, 2000-2004
t = np.linspace(0, 5, 50)

all_runs = []

# loop through each row of data
for (rowNumber, rowContents_X0), (rowNumber, rowContents_growth), (interaction_strength_chunk) in zip(X0.iterrows(),
                                                                                                      growthRates.iterrows(),
                                                                                                      np.array_split(
                                                                                                              interaction_strength,
                                                                                                              NUMBER_OF_SIMULATIONS)):
    first_ABC = integrate.odeint(ecoNetwork, rowContents_X0, t, args=(interaction_strength_chunk, rowContents_growth))
    # append to list
    all_runs = np.append(all_runs, first_ABC)

final_runs = pd.DataFrame(all_runs.reshape(NUMBER_OF_SIMULATIONS * 50, len(species)), columns=species)


###### --------- Filter out unrealistic runs -----------

# assign conditions - e.g. make sure none of the nodes are below zero at time stamp ~5 yrs


# # ###### --------- Take posteriors of runs and plug into next  -----------


# # ##### ------ Plotting Populations ---------

herbivores, youngScrub, matureScrub = first_ABC.T
plt.plot(t, herbivores, label = 'Herbivores')
plt.plot(t, youngScrub, label = 'Young scrub')
plt.plot(t, matureScrub, label = 'Mature scrub')
plt.legend(loc='upper right')
plt.show()


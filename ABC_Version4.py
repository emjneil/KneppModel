# ---- Approximate Bayesian Computation Model of the Knepp Estate ------
### This has an issue in line 133 that I'm working on

from scipy import integrate
import pylab as p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools as IT
import string


##### ------- Define the function -------------

def ecoNetwork(X, t=0):

#define new array to return
    new_array = []

    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species):
        new_sum = growthRates[outer_species] * X[outer_index]
        for inner_index, inner_species in enumerate(species):
            new_sum += np.sum(interaction_strength[outer_species][inner_species]*X[outer_index]*X[inner_index])

        # append values to new_array
        new_array.append(new_sum)

    # Define the maximum number of herbivores (culling)
    herb_max = 250

    # for the herbivores in new_sum, don't allow them to go over the max herbivore value defined above
    # make sure herbivores are in position 1
    new_array[0] = np.where((X[0] >= herb_max),herb_max-X[0],new_array[0])

    # return array
    return new_array


# Define the number of simulations to try
NUMBER_OF_SIMULATIONS = 3 # just to test that this works



# ------ Generate the interaction pairs ------ 

# PARAMETER MATRIX
interactionMatrix_csv = pd.read_csv ('/Users/user/Desktop/Knepp ODE/ODE code/parameterMatrix.csv', index_col=[0])
# store species in a list
species = list(interactionMatrix_csv.columns.values)
# remember the shape of the csv array
interaction_length = len(interactionMatrix_csv)


# pull the original sign / number from csv
original_sign= []

for outer_index, outer_species in enumerate(species):
    for inner_index, inner_species in enumerate(species):
        # pull the sign
        new_sign = interactionMatrix_csv[inner_species][outer_species]
        original_sign.append(new_sign)
# find the shape of original_sign
shape = len(original_sign)


# make lots of these arrays, keeping sign consistent with original_sign
iterStrength_list = np.array([[np.random.uniform(0,1,(shape,))*original_sign] for i in range(NUMBER_OF_SIMULATIONS)])
iterStrength_list.shape = (NUMBER_OF_SIMULATIONS, interaction_length, interaction_length)


# convert to multi-index so that species' headers/cols can be added
names = ['x', 'y', 'z']
index = pd.MultiIndex.from_product([range(s)for s in iterStrength_list.shape], names=names)
interaction_strength = pd.DataFrame({'iterStrength_list': iterStrength_list.flatten()}, index=index)['iterStrength_list']
#  re-shape so each run in NUMBER_OF_SIMULATIONS is its own column
interaction_strength = interaction_strength.unstack(level='z').swaplevel(0,1).sort_index()
interaction_strength.index = interaction_strength.index.swaplevel(0,1)
interaction_strength = interaction_strength.sort_values(by = ['x', 'y'], ascending = [True, True])
# # add headers/columns
interaction_strength.index = species*NUMBER_OF_SIMULATIONS
interaction_strength.columns = species





# ------- Define the parameters to try in the first ABC ------


# --- GROWTH RATES
growthRates_csv = pd.read_csv ('/Users/user/Desktop/Knepp ODE/ODE code/growthRates.csv')
# select the minimum and maximum values in the csv
for index, species_name in enumerate(species):
    min_growthRate = np.array(growthRates_csv.iloc[0])
    max_growthRate = np.array(growthRates_csv.iloc[1])

# put these into a new dataframe and generate x values in that range
growthRates = pd.DataFrame([np.random.uniform(minG, maxG, NUMBER_OF_SIMULATIONS) for minG,maxG in zip(min_growthRate, max_growthRate)], index=species)
# transpose the data frame
growthRates = growthRates.T



# --- INITIAL NUMBERS
initial_numbers_csv = pd.read_csv ('/Users/user/Desktop/Knepp ODE/ODE code/initial_numbers.csv')
# select the minimum and maximum values in the csv
for index, species_name in enumerate(species):
    min_X0 = np.array(initial_numbers_csv.iloc[0])
    max_X0 = np.array(initial_numbers_csv.iloc[1])
# put these into a new dataframe and generate x values in that range
X0 = pd.DataFrame([np.random.uniform(minX0, maxX0, NUMBER_OF_SIMULATIONS) for minX0,maxX0 in zip(min_X0, max_X0)], index=species)
X0 = X0.T





###### --------- Run the model -----------

# Define time points: first 5 years, 2000-2004
t = np.linspace(0, 5)

# run the model for each parameter
all_runs = []
### need to fix this
# want to loop through each row of the growth rates and starting values sheets...
for i, row in growthRates.iterrows():
    for j, r in X0.iterrows():
# then loop through the interaction strength....
        # for X0, row in initial_numbers.iterrows():

# then integrate those into the ODE 
        first_ABC = integrate.odeint(ecoNetwork, X0, t)
            # all_runs.append(first_ABC)

print(first_ABC)







###### --------- Filter out unrealistic runs -----------

# make sure none of the nodes are below zero


# plug in conditions 




##### ------ Plotting Populations ---------
herbivores, youngScrub, matureScrub, sapling, matureTree, grassHerbs = first_ABC.T
plt.plot(t, herbivores, label = 'Herbivores')
plt.plot(t, matureScrub, label = 'Mature scrub')
plt.plot(t, youngScrub, label = 'Young scrub')
plt.plot(t, matureTree, label = 'Mature trees')
plt.plot(t, sapling, label = 'Saplings')
plt.plot(t, grassHerbs, label = 'Grass and herbaceous plants')
plt.legend(loc='upper right')
plt.show()

# phase plot
# plt.plot(herbivores, matureScrub, youngScrub, matureTree, sapling, grassHerbs '-->', markevery=5, label = 'phase plot')
# plt.legend(loc='upper right')
# #labels
# plt.xlabel("species")
# plt.ylabel("number of species")
# #title
# plt.title("Lotka-Volterra Practice Model")
# plt.show()








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
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
import hyperopt.pyll.stochastic
from collections import defaultdict
import pickle


# # # # # --------- Hyperopt Bayesian optimization ----------- # # # # # # # 
  
# store species in a list
species = ['arableGrass','fox','rabbits','raptors','reptilesAmphibians','roeDeer','songbirdsWaterfowl','thornyScrub','wetland','woodland']

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
            if amount.item() >= 1e3 or amount.item() is None or np.isnan(amount.item()) or np.isinf(amount.item()):
                amount = None
                break
        # append values to output_array
        output_array.append(amount)
    # return array
    return output_array


def objectiveFunction(x):
    # extract starting values
    X0 = pd.DataFrame(x['X0'].values(), index=x['X0'].keys())
    X0 = X0.values.tolist()
    X0 = sum(X0, [])
    # add growth rates of 0 (where no choices are being made between values)
    x['growthRate']['wetland'] = 0
    # extract growth rates
    growthRate = pd.DataFrame(x['growthRate'].values(), index=x['growthRate'].keys())
    rowContents_growth = growthRate.T
    # add interaction values of 0 (where no choices are being made between values)
    x['interact']['roeDeer']['roeDeer'] = 0
    x['interact']['roeDeer']['rabbits'] = 0
    x['interact']['roeDeer']['fox'] = 0
    x['interact']['roeDeer']['songbirdsWaterfowl'] = 0
    x['interact']['roeDeer']['raptors'] = 0
    x['interact']['roeDeer']['reptilesAmphibians'] = 0
    x['interact']['roeDeer']['wetland'] = 0
    x['interact']['rabbits']['roeDeer'] = 0
    x['interact']['rabbits']['rabbits'] = 0
    x['interact']['rabbits']['songbirdsWaterfowl'] = 0
    x['interact']['rabbits']['reptilesAmphibians'] = 0
    x['interact']['rabbits']['wetland'] = 0
    x['interact']['fox']['roeDeer'] = 0
    x['interact']['fox']['fox'] = 0
    x['interact']['fox']['raptors'] = 0
    x['interact']['fox']['arableGrass'] = 0
    x['interact']['fox']['woodland'] = 0
    x['interact']['fox']['thornyScrub'] = 0
    x['interact']['fox']['wetland'] = 0
    x['interact']['songbirdsWaterfowl']['roeDeer'] = 0
    x['interact']['songbirdsWaterfowl']['rabbits'] = 0
    x['interact']['songbirdsWaterfowl']['songbirdsWaterfowl'] = 0
    x['interact']['songbirdsWaterfowl']['reptilesAmphibians'] = 0
    x['interact']['songbirdsWaterfowl']['arableGrass'] = 0
    x['interact']['songbirdsWaterfowl']['woodland'] = 0
    x['interact']['songbirdsWaterfowl']['thornyScrub'] = 0
    x['interact']['songbirdsWaterfowl']['wetland'] = 0
    x['interact']['raptors']['roeDeer'] = 0
    x['interact']['raptors']['fox'] = 0
    x['interact']['raptors']['raptors'] = 0
    x['interact']['raptors']['arableGrass'] = 0
    x['interact']['raptors']['woodland'] = 0
    x['interact']['raptors']['thornyScrub'] = 0
    x['interact']['raptors']['wetland'] = 0
    x['interact']['reptilesAmphibians']['roeDeer'] = 0
    x['interact']['reptilesAmphibians']['rabbits'] = 0
    x['interact']['reptilesAmphibians']['songbirdsWaterfowl'] = 0
    x['interact']['reptilesAmphibians']['reptilesAmphibians'] = 0
    x['interact']['reptilesAmphibians']['arableGrass'] = 0
    x['interact']['reptilesAmphibians']['woodland'] = 0
    x['interact']['reptilesAmphibians']['thornyScrub'] = 0
    x['interact']['reptilesAmphibians']['wetland'] = 0
    x['interact']['arableGrass']['fox'] = 0
    x['interact']['arableGrass']['songbirdsWaterfowl'] = 0
    x['interact']['arableGrass']['raptors'] = 0
    x['interact']['arableGrass']['reptilesAmphibians'] = 0
    x['interact']['arableGrass']['arableGrass'] = 0
    x['interact']['arableGrass']['woodland'] = 0
    x['interact']['arableGrass']['thornyScrub'] = 0
    x['interact']['arableGrass']['wetland'] = 0
    x['interact']['woodland']['fox'] = 0
    x['interact']['woodland']['reptilesAmphibians'] = 0
    x['interact']['woodland']['woodland'] = 0
    x['interact']['woodland']['wetland'] = 0
    x['interact']['thornyScrub']['fox'] = 0
    x['interact']['thornyScrub']['raptors'] = 0
    x['interact']['thornyScrub']['thornyScrub'] = 0
    x['interact']['thornyScrub']['wetland'] = 0
    x['interact']['wetland']['rabbits'] = 0
    x['interact']['wetland']['fox'] = 0
    x['interact']['wetland']['wetland'] = 0
    # extract interaction strength
    interaction_strength_chunk = pd.DataFrame(x['interact'].values(), index=x['interact'].keys())
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    array_sum = np.sum(y[49:50:,])
    # if there are NaNs, make it return a high number
    if np.isnan(array_sum):
        print ("output is too high or low")
        result = 9e10
    # otherwise calculate the middle of the filter
    else:
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
        print(y[49:50:,]*200)
        result = (((y[49:50:, 0]-(3.6/200))**2) + ((y[49:50:, 1]-(8.4/200))**2) + ((y[49:50:, 2]-(296.6/200))**2) + ((y[49:50:, 3]-(0.91/200))**2) + ((y[49:50:, 4]-(20.4/200))**2) + ((y[49:50:, 5]-(247.1/200))**2)  + ((y[49:50:, 6]-(13.62/200))**2) + ((y[49:50:, 7]-(0.47/200))**2) + ((y[49:50:, 9]-(0.76/200))**2))
        print(result)
    # # multiply it by how many filters were passed
    # all_filters = [4.9 <= (y[49:50:, 5]*200) <= 449, 0.27 <= (y[49:50:, 2]*200) <= 593, 0.3 <= (y[49:50:, 4]*200) <= 40, 1.1 <= (y[49:50:, 1]*200) <= 15.7, 0.15 <= (y[49:50:, 6]*200) <= 27,  0.11 <= (y[49:50:, 3]*200) <= 1.7,  2.05 <= (y[49:50:, 0]*200) <= 4, 0.45 <= (y[49:50:, 9]*200) <= 0.63, 0.049 <= (y[49:50:, 7]*200) <= 0.445]
    # result2 = sum(all_filters)
    # # return total number of filters minus filters passed
    # result3 = 9-result2
    # result = result3 * result1
    return{'loss':result, 'status': STATUS_OK}

# VSC = middle * number of filters; 20k runs (Try 3)
# terminal = middle of filters; 20k runs (Try 2)

# order of outputs 
# ['arableGrass',    'fox',         'rabbits',      'raptors',          'reptiles',       'roeDeer',     'songbirdsWaterfowl', 'thornyScrub',            'wetland',       'woodland'])
#   2.05-4 (3.6)   1.1-15.7 (8.4)    0.27-593 (297)  0.11-1.7(0.91)     0.3-40(20.4)    4.9-449 (247)        0.15-27 (13.62)    0.049-0.445 (0.47)           0.018       0.45-0.63 (0.76)

param_hyperopt= {
    # starting value bounds
    'X0': {
    'roeDeer' : hp.uniform('roeDeer_x', 13.5/200, 101.1/200), 
    'rabbits': hp.uniform('rabbit_x', 0.27/200, 593/200),
    'fox': hp.uniform('fox_x', 1.1/200,15.7/200),
    'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl_x', 0.15/200,27/200),
    'raptors': hp.uniform('raptors_x', 0.11/200,1.7/200),
    'reptilesAmphibians': hp.uniform('reptilesAmphibians_x', 0.34/200,40.4/200),
    'arableGrass': hp.uniform('arableGrass_x', 2.05/200,4.0/200),
    'woodland': hp.uniform('woodland_x', 0.45/200,0.63/200),
    'thornyScrub': hp.uniform('thornyScrub_x', 0.049/200,0.445/200),
    'wetland': hp.uniform('wetland_x', 0.016/200,0.02/200)
    },
    # growth rate bounds
    'growthRate': {
    'roeDeer' : hp.uniform('growth_roe', 0,5), 
    'rabbits': hp.uniform('growth_rabbit',0,5),
    'fox': hp.uniform('growth_fox', 0,1),
    'songbirdsWaterfowl': hp.uniform('growth_songbird', 0,5),
    'raptors': hp.uniform('growth_raptor', 0,1),
    'reptilesAmphibians': hp.uniform('growth_ampib',0,5),
    'arableGrass': hp.uniform('growth_grass', 0,5),
    'woodland': hp.uniform('growth_wood', 0,5),
    'thornyScrub': hp.uniform('growth_scrub', 0,5),
    },
    # interaction matrix bounds
    'interact': 
    {
    'roeDeer': {'arableGrass': hp.uniform('arableGrass1', -1,0), 'woodland': hp.uniform('woodland1', -1,0), 'thornyScrub': hp.uniform('thornyScrub1', -1,0)},
    'rabbits': {'fox': hp.uniform('fox2',0,1), 'raptors': hp.uniform('raptors2', 0,1),  'arableGrass': hp.uniform('arableGrass2',-1,0), 'woodland': hp.uniform('woodland2', -1,0), 'thornyScrub': hp.uniform('thornyScrub2', -1,0)},
    'fox': {'rabbits': hp.uniform('rabbits3', -1,0), 'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl3', -1,0),  'reptilesAmphibians': hp.uniform('reptilesAmphibians3', -1,0)},
    'songbirdsWaterfowl': {  'fox': hp.uniform('fox4', 0,1),  'raptors': hp.uniform('raptors4', 0,1)},
    'raptors': { 'rabbits': hp.uniform('rabbits5', -1,0), 'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl5', -1,0),  'reptilesAmphibians': hp.uniform('reptilesAmphibians5', -1,0)},
    'reptilesAmphibians': { 'fox': hp.uniform('fox6', 0,1), 'raptors': hp.uniform('raptors6', 0,1)},
    'arableGrass': {'roeDeer':hp.uniform('roeDeer7', 0,1), 'rabbits': hp.uniform('rabbits7', 0,1)},
    'woodland': {'roeDeer':hp.uniform('roeDeer8', 0,1), 'rabbits': hp.uniform('rabbits8', 0,1), 'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl8', 0,1), 'raptors': hp.uniform('raptors8', 0,1),   'arableGrass': hp.uniform('arableGrass8', -1,0),  'thornyScrub': hp.uniform('thornyScrub8', -1,0)},
    'thornyScrub': {'roeDeer':hp.uniform('roeDeer9', 0,1),  'rabbits': hp.uniform('rabbits9', 0,1), 'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl9', 0,1),   'reptilesAmphibians': hp.uniform('reptilesAmphibians9', 0,1), 'arableGrass': hp.uniform('arableGrass9', -1,0), 'woodland': hp.uniform('woodland9', 0,1)},
    'wetland': {'roeDeer':hp.uniform('roeDeer10', 0,1),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl10', 0,1), 'raptors': hp.uniform('raptors10', 0,1),  'reptilesAmphibians': hp.uniform('reptilesAmphibians10', 0,1), 'arableGrass': hp.uniform('arableGrass10', -1,0), 'woodland': hp.uniform('woodland10', -1,0), 'thornyScrub': hp.uniform('thornyScrub10', -1,0)}

    }
}
                   
# print (hyperopt.pyll.stochastic.sample(param_hyperopt))

trials = Trials()
optimization = fmin(objectiveFunction, param_hyperopt, trials=trials, algo = tpe.suggest, max_evals = 10000)
print(optimization)
# print (space_eval(param_hyperopt, optimization))
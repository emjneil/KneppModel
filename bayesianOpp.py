
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
# species = ['arableGrass','beaver','fox','largeHerb','rabbits','raptors','reptilesAmphibians','roeDeer','songbirdsWaterfowl','tamworthPig','thornyScrub','wetland','woodland']
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
            if amount.item() >= 10000:
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
    # extract growth rates
    growthRate = pd.DataFrame(x['growthRate'].values(), index=x['X0'].keys())
    rowContents_growth = growthRate.T
    # extract interaction strength
    interaction_strength_chunk = pd.DataFrame(x['interact'].values(), index=x['interact'].keys())
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'LSODA')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters) & make sure the filters are also normalized (/200)
    # take outputs to start a genetic algorithm or another downhill method; sometimes chaining them works well. Take Bayesian output and plug it as initial guess to minimize. Start with filters passed; then switch to middle of range for second round
    print(y[49:50:,]*200)
    # middle of the filter
    result1 = ((y[49:50:, 5]-(226/200))**2) + ((y[49:50:, 2]-(227/200))**2) + ((y[49:50:, 4]-(20.4/200))**2) +  ((y[49:50:, 1]-(37.1/200))**2) + ((y[49:50:, 6]-(13.63/200))**2) + ((y[49:50:, 3]-(4.32/200))**2) + (((y[49:50:, 0]-(334/200))**2) + ((y[49:50:, 9]-(75.7/200))**2) + ((y[49:50:, 7]-(46.95/200))**2))
    # multiply it by how many filters were passed
    all_filters = [2.25/100 <= (y[49:50:, 5]*200) <= 449.8*100, 2.7/100 <= (y[49:50:, 2]*200) <= 452*100, 0.34/100 <= (y[49:50:, 4]*200) <= 40.4*100, 4.7/100 <= (y[49:50:, 1]*200) <= 69.5*100, 0.26/10 <= (y[49:50:, 6]*200) <= 27.0*100,  0.11/100 <= (y[49:50:, 3]*200) <= 8.42*10, 267/1-0 <= (y[49:50:, 0]*200) <= 401*100, 53.4/100 <= (y[49:50:, 9]*200) <= 97.9*100, 4.9/100 <= (y[49:50:, 7]*200) <= 89*100]
    result2 = sum(all_filters)
    # return total number of filters minus filters passed
    result3 = 9-result2
    result = result3 * result1
    print(result)
    return{'loss':result, 'status': STATUS_OK}
    
    # vsc doing filters * middle of filter
    # terminal doing only middle of filter

# arableGrass,    fox,    rabbits,   raptors,           reptilesAmphibians, roeDeer,      songbirdsWaterfowl, thornyScrub, wetland, woodland
# 267-401      4.7-69.5    2.7-452   0.11 to 8.42          4.32-40.4        2.25-449.8      0.26-27               4.9-89            53.4-97.9

param_hyperopt= {
    # starting value bounds
    'X0': {
    # 'largeHerb': hp.choice('largeHerb_x', (0,0)),
    'roeDeer' : hp.uniform('roeDeer_x', 2.25/200, 112.5/200), 
    # 'tamworthPig' : hp.choice('tamworthPig_x', (0,0)),
    # 'beaver': hp.choice('beaver_x', (0,0)),
    'rabbits': hp.uniform('rabbit_x', 2.7/200, 452/200),
    'fox': hp.uniform('fox_x', 4.7/200,69.5/200),
    'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl_x', 0.15/200,27/200),
    'raptors': hp.uniform('raptors_x', 0.11/200,8.42/200),
    'reptilesAmphibians': hp.uniform('reptilesAmphibians_x', 0.34/200,40.4/200),
    'arableGrass': hp.uniform('arableGrass_x', 338/200,401/200),
    'woodland': hp.uniform('woodland_x', 22.3/200,62.3/200),
    'thornyScrub': hp.uniform('thornyScrub_x', 4.5/200,44.5/200),
    'wetland': hp.uniform('wetland_x', 15/200,24/200)
    },
    # growth rate bounds
    'growthRate': {
    # 'largeHerb': hp.choice('growth_herb', (0,0)),
    'roeDeer' : hp.uniform('growth_roe', 0,1), 
    # 'tamworthPig' : hp.choice('growth_tam', (0,0)),
    # 'beaver': hp.choice('growth_beaver', (0,0)),
    'rabbits': hp.uniform('growth_rabbit',0, 1),
    'fox': hp.uniform('growth_fox', 0,1),
    'songbirdsWaterfowl': hp.uniform('growth_songbird', 0,1),
    'raptors': hp.uniform('growth_raptor', 0,1),
    'reptilesAmphibians': hp.uniform('growth_ampib',0,1),
    'arableGrass': hp.uniform('growth_grass', 0,1),
    'woodland': hp.uniform('growth_wood', 0,1),
    'thornyScrub': hp.uniform('growth_scrub', 0,1),
    'wetland': hp.choice('growth_water', (0,0))
    },
    # interaction matrix bounds
    'interact': 
    {
    'roeDeer': {'roeDeer': hp.choice('roeDeer2', (0,0)), 'rabbits': hp.choice('rabbits2', (0,0)),  'fox': hp.choice('fox2', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl2', (0,0)), 'raptors': hp.choice('raptors2', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians2', (0,0)), 'arableGrass': hp.uniform('arableGrass2', -1,0), 'woodland': hp.uniform('woodland2', -1,0), 'thornyScrub': hp.uniform('thornyScrub2', -1,0), 'wetland': hp.choice('wetland2', (0,0))},
    'rabbits': {'roeDeer':hp.choice('roeDeer5', (0,0)), 'rabbits': hp.choice('rabbits5', (0,0)),  'fox': hp.uniform('fox5',0,1),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl5', (0,0)), 'raptors': hp.uniform('raptors5', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians5', (0,0)), 'arableGrass': hp.uniform('arableGrass5',-1,0), 'woodland': hp.uniform('woodland5', -1,0), 'thornyScrub': hp.uniform('thornyScrub5', -1,0), 'wetland': hp.choice('wetland5', (0,0))},
    'fox': {'roeDeer':hp.choice('roeDeer6', (0,0)), 'rabbits': hp.uniform('rabbits6', -1,0),  'fox': hp.choice('fox6', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl6', -1,0), 'raptors': hp.choice('raptors6', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians6', -1,0), 'arableGrass': hp.choice('arableGrass6', (0,0)), 'woodland': hp.choice('woodland6', (0,0)), 'thornyScrub': hp.choice('thornyScrub6', (0,0)), 'wetland': hp.choice('wetland6', (0,0))},
    'songbirdsWaterfowl': {'roeDeer':hp.choice('roeDeer7', (0,0)), 'rabbits': hp.choice('rabbits7', (0,0)),  'fox': hp.uniform('fox7', 0,1),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl7', (0,0)), 'raptors': hp.uniform('raptors7', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians7', (0,0)), 'arableGrass': hp.choice('arableGrass7', (0,0)), 'woodland': hp.choice('woodland7', (0,0)), 'thornyScrub': hp.choice('thornyScrub7', (0,0)), 'wetland': hp.choice('wetland7', (0,0))},
    'raptors': {'roeDeer':hp.choice('roeDeer8', (0,0)), 'rabbits': hp.uniform('rabbits8', -1,0),  'fox': hp.choice('fox8', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl8', -1,0), 'raptors': hp.choice('raptors8', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians8', -1,0), 'arableGrass': hp.choice('arableGrass8', (0,0)), 'woodland': hp.choice('woodland8', (0,0)), 'thornyScrub': hp.choice('thornyScrub8', (0,0)), 'wetland': hp.choice('wetland8', (0,0))},
    'reptilesAmphibians': {'roeDeer':hp.choice('roeDeer9', (0,0)), 'rabbits': hp.choice('rabbits9', (0,0)),  'fox': hp.uniform('fox9', 0,1),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl9', (0,0)), 'raptors': hp.uniform('raptors9', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians9', (0,0)), 'arableGrass': hp.choice('arableGrass9', (0,0)), 'woodland': hp.choice('woodland9', (0,0)), 'thornyScrub': hp.choice('thornyScrub9', (0,0)), 'wetland': hp.choice('wetland9', (0,0))},
    'arableGrass': {'roeDeer':hp.uniform('roeDeer10', 0,1), 'rabbits': hp.uniform('rabbits10', 0,1),  'fox': hp.choice('fox10', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl10', (0,0)), 'raptors': hp.choice('raptors10', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians10',(0,0)), 'arableGrass': hp.choice('arableGrass10', (0,0)), 'woodland': hp.choice('woodland10', (0,0)), 'thornyScrub': hp.choice('thornyScrub10', (0,0)), 'wetland': hp.choice('wetland10', (0,0))},
    'woodland': {'roeDeer':hp.uniform('roeDeer11', 0,1), 'rabbits': hp.uniform('rabbits11', 0,1),  'fox': hp.choice('fox11', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl11', 0,1), 'raptors': hp.uniform('raptors11', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians11', (0,0)), 'arableGrass': hp.uniform('arableGrass11', -1,0), 'woodland': hp.choice('woodland11', (0,0)), 'thornyScrub': hp.uniform('thornyScrub11', -1,0), 'wetland': hp.choice('wetland11', (0,0))},
    'thornyScrub': {'roeDeer':hp.uniform('roeDeer12', 0,1),  'rabbits': hp.uniform('rabbits12', 0,1),  'fox': hp.choice('fox12', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl12', 0,1), 'raptors': hp.choice('raptors12', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians12', 0,1), 'arableGrass': hp.uniform('arableGrass12', -1,0), 'woodland': hp.uniform('woodland12', 0,1), 'thornyScrub': hp.choice('thornyScrub12', (0,0)), 'wetland': hp.choice('wetland12', (0,0))},
    'wetland': {'roeDeer':hp.uniform('roeDeer13', 0,1),  'rabbits': hp.choice('rabbits13', (0,0)),  'fox': hp.choice('fox13', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl13', 0,1), 'raptors': hp.uniform('raptors13', 0,1),  'reptilesAmphibians': hp.uniform('reptilesAmphibians13', 0,1), 'arableGrass': hp.uniform('arableGrass13', -1,0), 'woodland': hp.uniform('woodland13', -1,0), 'thornyScrub': hp.uniform('thornyScrub13', -1,0), 'wetland': hp.choice('wetland13', (0,0))}
   
   
#    'largeHerb': {'largeHerb': hp.choice('largeHerb1', (0,0)), 'roeDeer':hp.choice('roeDeer1', (0,0)), 'tamworthPig': hp.choice('tamworthPig1', (0,0)), 'beaver': hp.choice('beaver1', (0,0)), 'rabbits': hp.choice('rabbits1', (0,0)),  'fox': hp.choice('fox1', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl1', (0,0)), 'raptors': hp.choice('raptors1', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians1', (0,0)), 'arableGrass': hp.uniform('arableGrass1', -1,0), 'woodland': hp.uniform('woodland1', -1,0), 'thornyScrub': hp.uniform('thornyScrub1', -1,0), 'wetland': hp.choice('wetland1', (0,0))},
#     'roeDeer': {'largeHerb': hp.choice('largeHerb2', (0,0)), 'roeDeer': hp.choice('roeDeer2', (0,0)), 'tamworthPig': hp.choice('tamworthPig2', (0,0)), 'beaver': hp.choice('beaver2', (0,0)), 'rabbits': hp.choice('rabbits2', (0,0)),  'fox': hp.choice('fox2', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl2', (0,0)), 'raptors': hp.choice('raptors2', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians2', (0,0)), 'arableGrass': hp.uniform('arableGrass2', -1,0), 'woodland': hp.uniform('woodland2', -1,0), 'thornyScrub': hp.uniform('thornyScrub2', -1,0), 'wetland': hp.choice('wetland2', (0,0))},
#     'tamworthPig': {'largeHerb': hp.choice('largeHerb3', (0,0)), 'roeDeer': hp.choice('roeDeer3', (0,0)), 'tamworthPig': hp.choice('tamworthPig3', (0,0)), 'beaver': hp.choice('beaver3', (0,0)), 'rabbits': hp.choice('rabbits3', (0,0)),  'fox': hp.choice('fox3', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl3', (0,0)), 'raptors': hp.choice('raptors3', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians3', -1,0), 'arableGrass': hp.uniform('arableGrass3', -1,0), 'woodland': hp.uniform('woodland3', -1,0), 'thornyScrub': hp.uniform('thornyScrub3', -1,0), 'wetland': hp.choice('wetland3', (0,0))},
#     'beaver': {'largeHerb': hp.choice('largeHerb4', (0,0)), 'roeDeer':hp.choice('roeDeer4', (0,0)), 'tamworthPig': hp.choice('tamworthPig4', (0,0)), 'beaver': hp.choice('beaver4', (0,0)), 'rabbits': hp.choice('rabbits4', (0,0)),  'fox': hp.choice('fox4', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl4',(0,0)), 'raptors': hp.choice('raptors4', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians4', (0,0)), 'arableGrass': hp.uniform('arableGrass4', -1,0), 'woodland': hp.uniform('woodland4', -1,0), 'thornyScrub': hp.choice('thornyScrub4', (0,0)), 'wetland': hp.uniform('wetland4', 0,1)},
#     'rabbits': {'largeHerb': hp.choice('largeHerb5', (0,0)), 'roeDeer':hp.choice('roeDeer5', (0,0)), 'tamworthPig': hp.choice('tamworthPig5', (0,0)), 'beaver': hp.choice('beaver5', (0,0)), 'rabbits': hp.choice('rabbits5', (0,0)),  'fox': hp.uniform('fox5',0,1),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl5', (0,0)), 'raptors': hp.uniform('raptors5', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians5', (0,0)), 'arableGrass': hp.uniform('arableGrass5',-1,0), 'woodland': hp.uniform('woodland5', -1,0), 'thornyScrub': hp.uniform('thornyScrub5', -1,0), 'wetland': hp.choice('wetland5', (0,0))},
#     'fox': {'largeHerb': hp.choice('largeHerb6', (0,0)), 'roeDeer':hp.choice('roeDeer6', (0,0)), 'tamworthPig': hp.choice('tamworthPig6', (0,0)), 'beaver': hp.choice('beaver6', (0,0)), 'rabbits': hp.uniform('rabbits6', -1,0),  'fox': hp.choice('fox6', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl6', -1,0), 'raptors': hp.choice('raptors6', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians6', -1,0), 'arableGrass': hp.choice('arableGrass6', (0,0)), 'woodland': hp.choice('woodland6', (0,0)), 'thornyScrub': hp.choice('thornyScrub6', (0,0)), 'wetland': hp.choice('wetland6', (0,0))},
#     'songbirdsWaterfowl': {'largeHerb': hp.choice('largeHerb7', (0,0)), 'roeDeer':hp.choice('roeDeer7', (0,0)), 'tamworthPig': hp.choice('tamworthPig7', (0,0)), 'beaver': hp.choice('beaver7', (0,0)), 'rabbits': hp.choice('rabbits7', (0,0)),  'fox': hp.uniform('fox7', 0,1),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl7', (0,0)), 'raptors': hp.uniform('raptors7', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians7', (0,0)), 'arableGrass': hp.choice('arableGrass7', (0,0)), 'woodland': hp.choice('woodland7', (0,0)), 'thornyScrub': hp.choice('thornyScrub7', (0,0)), 'wetland': hp.choice('wetland7', (0,0))},
#     'raptors': {'largeHerb': hp.choice('largeHerb8', (0,0)), 'roeDeer':hp.choice('roeDeer8', (0,0)), 'tamworthPig': hp.choice('tamworthPig8', (0,0)), 'beaver': hp.choice('beaver8', (0,0)), 'rabbits': hp.uniform('rabbits8', -1,0),  'fox': hp.choice('fox8', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl8', -1,0), 'raptors': hp.choice('raptors8', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians8', -1,0), 'arableGrass': hp.choice('arableGrass8', (0,0)), 'woodland': hp.choice('woodland8', (0,0)), 'thornyScrub': hp.choice('thornyScrub8', (0,0)), 'wetland': hp.choice('wetland8', (0,0))},
#     'reptilesAmphibians': {'largeHerb': hp.choice('largeHerb9', (0,0)), 'roeDeer':hp.choice('roeDeer9', (0,0)), 'tamworthPig': hp.uniform('tamworthPig9', 0,1), 'beaver': hp.choice('beaver9', (0,0)), 'rabbits': hp.choice('rabbits9', (0,0)),  'fox': hp.uniform('fox9', 0,1),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl9', (0,0)), 'raptors': hp.uniform('raptors9', 0,1),  'reptilesAmphibians': hp.choice('reptilesAmphibians9', (0,0)), 'arableGrass': hp.choice('arableGrass9', (0,0)), 'woodland': hp.choice('woodland9', (0,0)), 'thornyScrub': hp.choice('thornyScrub9', (0,0)), 'wetland': hp.choice('wetland9', (0,0))},
#     'arableGrass': {'largeHerb': hp.uniform('largeHerb10', 0,1), 'roeDeer':hp.uniform('roeDeer10', 0,1), 'tamworthPig': hp.uniform('tamworthPig10', 0,1), 'beaver': hp.uniform('beaver10', 0,1), 'rabbits': hp.uniform('rabbits10', 0,1),  'fox': hp.choice('fox10', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl10', (0,0)), 'raptors': hp.choice('raptors10', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians10',(0,0)), 'arableGrass': hp.choice('arableGrass10', (0,0)), 'woodland': hp.choice('woodland10', (0,0)), 'thornyScrub': hp.choice('thornyScrub10', (0,0)), 'wetland': hp.choice('wetland10', (0,0))},
#     'woodland': {'largeHerb': hp.uniform('largeHerb11', 0,1), 'roeDeer':hp.uniform('roeDeer11', 0,1), 'tamworthPig': hp.uniform('tamworthPig11', 0,1), 'beaver': hp.uniform('beaver11', 0,1), 'rabbits': hp.uniform('rabbits11', 0,1),  'fox': hp.choice('fox11', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl11', (0,0)), 'raptors': hp.choice('raptors11', (0,0)),  'reptilesAmphibians': hp.choice('reptilesAmphibians11', (0,0)), 'arableGrass': hp.uniform('arableGrass11', -1,0), 'woodland': hp.choice('woodland11', (0,0)), 'thornyScrub': hp.uniform('thornyScrub11', -1,0), 'wetland': hp.choice('wetland11', (0,0))},
#     'thornyScrub': {'largeHerb': hp.uniform('largeHerb12', 0,1), 'roeDeer':hp.uniform('roeDeer12', 0,1), 'tamworthPig': hp.uniform('tamworthPig12', 0,1), 'beaver': hp.choice('beaver12', (0,0)), 'rabbits': hp.uniform('rabbits12', 0,1),  'fox': hp.choice('fox12', (0,0)),  'songbirdsWaterfowl': hp.choice('songbirdsWaterfowl12', (0,0)), 'raptors': hp.choice('raptors12', (0,0)),  'reptilesAmphibians': hp.uniform('reptilesAmphibians12', 0,1), 'arableGrass': hp.uniform('arableGrass12', -1,0), 'woodland': hp.uniform('woodland12', 0,1), 'thornyScrub': hp.choice('thornyScrub12', (0,0)), 'wetland': hp.choice('wetland12', (0,0))},
#     'wetland': {'largeHerb': hp.uniform('largeHerb13', 0,1), 'roeDeer':hp.uniform('roeDeer13', 0,1), 'tamworthPig': hp.uniform('tamworthPig13', 0,1), 'beaver': hp.uniform('beaver13', 0,1), 'rabbits': hp.choice('rabbits13', (0,0)),  'fox': hp.choice('fox13', (0,0)),  'songbirdsWaterfowl': hp.uniform('songbirdsWaterfowl13', 0,1), 'raptors': hp.uniform('raptors13', 0,1),  'reptilesAmphibians': hp.uniform('reptilesAmphibians13', 0,1), 'arableGrass': hp.uniform('arableGrass13', -1,0), 'woodland': hp.uniform('woodland13', -1,0), 'thornyScrub': hp.uniform('thornyScrub13', -1,0), 'wetland': hp.choice('wetland13', (0,0))}
    }
}
                   
# print (hyperopt.pyll.stochastic.sample(param_hyperopt))

trials = Trials()
optimization = fmin(objectiveFunction, param_hyperopt, trials=trials, algo = tpe.suggest, max_evals = 750)
print(optimization)
# print (space_eval(param_hyperopt, optimization))
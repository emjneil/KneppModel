
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
species = ['arableGrass','fox','rabbits','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, interaction_strength_chunk, rowContents_growth):
    # define new array to return
    output_array = []
    # scale habitat data
    # constrain output array so sumHabitat is right before return; constrain so ouput + X[0] = 

    # make X zero if it falls under some threshold
    for n, i in enumerate(X):
        if i < 1e-8:
            X[n] = 0
    # loop through all the species, apply generalized Lotka-Volterra equation
    for outer_index, outer_species in enumerate(species): 
        # grab one set of growth rates at a time
        amount = rowContents_growth[outer_species] * X[outer_index]
        for inner_index, inner_species in enumerate(species):
            # grab one set of interaction matrices at a time
            amount2 = amount + (interaction_strength_chunk[outer_species][inner_species] * X[outer_index] * X[inner_index]) 
        # append values to output_array
        output_array.append(amount2)
    # scale habitat data - this needs to go into equation
    sumHabitat_ODE = X[0] + X[4] + X[5]
    X[0] = (X[0]/sumHabitat_ODE) * 4.45
    X[4] = (X[4]/sumHabitat_ODE) * 4.45
    X[5] = (X[5]/sumHabitat_ODE) * 4.45
    print(sumHabitat_ODE)
    return output_array


def objectiveFunction(x):
    # extract starting values
    X0 = pd.DataFrame(x['X0'].values(), index=x['X0'].keys())
    X0 = X0.values.tolist()
    X0 = sum(X0, [])
    # keep habitat types scaled to the total size (4.45km2)
    sumHabitat = X0[0] +  X0[4] + X0[5]
    X0[0] = (X0[0]/sumHabitat) * 4.45
    X0[4] = (X0[4]/sumHabitat) * 4.45
    X0[5] = (X0[5]/sumHabitat) * 4.45
    # extract growth rates
    growthRate = pd.DataFrame(x['growthRate'].values(), index=x['growthRate'].keys())
    rowContents_growth = growthRate.T
    rowContents_growth = rowContents_growth.squeeze()
    # add interaction values of 0 (where no choices are being made between values)
    x['interact']['arableGrass']['woodland'] = 0
    x['interact']['arableGrass']['thornyScrub'] = 0
    x['interact']['rabbits']['roeDeer'] = 0
    x['interact']['roeDeer']['rabbits'] = 0
    x['interact']['roeDeer']['fox'] = 0
    x['interact']['woodland']['fox'] = 0
    x['interact']['thornyScrub']['fox'] = 0
    x['interact']['arableGrass']['fox'] = 0
    x['interact']['fox']['arableGrass'] = 0
    x['interact']['fox']['thornyScrub'] = 0
    x['interact']['fox']['woodland'] = 0
    x['interact']['fox']['roeDeer'] = 0
    # extract interaction strength
    interaction_strength_chunk = pd.DataFrame(x['interact'].values(), index=x['interact'].keys(), columns=x['interact'].keys())
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(interaction_strength_chunk, rowContents_growth), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters)
    # print(y[49:50:,])
    # make sure habitat types add to 4.45km2
    print(y[49:50:,0]+y[49:50:,4]+y[49:50:,5])
    result = (((y[49:50:, 0]-3.6)**2) +  ((y[49:50:, 1]-1.2)**2) + ((y[49:50:, 2]-1130.5)**2) + ((y[49:50:, 3]-6.7)**2)  + ((y[49:50:, 4]-.5)**2) + ((y[49:50:, 5]-.8)**2))
    # print(result)
    return{'loss':result, 'status': STATUS_OK}

# order of outputs   
# ['arableGrass',    'fox'                  'rabbits'         'roeDeer',     'thornyScrub',            'woodland'])
#   2.9-4.2 (3.6)    0.22 to 2.2 (1.2)  1-2260 (1130.5)      4.9-9(6.7)     0.05 to 0.89 (0.5)      0.53-0.97 (0.8)

param_hyperopt= {
    # starting value bounds
    'X0': {
    'roeDeer' : hp.uniform('roeDeer_x', 1.3, 5.3), 
    'rabbits' : hp.uniform('rabbits_x', 1, 2260),
    'arableGrass': hp.uniform('arableGrass_x', 2.1,4),
    'woodland': hp.uniform('woodland_x', 0.5,0.6),
    'thornyScrub': hp.uniform('thornyScrub_x', 0.1,0.5),
    'fox' : hp.uniform('fox_x', 0.2,2.2) 
    },
    # growth rate bounds
    'growthRate': {
    'roeDeer' : hp.uniform('growth_roe', 0,1), 
    'fox' : hp.uniform('growth_fox', 0,1), 
    'rabbits' : hp.uniform('growth_rabbits', 0,1),
    'arableGrass': hp.uniform('growth_grass', 0,1),
    'woodland': hp.uniform('growth_wood', 0,1),
    'thornyScrub': hp.uniform('growth_scrub', 0,1)
    },
    # interaction matrix bounds
    'interact': 
    {
    'arableGrass': {'roeDeer':hp.uniform('roeDeer1', 0,1), 'arableGrass':hp.uniform('arableGrass1', -1,0), 'rabbits':hp.uniform('rabbits1', 0,1)},
    'fox': {'rabbits':hp.uniform('rabbits2', -1,0), 'fox': hp.uniform('fox2', -1,0),},
    'rabbits': {'arableGrass':hp.uniform('arableGrass3', -1,0), 'rabbits':hp.uniform('rabbits3', -1,0),'woodland':hp.uniform('woodland3', -1,0), 'thornyScrub':hp.uniform('thornyScrub3', -1,0), 'fox':hp.uniform('fox3',0,1)},
    'roeDeer': {'arableGrass': hp.uniform('arableGrass4', -1,0), 'roeDeer': hp.uniform('roeDeer4', -1,0), 'woodland': hp.uniform('woodland4', -1,0), 'thornyScrub': hp.uniform('thornyScrub4', -1,0)},
    'thornyScrub': {'roeDeer':hp.uniform('roeDeer5', 0,1), 'thornyScrub':hp.uniform('thornyScrub5', -1,0), 'rabbits':hp.uniform('rabbits5', 0,1), 'arableGrass': hp.uniform('arableGrass5', -1,0), 'woodland': hp.uniform('woodland5', 0,1)},
    'woodland': {'roeDeer':hp.uniform('roeDeer6', 0,1),  'woodland': hp.uniform('woodland6', -1,0), 'rabbits':hp.uniform('rabbits6', 0,1), 'arableGrass': hp.uniform('arableGrass6', -1,0),  'thornyScrub': hp.uniform('thornyScrub6', -1,0)}
    }
}
                   
trials = Trials()
optimization = fmin(objectiveFunction, param_hyperopt, trials=trials, algo = tpe.suggest, max_evals = 5000)
print(optimization)

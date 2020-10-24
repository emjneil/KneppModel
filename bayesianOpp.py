# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


# # # # # --------- Hyperopt Bayesian optimization ----------- # # # # # # # 
  
# store species in a list
species = ['arableGrass','organicCarbon','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    return X * (r + np.matmul(A, X))


def objectiveFunction(x):
    # extract growth rates
    growthRate = pd.DataFrame(x['growthRate'].values(), index=x['growthRate'].keys())
    growthRate = growthRate.values.tolist()
    r = sum(growthRate, [])
    # add interaction values of 0 (where no choices are being made between values)
    x['interact']['arableGrass']['organicCarbon'] = 0
    x['interact']['roeDeer']['organicCarbon'] = 0
    x['interact']['thornyScrub']['arableGrass'] = 0
    x['interact']['thornyScrub']['organicCarbon'] = 0
    x['interact']['woodland']['arableGrass'] = 0
    x['interact']['woodland']['organicCarbon'] = 0
    # add X0
    X0 = [1] * len(species)
    # extract interaction strength
    A = pd.DataFrame(x['interact'].values(), index=x['interact'].keys(), columns=x['interact'].keys())
    A = A.to_numpy()
    t = np.linspace(0, 5, 50)
    results = solve_ivp(ecoNetwork, (0, 5), X0,  t_eval = t, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters)
    print((y[49:50,:]))
    result = (((y[49:50, 0]-0.86)**2) +  ((y[49:50, 1]-1.4)**2) + ((y[49:50, 2]-2.2)**2) + ((y[49:50, 3]-10.9)**2) + ((y[49:50, 4]-0.91)**2))
    print(result)    
    return{'loss':result, 'status': STATUS_OK}

# order of outputs   
# ['arableGrass',   orgCarb   'roeDeer',     'thornyScrub',  'woodland'])
#   0.86            1.4        2.2              10.9               0.91

param_hyperopt= {
    # growth rate bounds
    'growthRate': {
    'arableGrass': hp.uniform('growth_grass', 0,1),
    'organicCarbon': hp.uniform('growth_carbon',0,1),
    'roeDeer' : hp.uniform('growth_roe', 0,1), 
    'thornyScrub': hp.uniform('growth_scrub', 0,1),
    'woodland': hp.uniform('growth_wood', 0,1)
    },
    # interaction matrix bounds
    'interact': 
    {
    'arableGrass': {'arableGrass':hp.uniform('arableGrass1', -1,0), 'roeDeer':hp.uniform('roeDeer1', -1,0), 'thornyScrub':hp.uniform('thornyScrub1', -0.1,0), 'woodland':hp.uniform('woodland1', -0.1,0)},
    'organicCarbon':{'arableGrass':hp.uniform('arableGrass3', 0,1),'organicCarbon':hp.uniform('organicCarbon3',-1,0), 'roeDeer': hp.uniform('roeDeer3', 0,1),  'thornyScrub': hp.uniform('thornyScrub3', 0,1), 'woodland': hp.uniform('woodland3', 0,1)},
    'roeDeer': {'arableGrass': hp.uniform('arableGrass5', 0,1), 'roeDeer': hp.uniform('roeDeer5', -1,0),  'thornyScrub': hp.uniform('thornyScrub5', 0,1), 'woodland': hp.uniform('woodland5', 0,1)},
    'thornyScrub': {'roeDeer':hp.uniform('roeDeer6', -1,0), 'thornyScrub':hp.uniform('thornyScrub6', -1,0), 'woodland': hp.uniform('woodland6', -1,0)},
    'woodland': {'roeDeer':hp.uniform('roeDeer7', -1,0),  'thornyScrub': hp.uniform('thornyScrub7', 0,1), 'woodland': hp.uniform('woodland7', -1,0)}
    }
}
                   
trials = Trials()
optimization = fmin(objectiveFunction, param_hyperopt, trials=trials, algo = tpe.suggest, max_evals = 5000)
print(optimization)

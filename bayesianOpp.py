# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
# from sklearn.preprocessing import StandardScaler


# # # # # --------- Hyperopt Bayesian optimization ----------- # # # # # # # 
  
# store species in a list
species = ['arableGrass','fox','organicCarbon','rabbits','roeDeer','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    # return array
    # return X * (r + np.matmul(A, X))
    return r * X + np.matmul(A, X) * X


def objectiveFunction(x):
    # extract starting values
    X0 = pd.DataFrame(x['X0'].values(), index=x['X0'].keys())
    # X0 = X0.to_numpy()
    X0 = X0.values.tolist()
    X0 = sum(X0, [])
    # normalize it
    max_value = max(X0)
    X0[:] = [x / max_value for x in X0]
    # extract growth rates
    growthRate = pd.DataFrame(x['growthRate'].values(), index=x['growthRate'].keys())
    growthRate = growthRate.values.tolist()
    r = sum(growthRate, [])
    # add in organic carbon which has growth rate of zero
    # r = growthRate[:]
    # r.insert(2, 0)
    # add interaction values of 0 (where no choices are being made between values)
    x['interact']['arableGrass']['fox'] = 0
    x['interact']['arableGrass']['organicCarbon'] = 0
    x['interact']['fox']['arableGrass'] = 0
    x['interact']['fox']['organicCarbon'] = 0
    x['interact']['fox']['roeDeer'] = 0
    x['interact']['fox']['thornyScrub'] = 0
    x['interact']['fox']['woodland'] = 0
    x['interact']['rabbits']['organicCarbon'] = 0
    x['interact']['rabbits']['roeDeer'] = 0
    x['interact']['roeDeer']['fox'] = 0
    x['interact']['roeDeer']['organicCarbon'] = 0
    x['interact']['roeDeer']['rabbits'] = 0
    x['interact']['thornyScrub']['arableGrass'] = 0
    x['interact']['thornyScrub']['fox'] = 0
    x['interact']['thornyScrub']['organicCarbon'] = 0
    x['interact']['woodland']['arableGrass'] = 0
    x['interact']['woodland']['fox'] = 0
    x['interact']['woodland']['organicCarbon'] = 0
    # extract interaction strength
    A = pd.DataFrame(x['interact'].values(), index=x['interact'].keys(), columns=x['interact'].keys())
    A = A.to_numpy()
    t = np.linspace(0, 10, 50)
    results = solve_ivp(ecoNetwork, (0, 10), X0,  t_eval = t, args=(A, r), method = 'RK23')
    # reshape the outputs
    y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters)
    print((y[49:50,:])*max_value)
    result1 = (((y[49:50, 0]-3.6/max_value)**2) +  ((y[49:50, 1]-7.1/max_value)**2) + ((y[49:50, 2]-1.5/max_value)**2) + ((y[49:50, 3]-1131/max_value)**2) + ((y[49:50, 4]-6.7/max_value)**2)  + ((y[49:50, 5]-0.6/max_value)**2) + ((y[49:50, 6]-0.5/max_value)**2))
    # multiply it by how many filters were passed
    all_filters = [3.1 <= (y[49:50:, 0]*max_value) <= 4, 0.22 <= (y[49:50:, 1]*max_value) <= 13.9, 1 <= (y[49:50:, 2]*max_value) <= 2, 1 <= (y[49:50:, 3]*max_value) <= 2260, 4.9 <= (y[49:50:, 4]*max_value) <= 9,  0.2 <= (y[49:50:, 5]*max_value) <= 0.9,  0.2 <= (y[49:50:, 6]*max_value) <= 0.8]
    result2 = sum(all_filters)
    # return total number of filters minus filters passed; or +1 filter so it keeps getting lower instead of to 0
    result3 = 8-result2
    result = result3 * result1
    print(result)    
    return{'loss':result, 'status': STATUS_OK}

# order of outputs   
# ['arableGrass',    'fox'                orgCarb            'rabbits'         'roeDeer',     'thornyScrub',     'woodland'])
#   3.1 to 4 (3.6)    0.22 to 13.9 (7.1)     1-2(1.5)       1-2260 (1130.5)      4.9-9(6.7)      0.2-0.9 (0.6)      0.2-0.8 (0.5)

param_hyperopt= {
    # starting value bounds
    'X0': {
    'arableGrass': hp.uniform('arableGrass_x', 3.6,4),
    'fox' : hp.uniform('fox_x', 0.2,2.2), 
    'organicCarbon': hp.uniform('organicC_x',1,1.1),
    'rabbits' : hp.uniform('rabbits_x', 1, 2260),
    'roeDeer' : hp.uniform('roeDeer_x', 1.3, 5.3), 
    'thornyScrub': hp.uniform('thornyScrub_x', 0.1,0.5),
    'woodland': hp.uniform('woodland_x', 0.2,0.7)
    },
    # growth rate bounds
    'growthRate': {
    'arableGrass': hp.uniform('growth_grass', 0,1),
    'fox' : hp.uniform('growth_fox', 0,1), 
    'organicCarbon': hp.uniform('growth_carbon',0,1),
    'rabbits' : hp.uniform('growth_rabbits', 0,1),
    'roeDeer' : hp.uniform('growth_roe', 0,1), 
    'thornyScrub': hp.uniform('growth_scrub', 0,1),
    'woodland': hp.uniform('growth_wood', 0,1)
    },
    # interaction matrix bounds
    'interact': 
    {
    'arableGrass': {'arableGrass':hp.uniform('arableGrass1', -1,0), 'rabbits':hp.uniform('rabbits1', -1,0), 'roeDeer':hp.uniform('roeDeer1', -1,0), 'thornyScrub':hp.uniform('thornyScrub1', -1,0), 'woodland':hp.uniform('woodland1', -1,0)},
    'fox': {'fox': hp.uniform('fox2', -1,0), 'rabbits':hp.uniform('rabbits2', 0,1)},
    'organicCarbon':{'arableGrass':hp.uniform('arableGrass3', 0,1),'fox':hp.uniform('fox3',0,1),'organicCarbon':hp.uniform('organicCarbon3',-1,0),'rabbits':hp.uniform('rabbits3', 0,1), 'roeDeer': hp.uniform('roeDeer3', 0,1),  'thornyScrub': hp.uniform('thornyScrub3', 0,1), 'woodland': hp.uniform('woodland3', 0,1)},
    'rabbits': {'arableGrass':hp.uniform('arableGrass4', 0,1), 'fox':hp.uniform('fox4',-1,0),  'rabbits':hp.uniform('rabbits4', -1,0),'thornyScrub':hp.uniform('thornyScrub4', 0,1), 'woodland':hp.uniform('woodland4', 0,1)},
    'roeDeer': {'arableGrass': hp.uniform('arableGrass5', 0,1), 'roeDeer': hp.uniform('roeDeer5', -1,0),  'thornyScrub': hp.uniform('thornyScrub5', 0,1), 'woodland': hp.uniform('woodland5', 0,1)},
    'thornyScrub': {'rabbits':hp.uniform('rabbits6', -1,0), 'roeDeer':hp.uniform('roeDeer6', -1,0), 'thornyScrub':hp.uniform('thornyScrub6', -1,0), 'woodland': hp.uniform('woodland6', -1,0)},
    'woodland': {'rabbits':hp.uniform('rabbits7', -1,0),'roeDeer':hp.uniform('roeDeer7', -1,0),  'thornyScrub': hp.uniform('thornyScrub7', 0,1), 'woodland': hp.uniform('woodland7', -1,0)}
    }
}
                   
trials = Trials()
optimization = fmin(objectiveFunction, param_hyperopt, trials=trials, algo = tpe.suggest, max_evals = 5000)
print(optimization)

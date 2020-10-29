# ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

from scipy.integrate import solve_ivp
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK


# # # # # --------- Hyperopt Bayesian optimization ----------- # # # # # # # 
  
# store species in a list
species = ['arableGrass','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    return X * (r + np.matmul(A, X))


def objectiveFunction(x):
    # extract growth rates
    r = [0.7, 0, 0.003, 0.5, 0, 0.8, 0.05]
    # add interaction values of 0 (where no choices are being made between values)
    x['interact']['arableGrass']['arableGrass'] = -0.8
    x['interact']['arableGrass']['organicCarbon'] = 0
    x['interact']['arableGrass']['roeDeer'] = -0.003
    x['interact']['arableGrass']['thornyScrub'] = -0.0000002
    x['interact']['arableGrass']['woodland'] = -0.07
    x['interact']['largeHerb']['organicCarbon'] = 0
    x['interact']['largeHerb']['roeDeer'] = 0
    x['interact']['largeHerb']['tamworthPig'] = 0
    x['interact']['organicCarbon']['arableGrass'] = 0.2
    x['interact']['organicCarbon']['organicCarbon'] = -0.7
    x['interact']['organicCarbon']['roeDeer'] = 0.4
    x['interact']['organicCarbon']['thornyScrub'] = 0.005
    x['interact']['organicCarbon']['woodland'] = 0.1
    x['interact']['roeDeer']['arableGrass'] = 0.4
    x['interact']['roeDeer']['largeHerb'] = 0
    x['interact']['roeDeer']['organicCarbon'] = 0
    x['interact']['roeDeer']['roeDeer'] = -0.7
    x['interact']['roeDeer']['tamworthPig'] = 0
    x['interact']['roeDeer']['thornyScrub'] = 0.07
    x['interact']['roeDeer']['woodland'] = 0.08
    x['interact']['tamworthPig']['largeHerb'] = 0
    x['interact']['tamworthPig']['organicCarbon'] = 0
    x['interact']['tamworthPig']['roeDeer'] = 0
    x['interact']['thornyScrub']['arableGrass'] = 0
    x['interact']['thornyScrub']['organicCarbon'] = 0
    x['interact']['thornyScrub']['roeDeer'] = -0.02
    x['interact']['thornyScrub']['thornyScrub'] = -0.02
    x['interact']['thornyScrub']['woodland'] = -0.4
    x['interact']['woodland']['arableGrass'] = 0
    x['interact']['woodland']['organicCarbon'] = 0
    x['interact']['woodland']['roeDeer'] = -0.8
    x['interact']['woodland']['thornyScrub'] = 0.2
    x['interact']['woodland']['woodland'] = -0.07
    # add X0
    X0 = [0.86,1,1.4,2.2,1,11.1,0.91]
    # extract interaction strength
    A = pd.DataFrame(x['interact'].values(), index=x['interact'].keys(), columns=x['interact'].keys())
    A = A.to_numpy()
    t = np.linspace(0, 1, 5)
    second_ABC = solve_ivp(ecoNetwork, (0, 1), X0,  t_eval = t, args=(A, r), method = 'RK23')        
    # take those values and re-run for another year, adding forcings
    starting_2010 = second_ABC.y[0:7, 4:5]
    starting_values_2010 = starting_2010.flatten()
    starting_values_2010[1] = X0[1]*2.0
    starting_values_2010[4] = X0[4]*0.5
    # run the model for another year 2010-2011
    third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2011 = third_ABC.y[0:7, 4:5]
    starting_values_2011 = starting_2011.flatten()
    starting_values_2011[1] = X0[1]*1.1
    starting_values_2011[4] = X0[4]*1.3
    # run the model for 2011-2012
    fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2012 = fourth_ABC.y[0:7, 4:5]
    starting_values_2012 = starting_2012.flatten()
    starting_values_2012[1] = X0[1]*1.1
    starting_values_2012[4] = X0[4]*1.5
    # run the model for 2012-2013
    fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2013 = fifth_ABC.y[0:7, 4:5]
    starting_values_2013 = starting_2013.flatten()
    starting_values_2013[1] = X0[1]*1.8
    starting_values_2013[4] = X0[4]*0.18
    # run the model for 2011-2012
    sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2014 = sixth_ABC.y[0:7, 4:5]
    starting_values_2014 = starting_2014.flatten()
    starting_values_2014[1] = X0[1]*0.6
    starting_values_2014[4] = X0[4]*3
    # run the model for 2011-2012
    seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2015 = seventh_ABC.y[0:7, 4:5]
    starting_values_2015 = starting_2015.flatten()
    starting_values_2015[1] = X0[1]*1.2
    starting_values_2015[4] = X0[4]*0.5
    # run the model for 2011-2012
    eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2016 = eighth_ABC.y[0:7, 4:5]
    starting_values_2016 = starting_2016.flatten()
    starting_values_2016[1] = X0[1]*1.21
    starting_values_2016[4] = X0[4]*0.5
    # run the model for 2011-2012
    ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2016,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2017 = ninth_ABC.y[0:7, 4:5]
    starting_values_2017 = starting_2017.flatten()
    starting_values_2017[1] = np.random.uniform(low=0.56,high=2.0)
    starting_values_2017[4] = np.random.uniform(low=0.18,high=3)
    # run the model for 2011-2012
    tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2017,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2018 = tenth_ABC.y[0:7, 4:5]
    starting_values_2018 = starting_2018.flatten()
    starting_values_2018[1] = np.random.uniform(low=0.56,high=2.0)
    starting_values_2018[4] = np.random.uniform(low=0.18,high=3)
    # run the model for 2011-2012
    eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2018,  t_eval = t, args=(A, r), method = 'RK23')
    # concatenate & append all the runs
    combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y))
     # reshape the outputs
    y = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 50).transpose(),1)))
    # choose the final year (we want to compare the final year to the middle of the filters)
    print((y[49:50,:]))
    result = (((y[49:50, 0]-0.72)**2) +  ((y[49:50, 2]-2)**2) + ((y[49:50, 3]-4.1)**2) + ((y[49:50, 5]-28.8)**2) + ((y[49:50, 6]-0.9)**2))
    print(result)    
    return{'loss':result, 'status': STATUS_OK}

# order of outputs   
# ['arableGrass',  largeHerb,   orgCarb   'roeDeer',  tamworthPig,   'thornyScrub',  'woodland'])
#   0.72                          2          4.1                     28.8              0.91

param_hyperopt = {
    # interaction matrix bounds
    'interact': 
    {
    'arableGrass': {'largeHerb':hp.uniform('largeHerb1', -0.1,0), 'tamworthPig':hp.uniform('tamworthPig1', -0.1,0)},
    'largeHerb': {'arableGrass':hp.uniform('arableGrass2', 0,0.1),'largeHerb':hp.uniform('largeHerb2', -1,0), 'thornyScrub':hp.uniform('thornyScrub2', 0,0.1),'woodland':hp.uniform('woodland2', 0,0.1)},
    'organicCarbon':{'largeHerb':hp.uniform('largeHerb3', 0,1), 'tamworthPig':hp.uniform('tamworthPig3', 0,1)},
    'roeDeer': {},
    'tamworthPig': {'arableGrass':hp.uniform('arableGrass5', 0,0.1),'tamworthPig':hp.uniform('tamworthPig5', -1,0),'thornyScrub':hp.uniform('thornyScrub5', 0,0.1),'woodland':hp.uniform('woodland5', 0,0.1)},
    'thornyScrub': {'largeHerb':hp.uniform('largeHerb6', -0.1,0), 'tamworthPig':hp.uniform('tamworthPig6', -0.1 ,0)},
    'woodland': {'largeHerb':hp.uniform('largeHerb7', -1,0), 'tamworthPig':hp.uniform('tamworthPig7', -1,0)}
    }
}
                   
trials = Trials()
optimization = fmin(objectiveFunction, param_hyperopt, trials=trials, algo = tpe.suggest, max_evals = 5000)
print(optimization)

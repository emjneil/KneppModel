# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------

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
from hyperopt import hp, fmin, tpe, space_eval
# from GPyOpt.methods import BayesianOptimization
from skopt import gp_minimize


# # # # # --------- MINIMIZATION ----------- # # # # # # # 

species = ['arableGrass','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']

def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    X[X>1000] = 1000
    # return array
    return X * (r + np.matmul(A, X))


def objectiveFunction(x): 
    # insert growths
    x = np.insert(x,0,0.74)
    x = np.insert(x,2,0.04)
    x = np.insert(x,3,0.34)
    x = np.insert(x,5,0.87)
    x = np.insert(x,6,0.45)
    # interaction
    x = np.insert(x,7,-0.34)
    x = np.insert(x,9,0)
    x = np.insert(x,10,0.03)
    x = np.insert(x,12,-0.03)
    x = np.insert(x,13,-0.38)
    x = np.insert(x,16,0)
    x = np.insert(x,17,0)
    x = np.insert(x,18,0)
    x = np.insert(x,21,0.2)
    x = np.insert(x,23,-0.73)
    x = np.insert(x,24,0.17)
    x = np.insert(x,26,0.02)
    x = np.insert(x,27,0.27)
    x = np.insert(x,28,0.18)
    x = np.insert(x,29,0)
    x = np.insert(x,30,0)
    x = np.insert(x,31,-0.52)
    x = np.insert(x,32,0)
    x = np.insert(x,33,0.06)
    x = np.insert(x,34,0.08)
    x = np.insert(x,36,0)
    x = np.insert(x,37,0)
    x = np.insert(x,38,0)
    x = np.insert(x,42,0.38)
    x = np.insert(x,44,0)
    x = np.insert(x,45,-0.005)
    x = np.insert(x,47,-0.09)
    x = np.insert(x,48,-0.19)
    x = np.insert(x,49,-0.05)
    x = np.insert(x,51,0)
    x = np.insert(x,52,-0.003)
    x = np.insert(x,54,0.04)
    x = np.insert(x,55,-0.87)
    # define X0, growthRate, interactionMatrix
    X0 = [0.86,1,1.4,2.2,1,11.1,0.91]
    # growth rates
    r = x[0:7]
    interaction_strength = x[7:56]
    interaction_strength = pd.DataFrame(data=interaction_strength.reshape(7,7),index = species, columns=species)
    A = interaction_strength.to_numpy()
    t = np.linspace(0, 1, 5)
    second_ABC = solve_ivp(ecoNetwork, (0, 1), X0,  t_eval = t, args=(A, r), method = 'RK23')        
    # take those values and re-run for another year, adding forcings
    starting_2010 = second_ABC.y[0:7, 4:5]
    starting_values_2010 = starting_2010.flatten()
    starting_values_2010[1] = 2.0
    starting_values_2010[4] = 0.5
    # run the model for another year 2010-2011
    third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2011 = third_ABC.y[0:7, 4:5]
    starting_values_2011 = starting_2011.flatten()
    starting_values_2011[1] = 1.1
    starting_values_2011[4] = 1.3
    # run the model for 2011-2012
    fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2012 = fourth_ABC.y[0:7, 4:5]
    starting_values_2012 = starting_2012.flatten()
    starting_values_2012[1] = 1.1
    starting_values_2012[4] = 1.5
    # run the model for 2012-2013
    fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2013 = fifth_ABC.y[0:7, 4:5]
    starting_values_2013 = starting_2013.flatten()
    starting_values_2013[1] = 1.8
    starting_values_2013[4] = 0.18
    # run the model for 2011-2012
    sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2014 = sixth_ABC.y[0:7, 4:5]
    starting_values_2014 = starting_2014.flatten()
    starting_values_2014[1] = 0.6
    starting_values_2014[4] = 3
    # run the model for 2011-2012
    seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2015 = seventh_ABC.y[0:7, 4:5]
    starting_values_2015 = starting_2015.flatten()
    starting_values_2015[1] = 1.2
    starting_values_2015[4] = 0.5
    # run the model for 2011-2012
    eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t, args=(A, r), method = 'RK23')
    # take those values and re-run for another year, adding forcings
    starting_2016 = eighth_ABC.y[0:7, 4:5]
    starting_values_2016 = starting_2016.flatten()
    starting_values_2016[1] = 1.21
    starting_values_2016[4] = 0.5
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
    result = (((y[49:50, 0]-0.72)**2) +  ((y[49:50, 2]-2)**2) + ((y[49:50, 3]-4.1)**2) + ((y[49:50, 5]-28.8)**2) + ((y[49:50, 6]-0.91)**2))
    print(result)    
    return (result)

# second ODE: 
# ['arableGrass',  largeHerb, orgCarb  'roeDeer',tamworthPig,  'thornyScrub','woodland'])
#   0.72                       2          4.1                     28.8          0.91


growth_bds = ((0.1,0.25),(0.1,0.5))
interactionbds = (
                    (0.03,0.3),(0.03,0.3),
                    (0,0.25),(-1,-0.8),(0,0.25),(0,0.25),
                    (0,0.25),(0,0.25),
                    (0,0.25),(-1,-0.8),(0,0.25),(0,0.25),
                    (-0.05,-0.005),(-0.05,-0.005),
                    (-0.03,-0.003),(-0.03,-0.003)
)

# combine them into one dataframe
bds =  growth_bds + interactionbds

#L-BFGS-B, Powell, TNC, SLSQP, can have bounds
# optimization = optimize.minimize(objectiveFunction, x0 = guess, bounds = bds, method = 'L-BFGS-B', options ={'maxiter': 10000}, tol=1e-6)
optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 500)
print(optimization)


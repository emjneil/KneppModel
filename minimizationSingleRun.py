# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
import pandas as pd
from scipy.integrate import solve_ivp
import numpy as np
import itertools as IT
import string
import numpy.matlib

# store species in a list
species = ['arableGrass','largeHerb','organicCarbon','roeDeer','tamworthPig','thornyScrub','woodland']


# define the GLV function
def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-5] = 0
    # stop things from getting too high (to stop the scalar overflow error)
    X[X>1e5] = 1e5
    return X * (r + np.matmul(A, X))


# generate parameters 
def generateInteractionMatrix():
    # define the array
    interaction_matrix = [
                [-1,1,0,1,1,-1,-1],
                [1,-1,0,0,0,1,1],
                [1,1,-1,1,1,1,1],
                [1,0,0,-1,0,1,1],
                [1,0,0,0,-1,1,1],
                [2,-1,0,-1,-1,-1,-1],
                [2,-1,0,-1,-1,1,-1]
                ]
    # generate random uniform numbers
    interaction_matrix = [[np.random.uniform(-1, 0) if i < 0 else i for i in row] for row in interaction_matrix]
    interaction_matrix = [[np.random.uniform(0, 1) if i == 1 else i for i in row] for row in interaction_matrix]
    # these can be either positive or negative
    interaction_matrix = [[np.random.uniform(-1, 1) if i == 2 else i for i in row] for row in interaction_matrix]
    # return array
    return interaction_matrix


def generateGrowth():
    # generate random growth rates between 0 and 1 for each node
    growth = np.random.uniform(low=0, high=1, size=len(species))
    return growth
    

def generateX0():
    # initially scale everything to abundance of one (except species to be reintroduced)
    X0 = [1, 0, 1, 1, 0, 1, 1]
    return X0


# calculate stability
def calcJacobian(A, r, n):
    # make an empty array to fill (with diagonals = 1, zeros elsewhere since we want eigenalue)
    i_matrix = np.eye(len(n))
    # put n into an array to multiply by A
    n_array = np.matlib.repmat(n, 1, len(n))
    n_array = np.reshape (n_array, (7,7))
    # calculate
    J = i_matrix * r + A * n_array + i_matrix * np.matmul(A, n)
    return J

def calcStability(A, r, n):
    J = calcJacobian(A, r, n)
    ev = np.real(np.linalg.eig(J)[0])
    max_eig = np.max(ev)
    if max_eig < 0:
        return True
    else:
        return False


# find the middle of filters (for both ODE 1 and 2)
def objectiveFunction(): 
    # get the parameters
    A = generateInteractionMatrix()
    r = generateGrowth()
    X0 = generateX0()
    # check viability of the parameter set (is it stable?)
    ia = np.linalg.inv(A)
    # n is the equilibrium state; calc as inverse of -A*r
    n = -np.matmul(ia, r)
    # if all the values of n are above zero at equilibrium, then check the stability 
    if np.all(n > 0):
        isStable = calcStability(A, r, n)
        # if the parameter set is viable (stable & all n > 0 at equilibrium); do the calculation
        if isStable == True:
            # ODE1
            t_1 = np.linspace(0, 4, 50)
            results = solve_ivp(ecoNetwork, (0,4), X0,  t_eval = t_1, args=(A, r), method = 'RK23')
            # reshape the outputs
            y = (np.vstack(np.hsplit(results.y.reshape(len(species), 50).transpose(),1)))
            # ODE 2
            t = np.linspace(0, 1, 5)
            last_results = y[49:50,:].flatten()
            # put the reintroduced species' X0 at 1
            last_results[1] = 1
            last_results[4] = 1
            second_ABC = solve_ivp(ecoNetwork, (0, 1), last_results,  t_eval = t, args=(A, r), method = 'RK23')   
            # take those values and re-run for another year, adding forcings
            starting_2010 = second_ABC.y[0:7, 4:5]
            starting_values_2010 = starting_2010.flatten()
            starting_values_2010[1] = 2.0
            starting_values_2010[4] = 0.5
            third_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2010,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2011 = third_ABC.y[0:7, 4:5]
            starting_values_2011 = starting_2011.flatten()
            starting_values_2011[1] = 1.1
            starting_values_2011[4] = 1.3
            fourth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2011,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2012 = fourth_ABC.y[0:7, 4:5]
            starting_values_2012 = starting_2012.flatten()
            starting_values_2012[1] = 1.1
            starting_values_2012[4] = 1.5
            fifth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2012,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2013 = fifth_ABC.y[0:7, 4:5]
            starting_values_2013 = starting_2013.flatten()
            starting_values_2013[1] = 1.8
            starting_values_2013[4] = 0.18
            sixth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2013,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2014 = sixth_ABC.y[0:7, 4:5]
            starting_values_2014 = starting_2014.flatten()
            starting_values_2014[1] = 0.6
            starting_values_2014[4] = 3
            seventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2014,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2015 = seventh_ABC.y[0:7, 4:5]
            starting_values_2015 = starting_2015.flatten()
            starting_values_2015[1] = 1.2
            starting_values_2015[4] = 0.5
            eighth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2015,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2016 = eighth_ABC.y[0:7, 4:5]
            starting_values_2016 = starting_2016.flatten()
            starting_values_2016[1] = 1.21
            starting_values_2016[4] = 0.5
            ninth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2016,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2017 = ninth_ABC.y[0:7, 4:5]
            starting_values_2017 = starting_2017.flatten()
            starting_values_2017[1] = np.random.uniform(low=0.56,high=2.0)
            starting_values_2017[4] = np.random.uniform(low=0.18,high=3)
            tenth_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2017,  t_eval = t, args=(A, r), method = 'RK23')
            # take those values and re-run for another year, adding forcings
            starting_2018 = tenth_ABC.y[0:7, 4:5]
            starting_values_2018 = starting_2018.flatten()
            starting_values_2018[1] = np.random.uniform(low=0.56,high=2.0)
            starting_values_2018[4] = np.random.uniform(low=0.18,high=3)
            eleventh_ABC = solve_ivp(ecoNetwork, (0, 1), starting_values_2018,  t_eval = t, args=(A, r), method = 'RK23')
            # concatenate & append all the runs
            combined_runs = np.hstack((second_ABC.y, third_ABC.y, fourth_ABC.y, fifth_ABC.y, sixth_ABC.y, seventh_ABC.y, eighth_ABC.y, ninth_ABC.y, tenth_ABC.y, eleventh_ABC.y))
            # reshape the outputs
            y_2 = (np.vstack(np.hsplit(combined_runs.reshape(len(species), 50).transpose(),1)))
            # choose the final year (we want to compare the final year to the middle of the filters)
            result = np.array([(y[49:50, 0]-0.86)**2, (y[49:50, 2]-1.4)**2, (y[49:50, 3]-2.2)**2, (y[49:50, 5]-11.1)**2, (y[49:50, 6]-0.91)**2, (y_2[49:50, 0]-0.72)**2, (y_2[49:50, 2]-2)**2, (y_2[49:50, 3]-4.1)**2, (y_2[49:50, 5]-28.8)**2, (y_2[49:50, 6]-0.91)**2])
            print(result)  
            return (result)
        # otherwise return some high number (to stop minimizer errors)
        else:
            return 1e10
    # otherwise return some high number
    else:
        print ("not stable")
        return 1e10

objectiveFunction()
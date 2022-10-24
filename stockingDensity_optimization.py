# # ---- Approximate Bayesian Computation Model of the Knepp Estate (2000-2019) ------
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
from skopt import gp_minimize
import numpy.matlib


# # # # # --------- MINIMIZATION ----------- # # # # # # #

species = ['exmoorPony','fallowDeer','grasslandParkland','longhornCattle','organicCarbon','redDeer','roeDeer','tamworthPig','thornyScrub','woodland']


def ecoNetwork(t, X, A, r):
    # put things to zero if they go below a certain threshold
    X[X<1e-8] = 0
    return X * (r + np.matmul(A, X))
   


def objectiveFunction(x):
    # get stocking densities
    stocking_exmoorPony =  x[0]
    stocking_fallowDeer = x[1]
    stocking_longhornCattle = x[2]
    stocking_redDeer = x[3]
    stocking_tamworthPig = x[4]


    # define growth 
    r_9 = [0, 0.083, 0.94, 0.1, 0.041, 0.2, 0.14, 0.44, 0.79, 0.046]

    # define interaction strength
    A_9 = [
                # exmoor pony - special case, no growth
                [-0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # fallow deer 
                [0, -0.08, 0.31, 0, 0, 0, 0, 0, 0.0048, 0.27], #parameter set 2
                # grassland parkland
                [-0.00045, -0.00011, -0.76, -0.00013, -0.00024, 0, -0.00026, -0.0016, -0.011, -0.14], #parameter set 2
                # longhorn cattle  
                [0, 0, 0.64, -0.72, 0, 0, 0, 0, 0.032, 0.3], #parameter set 2
                # organic carbon
                [0.00019, 0.00092, 0.13, 0.0006, -0.1, 0.002, 0.0025, 0.00068, 0.0015, 0.025],  
                # red deer  
                [0, 0, 0.47, 0, 0, -0.26, 0, 0, 0.0027, 0.4],
                # roe deer 
                [0, 0, 0.83, 0, 0, 0, -0.97, 0, 0.035, 0.74],
                # tamworth pig 
                [0, 0, 0.24, 0, 0, 0, 0, -0.99, 0.0024, 0.71],  
                # thorny scrub
                [-0.027, -0.019, 0, -0.052, 0, -0.019, -0.025, -0.017, -0.012, -0.1],
                # woodland
                [-0.0036, -0.003, 0, -0.0047, 0, -0.0033, -0.0027, -0.0039, 0.00065, -0.0092]
    ]



    # EXPERIMENT 2: What is the range of parameters needed for grass to collapse?    
    # define X0 in 2021
    X0 = [(0.65*stocking_exmoorPony), (5.88*stocking_fallowDeer), 0.65, (1.53*stocking_longhornCattle), 2.2, (2.69*stocking_redDeer), 2.6, (0.95*stocking_tamworthPig), 25.3, 1.31]

    t1_experiment2 = np.linspace(2021, 2021.95, 2)
    ABC_stockingRate_2021 = solve_ivp(ecoNetwork, (2021,2021.95), X0,  t_eval = t1_experiment2, args=(A_9, r_9), method = 'RK23')
    stocking_values_2022 = ABC_stockingRate_2021.y[0:10, 1:2].flatten()
    # 2022
    stocking_values_2022[0] =  0.65 * stocking_exmoorPony
    stocking_values_2022[1] =  5.88*stocking_fallowDeer
    stocking_values_2022[3] =  1.53*stocking_longhornCattle
    stocking_values_2022[5] =  2.69*stocking_redDeer
    stocking_values_2022[7] =  0.95*stocking_tamworthPig
    t13_stockingRate = np.linspace(2022, 2022.95, 2)
    ABC_stockingRate_2022 = solve_ivp(ecoNetwork, (2022,2022.95), stocking_values_2022,  t_eval = t13_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2023 = ABC_stockingRate_2022.y[0:10, 1:2].flatten()
    # 2023
    stocking_values_2023[0] =  0.65 * stocking_exmoorPony
    stocking_values_2023[1] =  5.88*stocking_fallowDeer
    stocking_values_2023[3] =  1.53*stocking_longhornCattle
    stocking_values_2023[5] =  2.69*stocking_redDeer
    stocking_values_2023[7] =  0.95*stocking_tamworthPig
    t14_stockingRate = np.linspace(2023, 2023.95, 2)
    ABC_stockingRate_2023 = solve_ivp(ecoNetwork, (2023,2023.95), stocking_values_2023,  t_eval = t14_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2024 = ABC_stockingRate_2023.y[0:10, 1:2].flatten()
    # 2024
    stocking_values_2024[0] =  0.65 * stocking_exmoorPony
    stocking_values_2024[1] =  5.88*stocking_fallowDeer
    stocking_values_2024[3] =  1.53*stocking_longhornCattle
    stocking_values_2024[5] =  2.69*stocking_redDeer
    stocking_values_2024[7] =  0.95*stocking_tamworthPig
    t15_stockingRate = np.linspace(2024, 2024.95, 2)
    ABC_stockingRate_2024 = solve_ivp(ecoNetwork, (2024,2024.95), stocking_values_2024,  t_eval = t15_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2025 = ABC_stockingRate_2024.y[0:10, 1:2].flatten()
    # 2025
    stocking_values_2025[0] =  0.65 * stocking_exmoorPony
    stocking_values_2025[1] =  5.88*stocking_fallowDeer
    stocking_values_2025[3] =  1.53*stocking_longhornCattle
    stocking_values_2025[5] =  2.69*stocking_redDeer
    stocking_values_2025[7] =  0.95*stocking_tamworthPig
    t16_stockingRate = np.linspace(2025, 2025.95, 2)
    ABC_stockingRate_2025 = solve_ivp(ecoNetwork, (2025,2025.95), stocking_values_2025,  t_eval = t16_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2026 = ABC_stockingRate_2025.y[0:10, 1:2].flatten()
    # 2026
    stocking_values_2026[0] =  0.65 * stocking_exmoorPony
    stocking_values_2026[1] =  5.88*stocking_fallowDeer
    stocking_values_2026[3] =  1.53*stocking_longhornCattle
    stocking_values_2026[5] =  2.69*stocking_redDeer
    stocking_values_2026[7] =  0.95*stocking_tamworthPig
    t17_stockingRate = np.linspace(2026, 2026.95, 2)
    ABC_stockingRate_2026 = solve_ivp(ecoNetwork, (2026,2026.95), stocking_values_2026,  t_eval = t17_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2027 = ABC_stockingRate_2026.y[0:10, 1:2].flatten()
    # 2027
    stocking_values_2027[0] =  0.65 * stocking_exmoorPony
    stocking_values_2027[1] =  5.88*stocking_fallowDeer
    stocking_values_2027[3] =  1.53*stocking_longhornCattle
    stocking_values_2027[5] =  2.69*stocking_redDeer
    stocking_values_2027[7] =  0.95*stocking_tamworthPig
    t18_stockingRate = np.linspace(2027, 2027.95, 2)
    ABC_stockingRate_2027 = solve_ivp(ecoNetwork, (2027,2027.95), stocking_values_2027,  t_eval = t18_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2028 = ABC_stockingRate_2027.y[0:10, 1:2].flatten()
    # 2028
    stocking_values_2028[0] =  0.65 * stocking_exmoorPony
    stocking_values_2028[1] =  5.88*stocking_fallowDeer
    stocking_values_2028[3] =  1.53*stocking_longhornCattle
    stocking_values_2028[5] =  2.69*stocking_redDeer
    stocking_values_2028[7] =  0.95*stocking_tamworthPig
    t19_stockingRate = np.linspace(2028, 2028.95, 2)
    ABC_stockingRate_2028 = solve_ivp(ecoNetwork, (2028,2028.95), stocking_values_2028,  t_eval = t19_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2029 = ABC_stockingRate_2028.y[0:10, 1:2].flatten()
    # 2029
    stocking_values_2029[0] =  0.65 * stocking_exmoorPony
    stocking_values_2029[1] =  5.88*stocking_fallowDeer
    stocking_values_2029[3] =  1.53*stocking_longhornCattle
    stocking_values_2029[5] =  2.69*stocking_redDeer
    stocking_values_2029[7] =  0.95*stocking_tamworthPig
    t20_stockingRate = np.linspace(2029, 2029.95, 2)
    ABC_stockingRate_2029 = solve_ivp(ecoNetwork, (2029,2029.95), stocking_values_2029,  t_eval = t20_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2030 = ABC_stockingRate_2029.y[0:10, 1:2].flatten()
    # 2030
    stocking_values_2030[0] =  0.65 * stocking_exmoorPony
    stocking_values_2030[1] =  5.88*stocking_fallowDeer
    stocking_values_2030[3] =  1.53*stocking_longhornCattle
    stocking_values_2030[5] =  2.69*stocking_redDeer
    stocking_values_2030[7] =  0.95*stocking_tamworthPig
    t21_stockingRate = np.linspace(2030, 2030.95, 2)
    ABC_stockingRate_2030 = solve_ivp(ecoNetwork, (2030, 2030.95), stocking_values_2030,  t_eval = t21_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2031 = ABC_stockingRate_2030.y[0:10, 1:2].flatten()
    # 2031
    stocking_values_2031[0] =  0.65*stocking_exmoorPony
    stocking_values_2031[1] =  5.88*stocking_fallowDeer
    stocking_values_2031[3] =  1.53*stocking_longhornCattle
    stocking_values_2031[5] =  2.69*stocking_redDeer
    stocking_values_2031[7] =  0.95*stocking_tamworthPig
    t22_stockingRate = np.linspace(2031, 2031.95, 2)
    ABC_stockingRate_2031 = solve_ivp(ecoNetwork, (2031, 2031.95), stocking_values_2031,  t_eval = t22_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2032 = ABC_stockingRate_2031.y[0:10, 1:2].flatten()
    # 2032
    stocking_values_2032[0] =  0.65 * stocking_exmoorPony
    stocking_values_2032[1] =  5.88*stocking_fallowDeer
    stocking_values_2032[3] =  1.53*stocking_longhornCattle
    stocking_values_2032[5] =  2.69*stocking_redDeer
    stocking_values_2032[7] =  0.95*stocking_tamworthPig
    t23_stockingRate = np.linspace(2032, 2032.95, 2)
    ABC_stockingRate_2032 = solve_ivp(ecoNetwork, (2032, 2032.95), stocking_values_2032,  t_eval = t23_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2033 = ABC_stockingRate_2032.y[0:10, 1:2].flatten()
    # 2033
    stocking_values_2033[0] =  0.65 * stocking_exmoorPony
    stocking_values_2033[1] =  5.88*stocking_fallowDeer
    stocking_values_2033[3] =  1.53*stocking_longhornCattle
    stocking_values_2033[5] =  2.69*stocking_redDeer
    stocking_values_2033[7] =  0.95*stocking_tamworthPig
    t24_stockingRate = np.linspace(2033, 2033.95, 2)
    ABC_stockingRate_2033 = solve_ivp(ecoNetwork, (2033, 2033.95), stocking_values_2033,  t_eval = t24_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2034 = ABC_stockingRate_2033.y[0:10, 1:2].flatten()
    # 2034
    stocking_values_2034[0] =  0.65 * stocking_exmoorPony
    stocking_values_2034[1] =  5.88*stocking_fallowDeer
    stocking_values_2034[3] =  1.53*stocking_longhornCattle
    stocking_values_2034[5] =  2.69*stocking_redDeer
    stocking_values_2034[7] =  0.95*stocking_tamworthPig
    t25_stockingRate = np.linspace(2034, 2034.95, 2)
    ABC_stockingRate_2034 = solve_ivp(ecoNetwork, (2034, 2034.95), stocking_values_2034,  t_eval = t25_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2035 = ABC_stockingRate_2034.y[0:10, 1:2].flatten()
    # 2035
    stocking_values_2035[0] =  0.65 * stocking_exmoorPony
    stocking_values_2035[1] =  5.88*stocking_fallowDeer
    stocking_values_2035[3] =  1.53*stocking_longhornCattle
    stocking_values_2035[5] =  2.69*stocking_redDeer
    stocking_values_2035[7] =  0.95*stocking_tamworthPig
    t26_stockingRate = np.linspace(2035, 2035.95, 2)
    ABC_stockingRate_2035 = solve_ivp(ecoNetwork, (2035, 2035.95), stocking_values_2035,  t_eval = t26_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2036 = ABC_stockingRate_2035.y[0:10, 1:2].flatten()
    # 2036
    stocking_values_2036[0] =  0.65 * stocking_exmoorPony
    stocking_values_2036[1] =  5.88*stocking_fallowDeer
    stocking_values_2036[3] =  1.53*stocking_longhornCattle
    stocking_values_2036[5] =  2.69*stocking_redDeer
    stocking_values_2036[7] =  0.95*stocking_tamworthPig
    t27_stockingRate = np.linspace(2036, 2036.95, 2)
    ABC_stockingRate_2036 = solve_ivp(ecoNetwork, (2036, 2036.95), stocking_values_2036,  t_eval = t27_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2037 = ABC_stockingRate_2036.y[0:10, 1:2].flatten()
    # 2037
    stocking_values_2037[0] =  0.65 * stocking_exmoorPony
    stocking_values_2037[1] =  5.88*stocking_fallowDeer
    stocking_values_2037[3] =  1.53*stocking_longhornCattle
    stocking_values_2037[5] =  2.69*stocking_redDeer
    stocking_values_2037[7] =  0.95*stocking_tamworthPig
    t28_stockingRate = np.linspace(2037, 2037.95, 2)
    ABC_stockingRate_2037 = solve_ivp(ecoNetwork, (2037, 2037.95), stocking_values_2037,  t_eval = t28_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2038 = ABC_stockingRate_2037.y[0:10, 1:2].flatten()
    # 2038
    stocking_values_2038[0] =  0.65 * stocking_exmoorPony
    stocking_values_2038[1] =  5.88*stocking_fallowDeer
    stocking_values_2038[3] =  1.53*stocking_longhornCattle
    stocking_values_2038[5] =  2.69*stocking_redDeer
    stocking_values_2038[7] =  0.95*stocking_tamworthPig
    t29_stockingRate = np.linspace(2038, 2038.95, 2)
    ABC_stockingRate_2038 = solve_ivp(ecoNetwork, (2038, 2038.95), stocking_values_2038,  t_eval = t29_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2039 = ABC_stockingRate_2038.y[0:10, 1:2].flatten()
    # 2039
    stocking_values_2039[0] =  0.65 * stocking_exmoorPony
    stocking_values_2039[1] =  5.88*stocking_fallowDeer
    stocking_values_2039[3] =  1.53*stocking_longhornCattle
    stocking_values_2039[5] =  2.69*stocking_redDeer
    stocking_values_2039[7] =  0.95*stocking_tamworthPig
    t30_stockingRate = np.linspace(2039, 2039.95, 2)
    ABC_stockingRate_2039 = solve_ivp(ecoNetwork, (2039, 2039.95), stocking_values_2039,  t_eval = t30_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2040 = ABC_stockingRate_2039.y[0:10, 1:2].flatten()
    # 2040
    stocking_values_2040[0] =  0.65 * stocking_exmoorPony
    stocking_values_2040[1] =  5.88*stocking_fallowDeer
    stocking_values_2040[3] =  1.53*stocking_longhornCattle
    stocking_values_2040[5] =  2.69*stocking_redDeer
    stocking_values_2040[7] =  0.95*stocking_tamworthPig
    t31_stockingRate = np.linspace(2040, 2040.95, 2)
    ABC_stockingRate_2040 = solve_ivp(ecoNetwork, (2040, 2040.95), stocking_values_2040,  t_eval = t31_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2041 = ABC_stockingRate_2040.y[0:10, 1:2].flatten()
    # 2041
    stocking_values_2041[0] =  0.65 * stocking_exmoorPony
    stocking_values_2041[1] =  5.88*stocking_fallowDeer
    stocking_values_2041[3] =  1.53*stocking_longhornCattle
    stocking_values_2041[5] =  2.69*stocking_redDeer
    stocking_values_2041[7] =  0.95*stocking_tamworthPig
    t32_stockingRate = np.linspace(2041, 2041.95, 2)
    ABC_stockingRate_2041 = solve_ivp(ecoNetwork, (2041, 2041.95), stocking_values_2041,  t_eval = t32_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2042 = ABC_stockingRate_2041.y[0:10, 1:2].flatten()
    # 2042
    stocking_values_2042[0] =  0.65 * stocking_exmoorPony
    stocking_values_2042[1] =  5.88*stocking_fallowDeer
    stocking_values_2042[3] =  1.53*stocking_longhornCattle
    stocking_values_2042[5] =  2.69*stocking_redDeer
    stocking_values_2042[7] =  0.95*stocking_tamworthPig
    t33_stockingRate = np.linspace(2042, 2042.95, 2)
    ABC_stockingRate_2042 = solve_ivp(ecoNetwork, (2042, 2042.95), stocking_values_2042,  t_eval = t33_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2043 = ABC_stockingRate_2042.y[0:10, 1:2].flatten()
    # 2043
    stocking_values_2043[0] =  0.65 * stocking_exmoorPony
    stocking_values_2043[1] =  5.88*stocking_fallowDeer
    stocking_values_2043[3] =  1.53*stocking_longhornCattle
    stocking_values_2043[5] =  2.69*stocking_redDeer
    stocking_values_2043[7] =  0.95*stocking_tamworthPig
    t34_stockingRate = np.linspace(2043, 2043.95, 2)
    ABC_stockingRate_2043 = solve_ivp(ecoNetwork, (2043, 2043.95), stocking_values_2043,  t_eval = t34_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2044 = ABC_stockingRate_2043.y[0:10, 1:2].flatten()
    # 2044
    stocking_values_2044[0] =  0.65 * stocking_exmoorPony
    stocking_values_2044[1] =  5.88*stocking_fallowDeer
    stocking_values_2044[3] =  1.53*stocking_longhornCattle
    stocking_values_2044[5] =  2.69*stocking_redDeer
    stocking_values_2044[7] =  0.95*stocking_tamworthPig
    t35_stockingRate = np.linspace(2044, 2044.95, 2)
    ABC_stockingRate_2044 = solve_ivp(ecoNetwork, (2044, 2044.95), stocking_values_2044,  t_eval = t35_stockingRate, args=(A_9, r_9), method = 'RK23')
    stocking_values_2045 = ABC_stockingRate_2044.y[0:10, 1:2].flatten()
    # 2045
    stocking_values_2045[0] =  0.65 * stocking_exmoorPony
    stocking_values_2045[1] =  5.88*stocking_fallowDeer
    stocking_values_2045[3] =  1.53*stocking_longhornCattle
    stocking_values_2045[5] =  2.69*stocking_redDeer
    stocking_values_2045[7] =  0.95*stocking_tamworthPig
    t36_stockingRate = np.linspace(2045, 2046, 2)
    ABC_stockingRate_2045 = solve_ivp(ecoNetwork, (2045, 2046), stocking_values_2045,  t_eval = t36_stockingRate, args=(A_9, r_9), method = 'RK23')
    
    # concantenate the runs
    combined_runs_stockingRate = np.hstack((ABC_stockingRate_2021.y, ABC_stockingRate_2022.y, ABC_stockingRate_2023.y, ABC_stockingRate_2024.y, ABC_stockingRate_2025.y, ABC_stockingRate_2026.y, ABC_stockingRate_2027.y, ABC_stockingRate_2028.y, ABC_stockingRate_2029.y, ABC_stockingRate_2030.y, ABC_stockingRate_2031.y, ABC_stockingRate_2032.y, ABC_stockingRate_2033.y, ABC_stockingRate_2034.y, ABC_stockingRate_2035.y, ABC_stockingRate_2036.y, ABC_stockingRate_2037.y, ABC_stockingRate_2038.y, ABC_stockingRate_2039.y, ABC_stockingRate_2040.y, ABC_stockingRate_2041.y, ABC_stockingRate_2042.y, ABC_stockingRate_2043.y, ABC_stockingRate_2044.y, ABC_stockingRate_2045.y))
    combined_times_stockingRate = np.hstack((ABC_stockingRate_2021.t, ABC_stockingRate_2022.t, ABC_stockingRate_2023.t, ABC_stockingRate_2024.t, ABC_stockingRate_2025.t, ABC_stockingRate_2026.t, ABC_stockingRate_2027.t, ABC_stockingRate_2028.t, ABC_stockingRate_2029.t, ABC_stockingRate_2030.t, ABC_stockingRate_2031.t, ABC_stockingRate_2032.t, ABC_stockingRate_2033.t, ABC_stockingRate_2034.t, ABC_stockingRate_2035.t, ABC_stockingRate_2036.t, ABC_stockingRate_2037.t, ABC_stockingRate_2038.t, ABC_stockingRate_2039.t, ABC_stockingRate_2040.t, ABC_stockingRate_2041.t, ABC_stockingRate_2042.t, ABC_stockingRate_2043.t, ABC_stockingRate_2044.t, ABC_stockingRate_2045.t))
    final_df = (np.vstack(np.hsplit(combined_runs_stockingRate.reshape(len(species), 50).transpose(),1)))
    final_df = pd.DataFrame(data=final_df, columns=species)
    final_df['time'] = combined_times_stockingRate
    # select the last year, we want to see how to stop habitat from becoming homogeneous:
    #       grassland loss: < 10% (normalized to 0.13) 
    #       grass takeover: > 90% (normalized to 1.13)
    last_year_finaldf = final_df.loc[final_df['time'] == 2046]
    last_year_finaldf = last_year_finaldf.drop('time', axis=1).values.flatten()
    result = ((last_year_finaldf[2]-1.13)**2)
    print(last_year_finaldf, '\n' "r:", result)
    return (result)

    

# vary the large herbivore stocking rates 
bds =  ((0,5),(0,5),(0,5),(0,5),(0,5)) 


optimization = differential_evolution(objectiveFunction, bounds = bds, maxiter = 50)
print(optimization)
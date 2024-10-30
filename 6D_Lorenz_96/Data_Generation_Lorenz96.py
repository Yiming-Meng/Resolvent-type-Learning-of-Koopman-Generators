import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing
import itertools
import os

system = '96Lorenz'
dim = 6
M_for_1d = 10
M = M_for_1d**dim

x_lim = 1
# Generate the grid for each dimension
grid_axes = [np.linspace(-x_lim, x_lim, M_for_1d) for _ in range(dim)]

# Generate the meshgrid by taking the Cartesian product of the grid axes
mesh_grid = np.array(list(itertools.product(*grid_axes)))

span = 4
t_span = [0, span]
gen_frequency = 2000
NN = gen_frequency*span + 1
t_eval=np.linspace(0, span, NN)
frequency = 100

F = 0.1 # please also change F in the lorenz_96 function
def lorenz_96(t, X):
    F = 0.1 
    N = len(X)
    dXdt = np.zeros(N)
    for k in range(N):
        if k == 0:
            dXdt[k] = - X[k] + F
        elif k == 1:
            dXdt[k] = X[k-1] * X[k+1] - X[k] + F
        elif k == N-1:
            dXdt[k] = -X[k-2] * X[k-1] - X[k] + F
        else:
            dXdt[k] = -X[k-2] * X[k-1] + X[k-1] * X[k+1] - X[k] + F 
    return dXdt

#Definie the trajectory solving and modification function for each initial condition.
def solve_ode(initial_setup, t_span, t_eval, frequency):
    y0 = initial_setup
    solution = solve_ivp(lorenz_96, t_span, y0, t_eval=t_eval, method='Radau', dense_output=True, atol=1e-10, rtol=1e-9) 
    NN = len(t_eval)
    loc = np.arange(0, NN, round((NN-1)/span/frequency))
    data = [[solution.y[i][loc] for i in range(dim)]]
    return data

def ode_data_generator(initial_setups, t_span, t_eval, frequency):
    print('Start solving ODE')
    tic1 = time.time()
    results = pool.starmap(solve_ode, [(setup, t_span, t_eval, frequency) for setup in initial_setups])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    I = np.stack([results[i][0] for i in range(M)], axis=0)
    print('ODE solving time = {} sec'.format(time.time()-tic1))

    total_time = time.time() - tic1
    print(f"Total time for data generation: {total_time:.2f} seconds")
    return I


if __name__ == "__main__": 
    
    sample = mesh_grid.reshape(-1, dim)
    initial_setups = [[*sample[i]] for i in range(M)]
    
    #Use all available CPU cores
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    ##############################################################################
    
    flow_data = ode_data_generator(initial_setups, t_span, t_eval, frequency)
    print(sample.shape, flow_data.shape)

    subfolder = "data"
    # Check if the folder exists, if not, create it
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    filenameX = f'{system}_dim_{dim}_F_{F}_SampleData_{M_for_1d}_span_{span}_x_{x_lim}.npy'
    filenameY = f'{system}_dim_{dim}_F_{F}_FlowData_{M_for_1d}_f_{frequency}_span_{span}_x_{x_lim}.npy'
    np.save(os.path.join(subfolder, filenameX), sample)
    np.save(os.path.join(subfolder,filenameY), flow_data)
 
    ##############################################################################
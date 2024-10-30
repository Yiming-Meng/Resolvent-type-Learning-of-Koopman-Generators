import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing
import os

system = 'two_machine'
dim = 2
M_for_1d = 150
M = M_for_1d**dim

x_lim = 1
xx = np.linspace(-x_lim, x_lim, M_for_1d)
yy = np.linspace(-x_lim, x_lim, M_for_1d)
x_mesh, y_mesh = np.meshgrid(xx, yy)

span = 5
t_span = [0, span]
gen_frequency = 10000
NN = gen_frequency*span + 1
t_eval=np.linspace(0, span, NN)

delta = np.pi/3

def ode_function(t, var):
    x1, x2 = var
    return [x2, -0.5*x2 - (np.sin(x1+delta)-np.sin(delta))]

#Definie the trajectory solving and modification function for each initial condition.
def solve_ode(initial_setup, t_span, t_eval):
    y0 = initial_setup
    solution = solve_ivp(ode_function, t_span, y0, t_eval=t_eval, method='Radau', dense_output=True, atol=1e-10, rtol=1e-7) 
    data0 = solution.y[0]
    data1 = solution.y[1]
    return [[data0, data1]]

def ode_data_generator(initial_setups, t_span, t_eval):
    print('Start solving ODE')
    tic1 = time.time()
    results = pool.starmap(solve_ode, [(setup, t_span, t_eval) for setup in initial_setups])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
    
    I = np.stack([results[i][0] for i in range(M)], axis=0)
    print('ODE solving time = {} sec'.format(time.time()-tic1))

    total_time = time.time() - tic1
    print(f"Total time for data generation: {total_time:.2f} seconds")
    return I


if __name__ == "__main__": 
    
    sample = np.column_stack((x_mesh.ravel(), y_mesh.ravel()))
    initial_setups = [[*sample[i]] for i in range(M)]
    
    #Use all available CPU cores
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    ##############################################################################
    
    flow_data = ode_data_generator(initial_setups, t_span, t_eval)
    
    # Define the path for the subfolder, and save the data in the subfolder
    subfolder = "data"
    # Check if the folder exists, if not, create it
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # given an array of frequency, the data at the corresponding points are extracted
    frequency = [2, 5, 10, 20, 50, 100, 200, 500]
    
    # extract data at the corresponding points and save as numpy files
    for i in range(len(frequency)):
        loc = np.arange(0, NN, round((NN-1)/span/frequency[i]))
        filenameY = os.path.join(subfolder,f'{system}_FlowData_{frequency[i]}_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy') 
        np.save(filenameY, flow_data[:,:,loc])
 
    filenameX = os.path.join(subfolder,f'{system}_SampleData_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')
    np.save(filenameX, sample)

    print('Data saved')
    ##############################################################################
    

            

    
    
    
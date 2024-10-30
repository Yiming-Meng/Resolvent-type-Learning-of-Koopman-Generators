import numpy as np
import time
import itertools
from scipy.linalg import logm
import os

system = '96Lorenz'
dim = 6
F = 0.1
M_for_1d = 10
M = M_for_1d**dim

monomials_per_dim = 2

# Define limits and grid
x_lim = 1
span = 4
t_span = [0, span]
frequency = 10
NN = frequency*span + 1

subfolder_data = "data"

filenameX = os.path.join(subfolder_data, f'{system}_dim_{dim}_F_{F}_SampleData_{M_for_1d}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder_data, f'{system}_dim_{dim}_F_{F}_FlowData_{M_for_1d}_f_{frequency}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)
flow_data = flow_data[:,:,1]

if __name__ == "__main__": 
    print('This is for frequency: ', frequency)
    tic2 = time.time()
    monomial_count = monomials_per_dim ** dim
    
    X = np.zeros((M, monomial_count))
    Y = np.zeros((M, monomial_count))

    for j, indices in enumerate(itertools.product(range(monomials_per_dim), repeat=dim)):
        X[:, j] = np.prod(sample**indices, axis=1)
        Y[:, j] = np.prod(flow_data**indices, axis=1)

    # print the total time taken
    print(f'Total time taken for computing bases: {time.time() - tic2}')
    ##############################################################################
    delta_t = 1/frequency
    X_TX = (X.T)@ X
    X_TY = (X.T)@ Y
        
    pinv = np.linalg.pinv(X_TX)
    K = pinv @ X_TY
    L_logm = logm(K)/delta_t

    # compute L for the finite difference method
    Id = np.eye(K.shape[0])
    L_fd = (K - Id)/delta_t

    # get the weights
    logm_weights = [L_logm[:,monomials_per_dim**i] for i in range(dim)]
    logm_weights = np.array(logm_weights)

    # get the weights for the finite difference method
    fd_weights = [L_fd[:,monomials_per_dim**i] for i in range(dim)]
    fd_weights = np.array(fd_weights)


    subfolder = "results"
    np.save(os.path.join(subfolder,f'{system}_logm_weights_M_{M_for_1d}_f_{frequency}.npy'), logm_weights)
    np.save(os.path.join(subfolder,f'{system}_FD_weights_M_{M_for_1d}_f_{frequency}.npy'), fd_weights)

    ##############################################################################
    # Compare the results with the exact coefficients
    coeff_exact = np.load(f'coeff_exact_{F}.npy')

    # reverse the weights in coeff_exact to match the order of the weights in logm_weights
    coeff_exact = np.flip(coeff_exact, axis=0)

    # calculate the root mean square error 
    logm_rmse = np.linalg.norm(coeff_exact - logm_weights)/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])
    print(f'Logm method: RMSE for frequency {frequency}: {logm_rmse:.2e}')

    FD_rmse = np.linalg.norm(coeff_exact - fd_weights)/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])
    print(f'Finite difference method: RMSE for frequency {frequency}: {FD_rmse:.2e}')

    print("##########################################")
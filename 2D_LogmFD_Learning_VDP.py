import numpy as np
import time
from scipy.linalg import logm
import utils
import os

np.random.seed(42)

system = 'Van_der_Pol'
dim = 2
M_for_1d = 100
M = M_for_1d**dim

m_monomial = 3
n_monomial = 3

x_lim = 1
span = 5
t_span = [0, span]
frequency = 100
NN = frequency*span + 1

subfolder_data = "data"

filenameX = os.path.join(subfolder_data, f'{system}_SampleData_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder_data, f'{system}_FlowData_{frequency}_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)
flow_data = flow_data[:,:,1]

if __name__ == "__main__": 

    tic2 = time.time()    
    ##############################################################################

    X = np.zeros((M, m_monomial*n_monomial))
    Y = np.zeros((M, m_monomial*n_monomial))
 
    i = 0
    for n in range (n_monomial):
        for m in range(m_monomial):
            X[:, i] = np.power(sample[:, 0], m) * np.power(sample[:, 1], n)
            Y[:, i] = np.power(flow_data[:, 0], m) * np.power(flow_data[:, 1], n)
            i+=1

    # Koopman-Logm method     
    X_TX = (X.T) @ X
    X_TY = (X.T) @ Y
    
    pinv = np.linalg.pinv(X_TX)

    K = pinv @ X_TY

    delta_t = 1/frequency
    # compute L for the logm method
    L = logm(K) / delta_t

    # compute L for the finite difference method
    Id = np.eye(K.shape[0])
    L_fd = (K - Id)/delta_t

    weight_for_f1 = L[:,1]
    weight_for_f2 = L[:,m_monomial] 
    logm_weights = np.vstack((weight_for_f1, weight_for_f2))

   # save the weights for the finite difference method
    fd_weight_for_f1 = L_fd[:,1] 
    fd_weight_for_f2 = L_fd[:,m_monomial] 

    FD_weights = np.vstack((fd_weight_for_f1, fd_weight_for_f2))

    subfolder = "results"
    np.save(os.path.join(subfolder, f'{system}_logm_weights_M_{M_for_1d}_f_{frequency}_m_{m_monomial}_n_{n_monomial}.npy'), logm_weights)
    np.save(os.path.join(subfolder, f'{system}_FD_weights_M_{M_for_1d}_f_{frequency}_m_{m_monomial}_n_{n_monomial}.npy'), FD_weights)

    # functin to get the exact weights
    try:
        coeff_exact = np.load(f'{system}_coeff_exact_m_{m_monomial}_n_{n_monomial}.npy')
    except:
        coeff_exact = utils.extract_coefficients(m_monomial, n_monomial)
        np.save(f'{system}_coeff_exact_m_{m_monomial}_n_{n_monomial}.npy', coeff_exact)

    # calculate the root mean square error 
    logm_rmse = np.linalg.norm(coeff_exact - logm_weights)/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])
    print(f'Logm method: RMSE for frequency {frequency}: {logm_rmse:.2e}')

    FD_rmse = np.linalg.norm(coeff_exact - FD_weights)/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])
    print(f'Finite difference method: RMSE for frequency {frequency}: {FD_rmse:.2e}')

    print("##########################################")
    
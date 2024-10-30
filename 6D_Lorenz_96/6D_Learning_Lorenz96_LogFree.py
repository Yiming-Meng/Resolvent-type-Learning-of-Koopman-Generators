import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
import itertools
import os

system = '96Lorenz'
dim = 6
M_for_1d = 10
M = M_for_1d**dim

monomials_per_dim = 2

# Define limits and grid
x_lim = 1
grid_axes = [np.linspace(-x_lim, x_lim, M_for_1d) for _ in range(dim)]
mesh_grid = np.array(list(itertools.product(*grid_axes)))

span = 4
t_span = [0, span]
frequency = 100
NN = frequency*span + 1
t_eval=np.linspace(0, span, NN)
t_data=np.linspace(0, span, NN) 

x_gauss, w_gauss = leggauss(NN)  # Get Gauss-Legendre nodes and weights for standard interval [-1, 1]

# Transform nodes to [0, 1]
x_gauss_transformed = 0.5 * (x_gauss + 1) * (span - 0) + 0

F = 0.1

subfolder_data = "data"

filenameX = os.path.join(subfolder_data, f'{system}_dim_{dim}_F_{F}_SampleData_{M_for_1d}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder_data, f'{system}_dim_{dim}_F_{F}_FlowData_{M_for_1d}_f_{frequency}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)

if __name__ == "__main__": 
    
    monomial_count = monomials_per_dim ** dim
    y_data = np.zeros((M, NN))
    interpolated_values = np.zeros((M, NN))

    YL = np.zeros((M, monomial_count))
    YR = np.zeros((M, monomial_count))
    
    # Given a list of mu values, save the error for each mu value
    mu_values = [0.2, 0.5, 1, 2, 5, 7, 8, 10, 12, 13, 15, 20, 50, 100]
    lamda = 1e8

    print('This is for frequency: ', frequency)
    error_values = []
    for mu in mu_values:
        print('Processing mu = ', mu)
        tic2 = time.time()
        exp_term = mu**2 * np.exp(-mu * t_data)
        for j, indices in enumerate(itertools.product(range(monomials_per_dim), repeat=dim)):
            # print(f'Processing basis j={j}')
            # tic1 = time.time()
            indices = np.array(indices)
            # print(indices)
            eta_flow = np.prod(flow_data**indices[:, np.newaxis], axis=1)  # shape: (8000, 1001)
            eta_sample = np.prod(sample**indices, axis=1)  # shape: (8000,)
            y_data = exp_term * eta_flow

            interpolator = interp1d(t_data, y_data, kind='cubic', axis=1, assume_sorted=True)
            interpolated_values = interpolator(x_gauss_transformed)

            interpolated_values = np.array(interpolated_values)

            inte = np.dot(interpolated_values, w_gauss) * 0.5 * span  # Shape: (M,)
            
            YL[:, j] = inte / (mu**2) * (lamda - mu) + eta_sample
            YR[:, j] = inte / mu * lamda- lamda* eta_sample

            # # print the time taken for each basis
            # print(f'Time taken for basis {j}: {time.time() - tic1}')
        # print the total time taken
        print(f'Total time taken for computing bases: {time.time() - tic2}')
        ##############################################################################
        
        pinv_L = np.linalg.pinv(YL)
        L_update = pinv_L @ YR 

        # get the weights
        weights = [L_update[:,monomials_per_dim**i] for i in range(dim)]

        logfree_weights = np.array(weights)

        # Define the path for the subfolder
        subfolder = "results"
        # Check if the folder exists, if not, create it
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        np.save(os.path.join(subfolder, f'{system}_logfree_weights_M_{M_for_1d}_f_{frequency}_mu_{mu}_lambda_{lamda}_span_{span}.npy'), logfree_weights)
        
        # Compare the results with the exact coefficients
        coeff_exact = np.load(f'coeff_exact_{F}.npy')
        coeff_exact = np.flip(coeff_exact, axis=0)

        # calculate the root mean square error
        rmse = np.linalg.norm(coeff_exact - logfree_weights)/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])

        print(f'RMSE for frequency {frequency} with mu={mu}, lambda = {lamda:.1e}: {rmse: .2e}')
        print('#'*50)
        error_values.append(rmse)

    error_values = np.array(error_values)
    np.save(f'{system}_error_values_f_{frequency}.npy', error_values)
    # plotting scatter plot
    plt.scatter(mu_values, error_values)
    plt.xlabel(f'values $\mu$')
    plt.ylabel('RMSE')
    plt.yscale('log') 
    plt.title(f'RMSE vs Mu for f={frequency} with $\lambda$={lamda:.1e}')
    plt.show()
    plt.savefig(f'{system}_RMSE_vs_Mu({lamda:.1e})_{frequency}_span_{span}.png')
import numpy as np
import time
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import utils
import os  

system = 'Lorenz_rescaled'
dim = 3
M_for_1d = 20
M = M_for_1d**dim

m_monomial = 2
n_monomial = 2
p_monomial = 2

x_lim = 1
xx = np.linspace(-x_lim, x_lim, M_for_1d)
yy = np.linspace(-x_lim, x_lim, M_for_1d)
zz = np.linspace(-x_lim, x_lim, M_for_1d)
x_mesh, y_mesh, z_mesh = np.meshgrid(xx, yy, zz)

span = 10
t_span = [0, span]
frequency = 100
NN = frequency*span + 1
t_eval=np.linspace(0, span, NN)
t_data=np.linspace(0, span, NN) 

x_gauss, w_gauss = leggauss(NN)  # Get Gauss-Legendre nodes and weights for standard interval [-1, 1]

# Transform nodes to [0, 10]
x_gauss_transformed = 0.5 * (x_gauss + 1) * (span - 0) + 0

subfolder_data = "data"

filenameX = os.path.join(subfolder_data, f'{system}_SampleData_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder_data, f'{system}_FlowData_{frequency}_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)


if __name__ == "__main__": 
    
    y_data = np.zeros((M, NN))
    interpolated_values = np.zeros((M, NN))
    
    YL = np.zeros((M, m_monomial*n_monomial*p_monomial))
    YR = np.zeros((M, m_monomial*n_monomial*p_monomial))

    # Given a list of lambda values, save the error for each lambda value
    mu_values = [0.002, 0.02, 0.2, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 5, 6, 7, 8, 10, 12, 15, 20]
    lamda = 1e8

    print('This is for frequency: ', frequency)
    error_values = []
    for mu in mu_values:
        print('Processing mu = ', mu)
        tic2 = time.time()
        j = 0
        exp_term = mu**2 * np.exp(-mu * t_data) 
        for p in range (p_monomial):
            for n in range (n_monomial):
                for m in range(m_monomial):
                    
                    eta_flow = np.power(flow_data[:,0,:], m) * np.power(flow_data[:,1,:], n) * np.power(flow_data[:,2,:], p)  
                    eta_sample = np.power(sample[:,0], m) * np.power(sample[:,1], n) * np.power(sample[:,2], p)         

                    y_data = exp_term * eta_flow  
                
                    interpolator = interp1d(t_data, y_data, kind='cubic', axis=1, assume_sorted=True)
                    interpolated_values = interpolator(x_gauss_transformed)  
                    
                    inte = np.dot(interpolated_values, w_gauss) * 0.5 * span  
     
                    YL[:, j] = inte / (mu**2) * (lamda - mu) + eta_sample
                    YR[:, j] = inte / mu * lamda- lamda* eta_sample
                    j+=1
        print('processing time = ', time.time()-tic2)
        
        pinv_L = np.linalg.pinv(YL)
        L_update = pinv_L @ YR #@ pinv @ X.T 

        logfree_weights = np.vstack((L_update[:,1], L_update[:,m_monomial],  L_update[:,n_monomial*m_monomial]))
        
        # Define the path for the subfolder
        subfolder = "results"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        np.save(os.path.join(subfolder, f'{system}_logfree_weights_M_{M_for_1d}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}_p_{p_monomial}.npy'), logfree_weights)

        ##############################################################################
        # check if the exact weights file exists or not, if yes, load the weights
        try:
            coeff_exact = np.load(f'{system}_coeff_exact_m_{m_monomial}_n_{n_monomial}_p_{p_monomial}.npy')
        except:
            coeff_exact = utils.extract_coefficients(m_monomial, n_monomial, p_monomial)
            np.save(f'{system}_coeff_exact_m_{m_monomial}_n_{n_monomial}_p_{p_monomial}.npy', coeff_exact)

        logfree_error = np.linalg.norm(coeff_exact - logfree_weights)

        # calculate the root mean square error 
        rmse = np.linalg.norm(coeff_exact - logfree_weights)/np.sqrt(coeff_exact.shape[0]*coeff_exact.shape[1])

        print(f'RMSE for frequency {frequency} with mu={mu}, lambda = {lamda:.1e}: {rmse: .2e}')

        error_values.append([mu, rmse])
        print('#'*40)

    error_values = np.array(error_values)
    # save the error values
    np.save(f'{system}_error_values_f_{frequency}.npy', error_values)
    # plotting scatter plot
    plt.scatter(mu_values, error_values[:,1])
    plt.xlabel(f'values $\mu$')
    plt.ylabel('RMSE')
    plt.yscale('log') 
    plt.title(f'RMSE vs Mu for f={frequency} with (m={m_monomial}, n={n_monomial}, p={p_monomial}), $\lambda$={lamda:.1e}')
    plt.show()
    # save the plot
    plt.savefig(f'{system}_RMSE_vs_Mu({lamda:.1e})_{frequency} with (m={m_monomial}, n={n_monomial}, p={p_monomial}).png')
    print('#'*50)
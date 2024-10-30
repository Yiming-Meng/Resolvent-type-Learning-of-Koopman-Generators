import numpy as np
import sympy 
import time
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d
import os

np.random.seed(42)

system = 'two_machine'
dim = 2
M_for_1d = 150
M = M_for_1d**dim

x_lim = 1
span = 5
t_span = [0, span]
frequency = 100
NN = frequency*span + 1
t_eval=np.linspace(0, span, NN)
t_data=np.linspace(0, span, NN) 

m = 20 # number of basis functions

W = np.random.randn(m, dim)
b = np.random.randn(m, 1)
 
# get the basis functions
x1, x2 = sympy.symbols('x1 x2')
vars = sympy.Matrix([x1, x2])

z = W * vars + b

# Apply tanh element-wise
tanh_basis = z.applyfunc(sympy.tanh)
basis = sympy.Matrix.vstack(tanh_basis, vars)

def activation(x):
    return np.tanh(x)

x_gauss, w_gauss = leggauss(NN)  # Get Gauss-Legendre nodes and weights for standard interval [-1, 1]

# Transform nodes to [0, 10]
x_gauss_transformed = 0.5 * (x_gauss + 1) * (span - 0) + 0

subfolder_data = "data"

filenameX = os.path.join(subfolder_data, f'{system}_SampleData_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder_data, f'{system}_FlowData_{frequency}_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)

# Define groud truth dynamics
def two_machine(t, var):
    delta = np.pi/3
    x1, x2 = var
    return [x2, -0.5*x2 - (np.sin(x1+delta)-np.sin(delta))]


if __name__ == "__main__": 
    print('#'*50)
    print(f"This is for frequency: {frequency}")
    tic2 = time.time()
    y_data = np.zeros((M, NN))
    interpolated_values = np.zeros((M, NN))

    YL = np.zeros((M, m+dim))
    YR = np.zeros((M, m+dim))

    mu_values = np.concatenate((np.arange(0.5, 5, 0.5), np.arange(4, 11, 1)))
    lamda = 1e8

    error_values = []
    for mu in mu_values:
        print('Processing mu = ', mu)
        exp_term = mu**2 * np.exp(-mu * t_data)
        j = 0
        for n in range(m):
            # print('processing basis i =', j)
            W_n = W[n, :]
            b_n = b[n]
            
            # Compute the dot product between flow_data and W_n
            dot_prod = np.tensordot(flow_data, W_n, axes=([1], [0])) + b_n
            y_activation = activation(dot_prod)  # Shape: (M, NN)
            
            y_data = exp_term * y_activation  # Broadcasting over NN
  
            interpolator = interp1d(t_data, y_data, kind='cubic', axis=1, assume_sorted=True)
            interpolated_values = interpolator(x_gauss_transformed)  # Shape: (M, NN)
            
            # Compute eta_sample
            eta_sample = activation(np.dot(sample, W_n) + b_n)  # Shape: (M,)
            
            # Compute integrals
            inte = np.dot(interpolated_values, w_gauss) * 0.5 * span  # Shape: (M,)

            YL[:, j] = inte / (mu**2) * (lamda - mu) + eta_sample
            YR[:, j] = inte / mu * lamda - lamda * eta_sample
            j += 1

        # Process the first two basis functions as x and y components
        for n in range(dim):
            # print('processing basis i =', j)
            eta_sample = sample[:, n]
            y_data = exp_term * flow_data[:, n, :]
            
            # Perform interpolation
            interpolator = interp1d(t_data, y_data, kind='cubic', axis=1, assume_sorted=True)
            interpolated_values = interpolator(x_gauss_transformed)
            
            # Compute integrals
            inte = np.dot(interpolated_values, w_gauss) * 0.5 * span
            
            YL[:, j] = inte / (mu**2) * (lamda - mu) + eta_sample
            YR[:, j] = inte / mu * lamda - lamda * eta_sample
            j += 1

        print('processing time = ', time.time() - tic2)

        pinv_L = np.linalg.pinv(YL)
        L_update = pinv_L @ YR

        logfree_weights = np.vstack((L_update[:,-2], L_update[:,-1]))
        f_koopman = logfree_weights * basis
        f_koopman_func = sympy.lambdify((x1, x2), f_koopman, 'numpy')

        # Define the ODE system
        def ode_system(t, var):
            x1, x2 = var
            dxdt = f_koopman_func(x1, x2)
            return np.array(dxdt).flatten()
        tic1 = time.time()
        N = 50
        initial_point = np.random.uniform(-x_lim, x_lim, (N, dim))

        solution_free = np.zeros((N, dim, len(t_eval)))
        solution_gt = np.zeros((N, dim, len(t_eval)))

        for i in range(N):
            solution_free[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval).y
            solution_gt[i,:,:] = solve_ivp(two_machine, t_span, initial_point[i], t_eval=t_eval).y
        # print the time for solving the ode
        print('time for solving the ode =', time.time()-tic1)
        error_logfree = np.mean(np.linalg.norm(solution_free - solution_gt, axis=(1, 2)))
        # print the errors
        print(f'Error for LogFree method with mu = {mu}: {error_logfree:.2e}')
        print('#'*50)
        # save the error, if the error is greater than the previous one, terminate the loop and save the weights
        error_values.append(error_logfree)
        if len(error_values) > 1:
            if error_values[-1] > error_values[-2] and error_values[-2] < 1e-3:
                break
        # Define the path for the subfolder
        subfolder = "results"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # Define the path for the result file inside the subfolder
        weight_file = os.path.join(subfolder, f'{system}_logfree_weights_f_{frequency}_mu_{mu}_m={m}.npy')        

        np.save(weight_file, logfree_weights)

        # save the weights W and biases b to the subfolder 
        np.save(os.path.join(subfolder, f'{system}_W_f_{frequency}_mu_{mu}_m={m}.npy'), W)
        np.save(os.path.join(subfolder, f'{system}_b_f_{frequency}_mu_{mu}_m={m}.npy'), b)


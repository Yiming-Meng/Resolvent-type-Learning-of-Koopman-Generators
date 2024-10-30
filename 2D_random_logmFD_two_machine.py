import numpy as np
import sympy 
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from scipy.linalg import logm, expm
import os

np.random.seed(42)

system = 'two_machine'
dim = 2
M_for_1d = 150
M = M_for_1d**dim

# for logfree method
mu = 2
frequency = 100

# for plotting and calculating the error
x_lim = 1
span = 5
t_span = [0, span]
NN = 1000*span + 1
t_eval=np.linspace(0, span, NN)

m = 100 # number of basis functions

def activation(x):
    return np.tanh(x)


# load the weights for the logfree method from results folder 
subfolder = "results"
logfree_file = os.path.join(subfolder, f'{system}_logfree_weights_f_{frequency}_mu_{mu}_m={m}.npy')
W = np.load(os.path.join(subfolder, f'{system}_W_f_{frequency}_mu_{mu}_m={m}.npy'))
b = np.load(os.path.join(subfolder, f'{system}_b_f_{frequency}_mu_{mu}_m={m}.npy'))

subfolder2 = "data"
# load the data from the data folder
filenameX = os.path.join(subfolder2, f'{system}_SampleData_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')
filenameY = os.path.join(subfolder2, f'{system}_FlowData_{frequency}_samples_{M_for_1d}_span_{span}_x_{x_lim}.npy')

sample = np.load(filenameX)
flow_data = np.load(filenameY)
flow_data = flow_data[:,:,1]

# Define groud truth dynamics
def two_machine(t, var):
    delta = np.pi/3
    x1, x2 = var
    return [x2, -0.5*x2 - (np.sin(x1+delta)-np.sin(delta))]

if __name__ == "__main__": 
    print('#'*50)
    print(f"This is for frequency: {frequency} with {m} bases")
    tic2 = time.time()

    X = np.zeros((M, m+dim)) # m+dim
    Y = np.zeros((M, m+dim))

    error_values = []
    j = 0
    for n in range(m):
        # print('processing basis i =', j)
        W_n = W[n, :]    
        b_n = b[n]  
        # Compute eta_sample
        X[:, j] = activation(np.dot(sample, W_n) + b_n)  # Shape: (M,)
        Y[:, j] = activation(np.dot(flow_data, W_n) + b_n)  # Shape: (M, NN)
        j += 1

    # Process the first two basis functions as x and y components
    for n in range(dim):
        # print('processing basis i =', j) 
        X[:, j] = sample[:, n]           
        Y[:, j] = flow_data[:, n]         
        j += 1

    print('processing time = ', time.time() - tic2)

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

    logm_weights = L[:,-dim:].T
    fd_weights = L_fd[:,-dim:].T

    logfree_weights = np.load(logfree_file)

    # get the expression and compute the trajectories using solve_ivp
    x1, x2 = sympy.symbols('x1 x2')
    vars = sympy.Matrix([x1, x2])

    z = W * vars + b
    tanh_basis = z.applyfunc(sympy.tanh)
    identity_basis = vars
    basis = sympy.Matrix.vstack(tanh_basis, identity_basis)

    f_koopman_logfree = logfree_weights * basis
    f_koopman_logm = np.real(logm_weights) * basis
    f_koopman_logm_i = np.imag(logm_weights) * basis
    f_koopman_fd = fd_weights * basis

    f_koopman_fc_logfree = sympy.lambdify((x1, x2), f_koopman_logfree, 'numpy')
    f_koopman_fc_logm = sympy.lambdify((x1, x2), f_koopman_logm, 'numpy')
    f_koopman_fc_logm_i = sympy.lambdify((x1, x2), f_koopman_logm_i, 'numpy')
    f_koopman_fc_fd = sympy.lambdify((x1, x2), f_koopman_fd, 'numpy')
    
    # Define the ODE system
    def ode_system(t, var, f_koopman_func):
        x1, x2 = var
        dxdt = f_koopman_func(x1, x2)
        return np.array(dxdt).flatten()

    tic1 = time.time()
    N = 100
    initial_point = np.random.uniform(-x_lim, x_lim, (N, dim))

    solution_logfree = np.zeros((N, dim, len(t_eval)))
    solution_logm = np.zeros((N, dim, len(t_eval)))
    solution_logm_i = np.zeros((N, dim, len(t_eval)))
    solution_fd = np.zeros((N, dim, len(t_eval)))
    solution_gt = np.zeros((N, dim, len(t_eval)))

    for i in range(N):
        solution_logfree[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(f_koopman_fc_logfree,)).y
        solution_logm[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(f_koopman_fc_logm,)).y
        solution_logm_i[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(f_koopman_fc_logm_i,)).y
        solution_fd[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(f_koopman_fc_fd,)).y
        solution_gt[i,:,:] = solve_ivp(two_machine, t_span, initial_point[i], t_eval=t_eval).y
    # print the time for solving the ode
    print('time for solving the ode =', time.time()-tic1)
    error_logfree = np.mean(np.linalg.norm(solution_logfree - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
    error_logm = np.mean(np.linalg.norm(solution_logm - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
    error_fd = np.mean(np.linalg.norm(solution_fd - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
    # print the errors
    print('#'*50)
    print(f'Error for Logfree method: {error_logfree:.2e}')
    print(f'Error for Logm method: {error_logm:.2e}')
    print(f'Error for Finite Difference method: {error_fd:.2e}')
    print('#'*50)
    # compute the norm of solution_logm_i and print it
    imag_part = np.mean(np.linalg.norm(solution_logm_i - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
    print(f'Imaginary part of the solution for logm method: {imag_part:.2e}')

    # np.save(f'{system}_logm_weights_random_f_{frequency}_m_{m}.npy', logm_weights)
    # np.save(f'{system}_fd_weights_random_f_{frequency}_m_{m}.npy', fd_weights)

    # Plotting the results
    w = -15
    plt.figure(figsize=(8, 10))
    for i in range(dim): 
        plt.subplot(dim, 1, i+1)
        plt.plot(t_eval, solution_logfree[w, i, :], label='RTM', marker='*', markersize=1, color='r', linewidth=2) 
        plt.plot(t_eval, solution_logm[w, i, :], label='KLM', linestyle=':', color='g', linewidth=2) # 
        plt.plot(t_eval, solution_fd[w, i, :], label='FDM', linestyle='--', color='blue', linewidth=2)
        plt.plot(t_eval, solution_gt[w, i, :], label='Ground Truth', linestyle='-.', color='k', linewidth=2)
        plt.xlabel('Time (t)', fontsize=14)
        if i == 0:
            plt.ylabel('$x_1(t)$', fontsize=16)
        else:
            plt.ylabel('$x_2(t)$', fontsize=16)
        # set the fontsize of the axis
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.grid(True)
        if i == 0:  # Display the legend only in the first subplot
            plt.legend()
            # set the fontsize of the legend
            plt.legend(fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    # plt.suptitle('Comparison with ground truth for two machine with random basis', fontsize=16)
    plt.savefig(f'trajectory_{system}_comparisons_f_{frequency}_m={m}.png',dpi=400)
    plt.show()
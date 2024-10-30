import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
import os

np.random.seed(42)

dim = 6
system = '96Lorenz'
xlim = 1
M_for_1d = 10
frequency = 100

# for logfree method
mu = 5
lamda = 1e8
t_int = 4

# load the weights for the logfree method from results folder 
subfolder = "results"
logfree_file = os.path.join(subfolder, f'{system}_logfree_weights_M_{M_for_1d}_f_{frequency}_mu_{mu}_lambda_{lamda}_span_{t_int}.npy')

# Load the matrix 
Logfree_M = np.load(logfree_file) 
print(Logfree_M.shape)

Logm_M = np.load(os.path.join(subfolder, f'{system}_logm_weights_M_{M_for_1d}_f_{frequency}.npy'))

FD_M = np.load(os.path.join(subfolder, f'{system}_FD_weights_M_{M_for_1d}_f_{frequency}.npy'))

X = np.array([sp.symbols(f'X{i+1}') for i in range(dim)])
F = 0.1

# Define the basis functions explicitly for the 6-dimensional system
def basis_functions(X):
    X1, X2, X3, X4, X5, X6 = X
    return np.array([
        1,
        X6,
        X5,
        X5*X6,
        X4,
        X4*X6,
        X4*X5,
        X4*X5*X6,
        X3,
        X3*X6,
        X3*X5,
        X3*X5*X6,
        X3*X4,
        X3*X4*X6,
        X3*X4*X5,
        X3*X4*X5*X6,
        X2,
        X2*X6,
        X2*X5,
        X2*X5*X6,
        X2*X4,
        X2*X4*X6,
        X2*X4*X5,
        X2*X4*X5*X6,
        X2*X3,
        X2*X3*X6,
        X2*X3*X5,
        X2*X3*X5*X6,
        X2*X3*X4,
        X2*X3*X4*X6,
        X2*X3*X4*X5,
        X2*X3*X4*X5*X6,
        X1,
        X1*X6,
        X1*X5,
        X1*X5*X6,
        X1*X4,
        X1*X4*X6,
        X1*X4*X5,
        X1*X4*X5*X6,
        X1*X3,
        X1*X3*X6,
        X1*X3*X5,
        X1*X3*X5*X6,
        X1*X3*X4,
        X1*X3*X4*X6,
        X1*X3*X4*X5,
        X1*X3*X4*X5*X6,
        X1*X2,
        X1*X2*X6,
        X1*X2*X5,
        X1*X2*X5*X6,
        X1*X2*X4,
        X1*X2*X4*X6,
        X1*X2*X4*X5,
        X1*X2*X4*X5*X6,
        X1*X2*X3,
        X1*X2*X3*X6,
        X1*X2*X3*X5,
        X1*X2*X3*X5*X6,
        X1*X2*X3*X4,
        X1*X2*X3*X4*X6,
        X1*X2*X3*X4*X5,
        X1*X2*X3*X4*X5*X6,
    ])

# Define the ODE system
def ode_system(t, X, weights):
    basis = basis_functions(X)
    dXdt = weights @ basis
    # reverse the order of the output from 6 to 1 to match the ground truth
    dXdt = dXdt[::-1]
    return dXdt

# Ground truth ODE function
def Lorenz96(t, X):
    N = len(X)
    dXdt = np.zeros(N)
    dXdt[0] = - X[0] + F
    dXdt[1] = X[0] * X[2] - X[1] + F
    dXdt[-1] = -X[-3] * X[-2] - X[-1] + F
    for k in range(2, N-1):
        dXdt[k] = -X[k-2] * X[k-1] + X[k-1] * X[k+1] - X[k] + F 
    return dXdt

# sample 100 initial points drawn from [-xlim, xlim] randomly
N = 100
initial_point = np.random.uniform(-xlim, xlim, (N, dim))
# Time span for the ODE solution
t_span = (0, t_int) 
t_eval = np.linspace(0, t_span[-1], 10001) 

# define the solutions with zeros for the odes
solution = np.zeros((N, dim, len(t_eval)))
solution_log = np.zeros((N, dim, len(t_eval)))
solution_fd = np.zeros((N, dim, len(t_eval)))
solution_gt = np.zeros((N, dim, len(t_eval)))

tic = time.time()
# with a for loop for the initial points, solve the odes
for i in range(N):
    # Solve the ODE
    solution[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(Logfree_M,)).y
    solution_log[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(Logm_M,)).y
    solution_fd[i,:,:] = solve_ivp(ode_system, t_span, initial_point[i], t_eval=t_eval, args=(FD_M,)).y
    solution_gt[i,:,:] = solve_ivp(Lorenz96, t_span, initial_point[i], t_eval=t_eval).y

# compare the solutions with the ground truth in norm and print the errors
error_logfree = np.mean(np.linalg.norm(solution - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
error_log = np.mean(np.linalg.norm(solution_log - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
error_fd = np.mean(np.linalg.norm(solution_fd - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))

# print the errors
print('#'*50)
print(f"This is for frequency: {frequency} with mu = {mu} and lambda = {lamda}")
print(f'Error for LogFree method: {error_logfree:.2e}')
print(f'Error for Log method: {error_log:.2e}')
print(f'Error for FD method: {error_fd:.2e}')

print('#'*50)
print('Time for the whole process: ', time.time()-tic)

# Plotting the results
plt.figure(figsize=(8, 10))
for i in range(dim): 
    plt.subplot(dim, 1, i+1)
    plt.plot(t_eval, solution[-1, i, :], label='Koopman-LogFree', color='black') 
    plt.plot(t_eval, solution_log[-1, i, :], label='Koopman-Log', linestyle=':', color='green')
    plt.plot(t_eval, solution_fd[-1, i, :], label='Koopman-FD', linestyle='--', color='blue')
    plt.plot(t_eval, solution_gt[-1, i, :], label='Ground Truth', linestyle='-.', color='red',)
    plt.xlabel('Time (t)', fontsize=14)
    plt.ylabel(f'$x_{i+1}(t)$', fontsize=14)
    # set larger font size for the x and y axis
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    if i == 0: 
        plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.suptitle(f'Comparison with ground truth for Lorenz96 ({frequency}Hz)', fontsize=14)
plt.savefig(f'trajectory_{system}_comparisons_f_{frequency}.png',dpi=300)
plt.show()
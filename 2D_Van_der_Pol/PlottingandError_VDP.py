import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import utils
import time
import os

dim = 2
system = 'Van_der_Pol'
xlim = 1
frequency = 10
mu = 2
M_for_1d = 100
M = M_for_1d**dim
m_monomial = 3
n_monomial = 3

# for logfree method
mu = 3
lamda = 1e8
frequency = 10

# load the weights for the logfree method from results folder 
subfolder = "results"
logfree_file = os.path.join(subfolder, f'{system}_logfree_weights_M_{M_for_1d}_f_{frequency}_mu_{mu}_lambda_{lamda}_m_{m_monomial}_n_{n_monomial}.npy')

# Load the matrix 
Logfree_M = np.load(logfree_file) 
print(Logfree_M.shape)

Logm_M = np.load(os.path.join(subfolder, f'{system}_logm_weights_M_{M_for_1d}_f_{frequency}_m_{m_monomial}_n_{n_monomial}.npy'))

FD_M = np.load(os.path.join(subfolder, f'{system}_FD_weights_M_{M_for_1d}_f_{frequency}_m_{m_monomial}_n_{n_monomial}.npy'))

# Define the ODE system
def ode_system(t, XY, weights):
    X1, X2 = XY
    basis_functions = utils.extract_bases(m_monomial, n_monomial)
    Basis = np.array([basis.subs({sp.symbols('X1'): X1, sp.symbols('X2'): X2}) for basis in basis_functions], dtype=np.float64)
    dxdt = weights @ Basis
    return dxdt

# Ground truth ODE function
def Lorenz(t, var):
    mu = 1.0
    x1, x2 = var
    return [-x2, x1 - mu * (1 - x1**2) * x2] 

# sample 100 initial points drawn from [-xlim, xlim] randomly
N = 100
initial_point = np.random.uniform(-xlim, xlim, (N, dim))
# Time span for the ODE solution
t_span = (0, 5)  # Example time span from 0 to 10
t_eval = np.linspace(0, t_span[-1], 50001)  # Points at which to store the solution

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
    solution_gt[i,:,:] = solve_ivp(Lorenz, t_span, initial_point[i], t_eval=t_eval).y

# compare the solutions with the ground truth in norm and print the errors
error_logfree = np.mean(np.linalg.norm(solution - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
error_log = np.mean(np.linalg.norm(solution_log - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))
error_fd = np.mean(np.linalg.norm(solution_fd - solution_gt, axis=(1, 2))/np.sqrt(len(t_eval)))

# print the errors
print('#'*50)
print(f"This is for frequency: {frequency}")
print(f'Error for LogFree method: {error_logfree:.2e}')
print(f'Error for Log method: {error_log:.2e}')
print(f'Error for FD method: {error_fd:.2e}')

print('#'*50)
print('Time for the whole process: ', time.time()-tic)

# Plotting the results
plt.figure(figsize=(8, 10))
for i in range(dim):  # Adjust this loop for the 6-dimensional system
    plt.subplot(dim, 1, i+1)
    plt.plot(t_eval, solution[-1, i, :], label='Koopman-LogFree', color='black') 
    plt.plot(t_eval, solution_log[-1, i, :], label='Koopman-Log', linestyle=':', color='green')
    plt.plot(t_eval, solution_fd[-1, i, :], label='Koopman-FD', linestyle='--', color='blue')
    plt.plot(t_eval, solution_gt[-1, i, :], label='Ground Truth', linestyle='-.', color='red',)
    plt.xlabel('Time (t)', fontsize=14)
    # if i == 0:
    #     plt.ylabel('$x_1$(t)', fontsize=14)
    # else:
    #     plt.ylabel('$x_2$(t)', fontsize=14)
    plt.ylabel(f'$x_{i+1}(t)$', fontsize=14)
    # plt.legend()
    plt.grid(True)
    if i == 0:  # Display the legend only in the first subplot
        plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.suptitle('Comparison with ground truth for VDP (m={m_monomial}, n={n_monomial})', fontsize=14)
plt.savefig(f'trajectory_{system}_comparisons_f_{frequency} with (m={m_monomial}, n={n_monomial}).png',dpi=300)
plt.show()
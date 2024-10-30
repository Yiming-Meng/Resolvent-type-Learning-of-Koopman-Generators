import sympy as sp
import numpy as np
import itertools

n = 6
X = np.array([sp.symbols(f'X{i+1}') for i in range(n)])
F = 0.1
monomials_per_dim = 2 

# Define the ODE system symbolically
dXdt = [None] * n
dXdt[0] = - X[0] + F
dXdt[1] = X[0] * X[2] - X[1] + F
dXdt[-1] = -X[-3] * X[-2] - X[-1] + F

for k in range(2, n-1):
    dXdt[k] = -X[k-2] * X[k-1] + X[k-1] * X[k+1] - X[k] + F 

# Generate all basis functions using itertools.product
basis_functions = []
for indices in itertools.product(range(monomials_per_dim), repeat=n):
    basis_function = sp.prod([X[d]**indices[d] for d in range(n)])
    basis_functions.append(basis_function)
    
# Initialize a matrix to store coefficients
coeff_matrix = sp.zeros(n, monomials_per_dim**n)

# Extract coefficients
for i in range(n):
    expanded_expr = sp.expand(dXdt[i])
    for j, basis in enumerate(basis_functions):
        # Get the coefficient of the exact monomial (basis function) in the expression
        coeff = expanded_expr.as_coefficients_dict().get(basis, 0)
        coeff_matrix[i, j] = coeff

# Convert to a numpy array if needed
coeff_matrix_np = np.array(coeff_matrix).astype(float)

# Save the coefficients to a file
np.save(f'coeff_exact_{F}.npy', coeff_matrix_np)
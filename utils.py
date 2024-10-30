import sympy as sp
import numpy as np

# define a function to extract the coefficients of the exact monomials
def extract_coefficients(m_monomial, n_monomial):
    # Define the number of monomials per dimension
    dim = 2
    X = np.array([sp.symbols(f'X{i+1}') for i in range(dim)])

    # Define the ODE system symbolically
    dXdt = [None] * dim
    mu = 1.0
    dXdt[0] = - X[1]
    dXdt[1] = X[0] - mu * (1 - X[0]**2) * X[1]

    # Generate all basis functions using itertools.product
    basis_functions = []
    for n in range (n_monomial):
        for m in range(m_monomial):
            basis_function = X[0]**m * X[1]**n
            basis_functions.append(basis_function)
        
    # Initialize a matrix to store coefficients
    coeff_matrix = sp.zeros(dim, m_monomial*n_monomial)

    # Extract coefficients
    for i in range(dim):
        expanded_expr = sp.expand(dXdt[i])
        for j, basis in enumerate(basis_functions):
            coeff = expanded_expr.as_coefficients_dict().get(basis, 0)
            coeff_matrix[i, j] = coeff

    coeff_matrix_np = np.array(coeff_matrix).astype(float)
    return coeff_matrix_np

# a function to extract the expressions of the bases
def extract_bases(m_monomial, n_monomial):
    dim = 2
    X = np.array([sp.symbols(f'X{i+1}') for i in range(dim)])

    basis_functions = []
    for n in range (n_monomial):
        for m in range(m_monomial):
            basis_function = X[0]**m * X[1]**n
            basis_functions.append(basis_function)
    
    return basis_functions

if __name__ == "__main__":

    m_monomial = 2
    n_monomial = 2
    p_monomial = 2
    
    # Extract the coefficients of the exact monomials
    coeff_matrix_np = extract_coefficients(m_monomial, n_monomial)
  
    # Save the coefficients to a file
    np.save(f'coeff_exact_m_{m_monomial}_n_{n_monomial}.npy', coeff_matrix_np)
    